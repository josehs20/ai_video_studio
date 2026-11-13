"""Cliente resiliente da API do ComfyUI para gerar as cenas do projeto.

Lê `config/projeto.json`, injeta prompt/seed/resolução no `workflow_api.json`
e aguarda a conclusão real via polling em `/history/{prompt_id}` (sem
`time.sleep` cego). O clipe bruto é copiado para `inputs/projeto_atual/` e o
status da cena passa para `"gerado"`.

Uso:
    python3 scripts/gerador_sequencia.py
    python3 scripts/gerador_sequencia.py --cena cena_03 --forcar
    python3 scripts/gerador_sequencia.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    STATUS_ERRO,
    STATUS_GERADO,
    STATUS_PENDENTE,
    atualizar_cena,
    caminho_relativo,
    carregar_json,
    carregar_projeto,
    configurar_logging,
    diretorios,
    exigir_arquivo_integro,
    ffprobe_info,
    obter_cena,
    resolver,
    salvar_projeto,
)

CLASSES_AMOSTRADOR = ("SamplerCustom", "KSampler", "KSamplerAdvanced")
CLASSES_TEXTO = ("CLIPTextEncode", "CLIPTextEncodeSDXL", "T5TextEncode")


def _requisicao_json(requisicao: request.Request, timeout: float) -> dict[str, Any]:
    """Executa uma requisição HTTP e devolve o JSON, com erros legíveis."""
    try:
        with request.urlopen(requisicao, timeout=timeout) as resposta:
            return json.loads(resposta.read().decode("utf-8") or "{}")
    except error.HTTPError as erro:
        detalhe = erro.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"ComfyUI respondeu HTTP {erro.code}: {detalhe[:1500]}"
        ) from erro
    except error.URLError as erro:
        raise RuntimeError(
            f"ComfyUI inacessível ({erro.reason}). "
            "Inicie o servidor com --lowvram e tente novamente."
        ) from erro


def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    return _requisicao_json(request.Request(url, method="GET"), timeout)


def post_json(url: str, dados: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    corpo = json.dumps(dados).encode("utf-8")
    requisicao = request.Request(
        url,
        data=corpo,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _requisicao_json(requisicao, timeout)


def _nos_por_classe(grafo: dict[str, Any], *classes: str) -> list[str]:
    """Ids dos nós cujo `class_type` está entre as classes informadas."""
    return [no_id for no_id, no in grafo.items() if str(no.get("class_type")) in classes]


def _localizar_texto(
    grafo: dict[str, Any],
    referencia: Any,
    papel: str,
    profundidade: int = 0,
) -> str | None:
    """Segue a cadeia de conditioning até achar o nó que codifica o texto."""
    if profundidade > 6 or not isinstance(referencia, list) or len(referencia) != 2:
        return None
    no_id = str(referencia[0])
    no = grafo.get(no_id)
    if not no:
        return None
    if str(no.get("class_type")) in CLASSES_TEXTO:
        return no_id
    entradas = no.get("inputs") or {}
    for chave in (papel, "conditioning"):
        proximo = entradas.get(chave)
        if proximo is None:
            continue
        encontrado = _localizar_texto(grafo, proximo, papel, profundidade + 1)
        if encontrado:
            return encontrado
    return None


def injetar_parametros(
    workflow: dict[str, Any],
    cena: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Devolve uma cópia do grafo com os parâmetros da cena aplicados."""
    grafo = copy.deepcopy(workflow)
    geracao = config.get("geracao") or {}
    frames = int(cena.get("frames") or geracao.get("frames", 33))
    fps = float(geracao.get("fps", 12))
    largura = int(cena.get("largura") or geracao.get("largura", 512))
    altura = int(cena.get("altura") or geracao.get("altura", 320))

    # Latente de vídeo: resolução leve, compatível com 6 GB de VRAM.
    for no_id in _nos_por_classe(grafo, "EmptyLTXVLatentVideo", "EmptyLatentImage"):
        entradas = grafo[no_id]["inputs"]
        entradas["width"] = largura
        entradas["height"] = altura
        if "length" in entradas:
            entradas["length"] = frames
        elif "batch_size" in entradas:
            entradas["batch_size"] = 1

    for no_id in _nos_por_classe(grafo, "LTXVConditioning"):
        grafo[no_id]["inputs"]["frame_rate"] = fps

    for no_id in _nos_por_classe(grafo, "LTXVScheduler"):
        grafo[no_id]["inputs"]["steps"] = int(cena.get("steps") or geracao.get("steps", 25))

    for no_id in _nos_por_classe(grafo, "KSamplerSelect"):
        grafo[no_id]["inputs"]["sampler_name"] = str(
            cena.get("sampler") or geracao.get("sampler", "euler")
        )

    positivo = cena.get("prompt")
    negativo = cena.get("negative_prompt") or geracao.get("prompt_negativo_padrao") or ""
    for no_id in _nos_por_classe(grafo, *CLASSES_AMOSTRADOR):
        entradas = grafo[no_id]["inputs"]
        if "noise_seed" in entradas:
            entradas["noise_seed"] = int(cena.get("seed", 0))
        elif "seed" in entradas:
            entradas["seed"] = int(cena.get("seed", 0))
        if "cfg" in entradas:
            entradas["cfg"] = float(cena.get("cfg") or geracao.get("cfg", 3.0))
        for papel, texto in (("positive", positivo), ("negative", negativo)):
            if texto is None:
                continue
            alvo = _localizar_texto(grafo, entradas.get(papel), papel)
            if alvo:
                grafo[alvo]["inputs"]["text"] = str(texto)

    # ReActor: liga/desliga a troca facial por cena.
    for no_id in _nos_por_classe(grafo, "ReActorFaceSwap", "ReActorFaceSwapOpt"):
        grafo[no_id]["inputs"]["enabled"] = bool(cena.get("aplicar_reactor", False))

    # Nó de saída: prefixo por cena facilita localizar o arquivo no /history.
    for no_id in _nos_por_classe(
        grafo, "SaveAnimatedWEBP", "SaveVideo", "VHS_VideoCombine", "SaveImage"
    ):
        entradas = grafo[no_id]["inputs"]
        if "filename_prefix" in entradas:
            entradas["filename_prefix"] = str(cena["id"])
        if "fps" in entradas:
            entradas["fps"] = fps

    for no_id in _nos_por_classe(grafo, "CreateVideo"):
        grafo[no_id]["inputs"]["fps"] = fps

    return grafo



def sincronizar_referencia(
    config: dict[str, Any],
    raiz: Path,
    cenas: list[dict[str, Any]],
) -> Path | None:
    """Copia a referência facial para `ComfyUI/input/` quando o ReActor é usado."""
    if not any(c.get("aplicar_reactor") for c in cenas):
        return None
    referencia = config.get("referencia_facial")
    if not referencia:
        raise ValueError("aplicar_reactor=true, mas 'referencia_facial' não está no config")
    origem = exigir_arquivo_integro(
        resolver(raiz, referencia), minimo_bytes=512, descricao="referência facial"
    )
    pasta_input = (config.get("comfyui") or {}).get("input_dir", "")
    if not pasta_input:
        raise ValueError("config.comfyui.input_dir não definido (necessário para o ReActor)")
    destino_dir = resolver(raiz, pasta_input)
    try:
        destino_dir.mkdir(parents=True, exist_ok=True)
    except OSError as erro:
        raise RuntimeError(f"Não foi possível preparar {destino_dir}: {erro}") from erro
    destino = destino_dir / origem.name
    if not destino.exists() or destino.stat().st_size != origem.stat().st_size:
        shutil.copy2(origem, destino)
    return destino


def enviar_prompt(url: str, workflow: dict[str, Any], timeout: float = 60.0) -> str:
    """Enfileira o grafo no ComfyUI e devolve o `prompt_id`."""
    resposta = post_json(
        f"{url}/prompt",
        {"prompt": workflow, "client_id": "ai_video_studio"},
        timeout=timeout,
    )
    prompt_id = resposta.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI não devolveu prompt_id: {json.dumps(resposta)[:800]}")
    return str(prompt_id)


def _mensagens_de_erro(status: dict[str, Any]) -> str:
    mensagens: list[str] = []
    for mensagem in status.get("messages") or []:
        if not isinstance(mensagem, (list, tuple)) or len(mensagem) < 2:
            continue
        tipo, conteudo = mensagem[0], mensagem[1]
        if tipo in {"execution_error", "execution_interrupted"} and isinstance(conteudo, dict):
            detalhe = conteudo.get("exception_message") or conteudo.get("message") or ""
            no = conteudo.get("node_type") or conteudo.get("node_id") or "?"
            mensagens.append(f"{tipo} em {no}: {str(detalhe).strip()}")
    return "; ".join(mensagens) or "erro sem detalhes no histórico"


def aguardar_prompt(
    url: str,
    prompt_id: str,
    timeout_s: float,
    intervalo: float,
    logger: Any,
    max_ausencias: int = 30,
) -> dict[str, Any]:
    """Aguarda a execução real consultando `/history` e `/queue`."""
    inicio = time.monotonic()
    ausencias = 0
    ultimo_log = -30.0
    while True:
        decorrido = time.monotonic() - inicio
        if decorrido > timeout_s:
            raise TimeoutError(
                f"prompt {prompt_id} não concluiu em {timeout_s:.0f}s "
                "(verifique /history e os logs do ComfyUI)"
            )
        historico = get_json(f"{url}/history/{prompt_id}", timeout=30.0)
        registro = historico.get(prompt_id)
        if registro:
            status = registro.get("status") or {}
            if str(status.get("status_str")) == "error":
                raise RuntimeError(f"ComfyUI falhou: {_mensagens_de_erro(status)}")
            if status.get("completed") or registro.get("outputs"):
                return registro
        else:
            ausencias += 1
            if ausencias >= max_ausencias:
                fila = get_json(f"{url}/queue", timeout=30.0)
                em_execucao = [item[1] for item in (fila.get("queue_running") or [])]
                pendentes = [item[1] for item in (fila.get("queue_pending") or [])]
                if prompt_id not in em_execucao and prompt_id not in pendentes:
                    raise RuntimeError(
                        f"prompt {prompt_id} não está mais na fila nem no histórico "
                        "(pode ter sido interrompido; reinicie a geração)"
                    )
                ausencias = 0
        if decorrido - ultimo_log >= 30.0:
            logger.info("  ... aguardando %s (%.0fs)", prompt_id, decorrido)
            ultimo_log = decorrido
        time.sleep(intervalo)


def extrair_arquivo_saida(registro: dict[str, Any], pasta_saida: Path) -> Path:
    """Localiza o primeiro arquivo produzido pelos nós de saída do ComfyUI."""
    candidatos: list[dict[str, Any]] = []
    for conteudo in (registro.get("outputs") or {}).values():
        if not isinstance(conteudo, dict):
            continue
        for itens in conteudo.values():
            if not isinstance(itens, list):
                continue
            for item in itens:
                if isinstance(item, dict) and item.get("filename"):
                    candidatos.append(item)
    if not candidatos:
        raise RuntimeError(
            "Nenhum arquivo de saída no histórico do ComfyUI "
            f"({sorted((registro.get('outputs') or {}).keys())})"
        )
    escolhido = next(
        (c for c in candidatos if str(c.get("type", "output")) == "output"), candidatos[0]
    )
    caminho = pasta_saida / str(escolhido.get("subfolder") or "") / str(escolhido["filename"])
    return exigir_arquivo_integro(caminho, minimo_bytes=1024, descricao="saída do ComfyUI")



def gerar_cena(
    cena: dict[str, Any],
    config: dict[str, Any],
    raiz: Path,
    projeto_path: Path,
    template: dict[str, Any],
    url: str,
    logger: Any,
    intervalo: float,
    timeout_s: float,
) -> Path:
    """Gera uma cena, copia o clipe bruto e atualiza o status."""
    cena_id = str(cena["id"])
    if not cena.get("prompt"):
        raise ValueError(f"{cena_id}: campo 'prompt' vazio")

    grafo = injetar_parametros(template, cena, config)
    if not _nos_por_classe(grafo, *CLASSES_TEXTO):
        raise RuntimeError(
            "workflow_api.json não possui nó de texto (CLIPTextEncode). "
            "Rode: python3 scripts/validar_ambiente.py --escrever-workflow"
        )

    sincronizar_referencia(config, raiz, [cena])
    prompt_id = enviar_prompt(url, grafo)
    logger.info(
        "%s enviada (prompt_id=%s, seed=%s, reactor=%s)",
        cena_id,
        prompt_id,
        cena.get("seed"),
        bool(cena.get("aplicar_reactor")),
    )
    registro = aguardar_prompt(url, prompt_id, timeout_s, intervalo, logger)

    pasta_saida = (config.get("comfyui") or {}).get("output_dir")
    if not pasta_saida:
        raise ValueError("config.comfyui.output_dir não definido")
    arquivo = extrair_arquivo_saida(registro, resolver(raiz, pasta_saida))

    pasta_raw = diretorios(config, raiz)["raw"]
    pasta_raw.mkdir(parents=True, exist_ok=True)
    destino = pasta_raw / f"{cena_id}_raw{arquivo.suffix.lower()}"
    shutil.copy2(arquivo, destino)
    exigir_arquivo_integro(destino, minimo_bytes=1024, descricao=f"clipe bruto {cena_id}")

    detalhes: dict[str, Any] = {}
    try:
        detalhes = ffprobe_info(destino, logger)
    except RuntimeError as erro:
        logger.warning("%s: ffprobe não leu o bruto (%s)", cena_id, erro)

    atualizar_cena(
        config,
        cena_id,
        status=STATUS_GERADO,
        arquivo_raw=caminho_relativo(raiz, destino),
        prompt_id=prompt_id,
        seed=cena.get("seed"),
        erro=None,
        info_raw=detalhes,
    )
    salvar_projeto(config, projeto_path)
    logger.info(
        "%s gerada: %s (%.1f KB)", cena_id, destino.name, destino.stat().st_size / 1024
    )
    return destino


def selecionar_cenas(
    config: dict[str, Any],
    alvos: list[str],
    forcar: bool,
) -> list[dict[str, Any]]:
    """Seleciona cenas por id ou por status (pendente/erro por padrão)."""
    if alvos:
        return [obter_cena(config, alvo) for alvo in alvos]
    if forcar:
        return list(config.get("cenas", []))
    return [
        cena
        for cena in config.get("cenas", [])
        if cena.get("status") in (STATUS_PENDENTE, STATUS_ERRO)
    ]



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera cenas via API do ComfyUI")
    parser.add_argument("--projeto", default="config/projeto.json")
    parser.add_argument("--cena", action="append", default=[], help="Id da cena (repetível)")
    parser.add_argument("--forcar", action="store_true", help="Refaz mesmo se já gerada")
    parser.add_argument("--dry-run", action="store_true", help="Valida a injeção sem enviar")
    parser.add_argument("--url", default=None, help="Sobrescreve config.comfyui.url")
    parser.add_argument("--parar-no-erro", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logger = configurar_logging(args.log_level)
    config, projeto_path, raiz = carregar_projeto(args.projeto)
    geracao = config.get("geracao") or {}
    url = str(
        args.url or (config.get("comfyui") or {}).get("url") or "http://127.0.0.1:8188"
    ).rstrip("/")
    intervalo = float(geracao.get("intervalo_polling_segundos", 2.0))
    timeout_s = float(geracao.get("timeout_segundos", 3600))

    workflow_path = resolver(
        raiz, config.get("workflow_api", "scripts/workflow_api.json")
    )
    if not workflow_path.is_file():
        logger.error(
            "workflow_api.json ausente em %s. Rode: "
            "python3 scripts/validar_ambiente.py --escrever-workflow",
            workflow_path,
        )
        return 1
    template = carregar_json(workflow_path)

    cenas = selecionar_cenas(config, args.cena, args.forcar)
    if not cenas:
        logger.info("Nenhuma cena para gerar (todas já estão além de 'pendente').")
        return 0

    logger.info("%d cena(s) na fila: %s", len(cenas), ", ".join(c["id"] for c in cenas))

    if args.dry_run:
        for cena in cenas:
            grafo = injetar_parametros(template, cena, config)
            amostradores = _nos_por_classe(grafo, *CLASSES_AMOSTRADOR)
            seeds = [
                grafo[no]["inputs"].get("noise_seed", grafo[no]["inputs"].get("seed"))
                for no in amostradores
            ]
            latentes = _nos_por_classe(grafo, "EmptyLTXVLatentVideo")
            if latentes:
                entradas = grafo[latentes[0]]["inputs"]
                dimensoes = (
                    f"{entradas.get('width')}x{entradas.get('height')}"
                    f"@{entradas.get('length')}f"
                )
            else:
                dimensoes = "latente não encontrado"
            logger.info(
                "[dry-run] %s | %s | seeds=%s | reactor=%s",
                cena["id"],
                dimensoes,
                seeds,
                bool(cena.get("aplicar_reactor")),
            )
        return 0

    try:
        get_json(f"{url}/system_stats", timeout=10.0)
    except RuntimeError as erro:
        logger.error("%s", erro)
        return 1

    falhas = 0
    for cena in cenas:
        try:
            gerar_cena(
                cena,
                config,
                raiz,
                projeto_path,
                template,
                url,
                logger,
                intervalo,
                timeout_s,
            )
        except (RuntimeError, ValueError, TimeoutError, FileNotFoundError) as erro:
            falhas += 1
            logger.error("Falha em %s: %s", cena.get("id"), erro)
            atualizar_cena(
                config, str(cena["id"]), status=STATUS_ERRO, erro=str(erro)[:1000]
            )
            salvar_projeto(config, projeto_path)
            if args.parar_no_erro:
                break

    if falhas:
        logger.warning("Geração finalizada com %d falha(s).", falhas)
        return 1
    logger.info("Geração concluída com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

