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
    ajustar_frames_ltx,
    atualizar_cena,
    avaliar_candidato,
    avaliar_orcamento_vram,
    caminho_relativo,
    carregar_json,
    carregar_projeto,
    configurar_logging,
    diretorios,
    exigir_arquivo_integro,
    ffprobe_info,
    inspecionar_imagem,
    nome_referencia_no_input,
    obter_cena,
    resolver,
    salvar_projeto,
    sincronizar_para_input,
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


def _primeiro_no(grafo: dict[str, Any], *classes: str) -> str | None:
    """Id do primeiro nó encontrado entre as classes informadas."""
    ids = _nos_por_classe(grafo, *classes)
    return ids[0] if ids else None


def _novo_no_id(grafo: dict[str, Any]) -> str:
    """Próximo id numérico livre (mantém o formato de string do grafo API)."""
    numeros = [int(no_id) for no_id in grafo if str(no_id).isdigit()]
    return str(max(numeros or [0]) + 1)


def _referencias(grafo: dict[str, Any], no_id: str) -> int:
    """Quantas entradas de outros nós apontam para `no_id`."""
    alvo = str(no_id)
    total = 0
    for no in grafo.values():
        for valor in (no.get("inputs") or {}).values():
            if isinstance(valor, list) and len(valor) == 2 and str(valor[0]) == alvo:
                total += 1
    return total


def _no_referencia_visual(grafo: dict[str, Any]) -> str | None:
    """`LoadImage` que não alimenta o ReActor — ou seja, o print da cena."""
    usados: set[str] = set()
    for no_id in _nos_por_classe(grafo, "ReActorFaceSwap", "ReActorFaceSwapOpt"):
        origem = (grafo[no_id].get("inputs") or {}).get("source_image")
        if isinstance(origem, list) and origem:
            usados.add(str(origem[0]))
    for no_id in _nos_por_classe(grafo, "LoadImage"):
        if no_id not in usados:
            return no_id
    return None


def _frames_da_cena(cena: dict[str, Any], geracao: dict[str, Any]) -> int:
    """Frames efetivos da cena, sempre normalizados para 8k+1 (LTX-Video)."""
    return ajustar_frames_ltx(int(cena.get("frames") or geracao.get("frames", 33)))


def _forca_referencia(cena: dict[str, Any], geracao: dict[str, Any]) -> float:
    """Força do condicionamento por imagem (0.0–1.0), por cena ou global."""
    valor = cena.get("forca_referencia")
    if valor is None:
        valor = geracao.get("forca_referencia", 0.65)
    return min(max(float(valor), 0.0), 1.0)


def _garantir_ramo_img2video(
    grafo: dict[str, Any],
) -> tuple[str, str | None]:
    """Devolve (nó LTXVImgToVideo, nó LoadImage da referência), criando-os se preciso."""
    existente = _primeiro_no(grafo, "LTXVImgToVideo")
    if existente:
        return existente, _no_referencia_visual(grafo)

    cond = _primeiro_no(grafo, "LTXVConditioning")
    amostrador = _primeiro_no(grafo, *CLASSES_AMOSTRADOR)
    if cond is None or amostrador is None:
        raise RuntimeError(
            "workflow_api.json sem LTXVConditioning/SamplerCustom: regenere com "
            "python3 scripts/validar_ambiente.py --escrever-workflow"
        )
    entradas_cond = grafo[cond]["inputs"]
    positivo = entradas_cond.get("positive")
    negativo = entradas_cond.get("negative")
    vae = None
    for no_id in _nos_por_classe(grafo, "VAEDecode"):
        vae = (grafo[no_id].get("inputs") or {}).get("vae") or vae
    if not positivo or not negativo or not vae:
        raise RuntimeError(
            "não foi possível derivar conditioning/VAE do grafo para o modo imagem"
        )

    no_imagem = _no_referencia_visual(grafo)
    if no_imagem is None:
        no_imagem = _novo_no_id(grafo)
        grafo[no_imagem] = {
            "class_type": "LoadImage",
            "inputs": {"image": "print_tela.png"},
        }
    i2v = _novo_no_id(grafo)
    grafo[i2v] = {
        "class_type": "LTXVImgToVideo",
        "inputs": {
            "positive": positivo,
            "negative": negativo,
            "vae": vae,
            "image": [no_imagem, 0],
            "width": 512,
            "height": 320,
            "length": 33,
            "batch_size": 1,
            "strength": 0.65,
        },
    }
    return i2v, no_imagem


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


def _aplicar_img_to_video(
    grafo: dict[str, Any],
    cena: dict[str, Any],
    config: dict[str, Any],
    nome_imagem_ref: str | None,
) -> None:
    """Liga o ramo Image-to-Video (LoadImage -> LTXVImgToVideo -> sampler)."""
    geracao = config.get("geracao") or {}
    i2v, no_imagem = _garantir_ramo_img2video(grafo)
    if no_imagem is None:
        raise RuntimeError("workflow sem nó LoadImage disponível para a referência da cena")
    largura = int(cena.get("largura") or geracao.get("largura", 512))
    altura = int(cena.get("altura") or geracao.get("altura", 320))
    frames = _frames_da_cena(cena, geracao)

    if nome_imagem_ref:
        grafo[no_imagem]["inputs"]["image"] = str(nome_imagem_ref)
    grafo[i2v]["inputs"].update(
        width=largura,
        height=altura,
        length=frames,
        batch_size=1,
        strength=_forca_referencia(cena, geracao),
    )

    cond = _primeiro_no(grafo, "LTXVConditioning")
    if cond:
        grafo[cond]["inputs"]["positive"] = [i2v, 0]
        grafo[cond]["inputs"]["negative"] = [i2v, 1]
    scheduler = _primeiro_no(grafo, "LTXVScheduler")
    if scheduler:
        grafo[scheduler]["inputs"]["latent"] = [i2v, 2]
    for no_id in _nos_por_classe(grafo, *CLASSES_AMOSTRADOR):
        entradas = grafo[no_id]["inputs"]
        if "latent_image" in entradas:
            entradas["latent_image"] = [i2v, 2]
    # O latente vazio só é usado no fluxo Text-to-Video.
    for no_id in _nos_por_classe(grafo, "EmptyLTXVLatentVideo"):
        del grafo[no_id]


def _aplicar_text_to_video(
    grafo: dict[str, Any],
    cena: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Fallback Text-to-Video puro (quando `imagem_referencia` é null)."""
    geracao = config.get("geracao") or {}
    i2v = _primeiro_no(grafo, "LTXVImgToVideo")
    positivo = negativo = None
    if i2v:
        entradas = grafo[i2v]["inputs"]
        positivo = entradas.get("positive")
        negativo = entradas.get("negative")
        del grafo[i2v]
    origem_visual = _no_referencia_visual(grafo)
    if origem_visual:
        del grafo[origem_visual]

    largura = int(cena.get("largura") or geracao.get("largura", 512))
    altura = int(cena.get("altura") or geracao.get("altura", 320))
    frames = _frames_da_cena(cena, geracao)
    latente = _primeiro_no(grafo, "EmptyLTXVLatentVideo")
    if latente is None:
        latente = _novo_no_id(grafo)
        grafo[latente] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": largura,
                "height": altura,
                "length": frames,
                "batch_size": 1,
            },
        }
    grafo[latente]["inputs"].update(width=largura, height=altura, length=frames)

    cond = _primeiro_no(grafo, "LTXVConditioning")
    if cond:
        entradas = grafo[cond]["inputs"]
        positivo = positivo or entradas.get("positive")
        negativo = negativo or entradas.get("negative")
        if positivo:
            entradas["positive"] = positivo
        if negativo:
            entradas["negative"] = negativo
    scheduler = _primeiro_no(grafo, "LTXVScheduler")
    if scheduler:
        grafo[scheduler]["inputs"]["latent"] = [latente, 0]
    for no_id in _nos_por_classe(grafo, *CLASSES_AMOSTRADOR):
        entradas = grafo[no_id]["inputs"]
        if "latent_image" in entradas:
            entradas["latent_image"] = [latente, 0]


def _aplicar_reactor(grafo: dict[str, Any], cena: dict[str, Any]) -> None:
    """Liga/desliga o ReActor. Desligado = bypass total (sem tocar nos frames)."""
    ativo = bool(cena.get("aplicar_reactor", False))
    reactores = _nos_por_classe(grafo, "ReActorFaceSwap", "ReActorFaceSwapOpt")
    for no_id in reactores:
        grafo[no_id]["inputs"]["enabled"] = ativo
    if ativo or not reactores:
        return

    destino: list[Any] | None = None
    fonte_id: str | None = None
    for no_id in reactores:
        entradas = grafo[no_id].get("inputs") or {}
        referencia_entrada = entradas.get("input_image")
        if isinstance(referencia_entrada, list) and referencia_entrada:
            destino = list(referencia_entrada)
        referencia_fonte = entradas.get("source_image")
        if isinstance(referencia_fonte, list) and referencia_fonte:
            fonte_id = str(referencia_fonte[0])
        del grafo[no_id]
    if destino is None:
        return
    for no in grafo.values():
        for chave, valor in (no.get("inputs") or {}).items():
            if isinstance(valor, list) and len(valor) == 2 and str(valor[0]) in reactores:
                no["inputs"][chave] = destino
    if fonte_id and _referencias(grafo, fonte_id) == 0:
        grafo.pop(fonte_id, None)


def injetar_parametros(
    workflow: dict[str, Any],
    cena: dict[str, Any],
    config: dict[str, Any],
    nome_imagem_ref: str | None = None,
    sufixo_saida: str | None = None,
) -> dict[str, Any]:
    """Devolve uma cópia do grafo com os parâmetros da cena aplicados.

    Com `imagem_referencia` na cena o grafo roda em Image-to-Video
    (`LoadImage` -> `LTXVImgToVideo`); sem ela, cai para Text-to-Video puro.
    `nome_imagem_ref` é o nome do arquivo já copiado para `ComfyUI/input/`.
    """
    grafo = copy.deepcopy(workflow)
    geracao = config.get("geracao") or {}
    frames = _frames_da_cena(cena, geracao)
    fps = float(cena.get("fps") or geracao.get("fps", 12))
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

    # Imagem de referência da cena x fallback Text-to-Video puro.
    referencia = cena.get("imagem_referencia")
    if referencia:
        _aplicar_img_to_video(grafo, cena, config, nome_imagem_ref)
    else:
        _aplicar_text_to_video(grafo, cena, config)

    # ReActor: liga/desliga a troca facial por cena (desligado = bypass real).
    _aplicar_reactor(grafo, cena)

    # Nó de saída: prefixo por cena/seed facilita localizar o arquivo no /history.
    prefixo = str(cena["id"])
    if sufixo_saida:
        prefixo = f"{prefixo}_{sufixo_saida}"
    for no_id in _nos_por_classe(
        grafo, "SaveAnimatedWEBP", "SaveVideo", "VHS_VideoCombine", "SaveImage"
    ):
        entradas = grafo[no_id]["inputs"]
        if "filename_prefix" in entradas:
            entradas["filename_prefix"] = prefixo
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


def sincronizar_referencia_cena(
    config: dict[str, Any],
    raiz: Path,
    cena: dict[str, Any],
    logger: Any = None,
) -> str | None:
    """Copia a imagem de referência da cena para `ComfyUI/input/`.

    Devolve o nome do arquivo como o nó `LoadImage` deve referenciá-lo, ou
    `None` quando a cena não tem `imagem_referencia` (modo Text-to-Video).
    """
    referencia = cena.get("imagem_referencia")
    if not referencia:
        return None
    origem = resolver(raiz, referencia)
    inspecionar_imagem(origem, logger)
    pasta_input = (config.get("comfyui") or {}).get("input_dir", "")
    if not pasta_input:
        raise ValueError(
            "config.comfyui.input_dir não definido "
            "(necessário para a imagem de referência das cenas)"
        )
    nome = nome_referencia_no_input(str(cena["id"]), origem)
    destino = sincronizar_para_input(origem, resolver(raiz, pasta_input), nome)
    if logger:
        logger.info("%s: referência visual pronta em %s", cena["id"], destino)
    return nome


def nome_previsto_referencia(cena: dict[str, Any]) -> str | None:
    """Nome de arquivo que a referência terá em `ComfyUI/input/` (para dry-run)."""
    referencia = cena.get("imagem_referencia")
    if not referencia:
        return None
    return nome_referencia_no_input(str(cena["id"]), Path(str(referencia)))


def duracao_esperada(cena: dict[str, Any], config: dict[str, Any]) -> float:
    """Duração em segundos implícita em `frames / fps` da cena."""
    geracao = config.get("geracao") or {}
    fps = float(cena.get("fps") or geracao.get("fps", 12))
    if fps <= 0:
        return 0.0
    return _frames_da_cena(cena, geracao) / fps


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
    timeout_requisicao: float = 120.0,
    max_ausencias: int = 30,
    max_falhas: int = 20,
) -> dict[str, Any]:
    """Aguarda a execução real consultando `/history` e `/queue`.

    Timeouts isolados de requisição são tratados como transitórios: o ComfyUI
    bloqueia o event loop durante nós pesados (ex.: ReActor), então o polling
    precisa sobreviver a janelas sem resposta.
    """
    inicio = time.monotonic()
    ausencias = 0
    falhas = 0
    ultimo_log = -30.0
    while True:
        decorrido = time.monotonic() - inicio
        if decorrido > timeout_s:
            raise TimeoutError(
                f"prompt {prompt_id} não concluiu em {timeout_s:.0f}s "
                "(verifique /history e os logs do ComfyUI)"
            )
        try:
            historico = get_json(
                f"{url}/history/{prompt_id}", timeout=timeout_requisicao
            )
        except (RuntimeError, TimeoutError, OSError) as erro:
            falhas += 1
            logger.debug(
                "consulta ao /history falhou (%d/%d): %s", falhas, max_falhas, erro
            )
            if falhas >= max_falhas:
                raise RuntimeError(
                    f"ComfyUI parou de responder ao /history após {falhas} tentativas: {erro}"
                ) from erro
            time.sleep(intervalo)
            continue
        falhas = 0

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
                try:
                    fila = get_json(f"{url}/queue", timeout=timeout_requisicao)
                except (RuntimeError, TimeoutError, OSError) as erro:
                    logger.debug("consulta ao /queue falhou: %s", erro)
                    ausencias = 0
                else:
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



def _renderizar_seed(
    cena: dict[str, Any],
    config: dict[str, Any],
    raiz: Path,
    template: dict[str, Any],
    url: str,
    logger: Any,
    intervalo: float,
    timeout_s: float,
    timeout_requisicao: float,
    seed: int,
    destino_base: Path,
    sufixo_saida: str | None = None,
) -> dict[str, Any]:
    """Envia uma variação (seed) da cena e devolve o clipe copiado localmente."""
    variacao = dict(cena)
    variacao["seed"] = int(seed)
    nome_ref = sincronizar_referencia_cena(config, raiz, variacao, logger)
    grafo = injetar_parametros(
        template,
        variacao,
        config,
        nome_imagem_ref=nome_ref,
        sufixo_saida=sufixo_saida,
    )
    if not _nos_por_classe(grafo, *CLASSES_TEXTO):
        raise RuntimeError(
            "workflow_api.json não possui nó de texto (CLIPTextEncode). "
            "Rode: python3 scripts/validar_ambiente.py --escrever-workflow"
        )
    sincronizar_referencia(config, raiz, [variacao])
    prompt_id = enviar_prompt(url, grafo)
    logger.info(
        "%s: seed %s enviada (prompt_id=%s, modo=%s, ref=%s, reactor=%s)",
        cena["id"],
        seed,
        prompt_id,
        "img2video" if nome_ref else "text2video",
        nome_ref or "-",
        bool(cena.get("aplicar_reactor")),
    )
    registro = aguardar_prompt(
        url, prompt_id, timeout_s, intervalo, logger, timeout_requisicao
    )

    pasta_saida = (config.get("comfyui") or {}).get("output_dir")
    if not pasta_saida:
        raise ValueError("config.comfyui.output_dir não definido")
    arquivo = extrair_arquivo_saida(registro, resolver(raiz, pasta_saida))
    destino = destino_base.with_suffix(arquivo.suffix.lower())
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arquivo, destino)
    exigir_arquivo_integro(
        destino, minimo_bytes=1024, descricao=f"clipe bruto {cena['id']} (seed {seed})"
    )
    return {"seed": int(seed), "prompt_id": prompt_id, "arquivo": destino}


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
    timeout_requisicao: float = 120.0,
    candidatos: int | None = None,
    ignorar_limite_vram: bool = False,
) -> Path:
    """Gera N candidatos por cena, valida e promove o melhor para `_raw`."""
    cena_id = str(cena["id"])
    if not cena.get("prompt"):
        raise ValueError(f"{cena_id}: campo 'prompt' vazio")
    geracao = config.get("geracao") or {}
    largura = int(cena.get("largura") or geracao.get("largura", 512))
    altura = int(cena.get("altura") or geracao.get("altura", 320))
    frames = _frames_da_cena(cena, geracao)
    steps = int(cena.get("steps") or geracao.get("steps", 25))
    avisos = avaliar_orcamento_vram(largura, altura, frames, steps)
    if avisos:
        for aviso in avisos:
            logger.warning("%s: %s", cena_id, aviso)
        if not ignorar_limite_vram:
            raise ValueError(
                f"{cena_id}: pode estourar 6 GB de VRAM ({'; '.join(avisos)}). "
                "Ajuste largura<=512/altura<=320/frames<=49 ou use "
                "--ignorar-limite-vram se souber o que está fazendo."
            )

    quantidade = max(
        1,
        int(
            candidatos
            or cena.get("candidatos_por_cena")
            or geracao.get("candidatos_por_cena")
            or 1
        ),
    )
    seeds = [int(cena.get("seed") or 0) + indice for indice in range(quantidade)]
    pastas = diretorios(config, raiz)
    pasta_raw = pastas["raw"]
    pasta_raw.mkdir(parents=True, exist_ok=True)
    pasta_cand = pastas["temp"] / "candidatos" / cena_id
    esperado = duracao_esperada(cena, config)
    registros: list[dict[str, Any]] = []

    logger.info(
        "%s: %d candidato(s), %dx%d@%df, ref=%s",
        cena_id,
        quantidade,
        largura,
        altura,
        frames,
        cena.get("imagem_referencia") or "-",
    )
    for indice, seed in enumerate(seeds, start=1):
        logger.info("%s: candidato %d/%d (seed %d)", cena_id, indice, quantidade, seed)
        if quantidade > 1:
            destino_base = pasta_cand / f"seed_{seed}"
            sufixo = f"s{seed}"
        else:
            destino_base = pasta_raw / f"{cena_id}_raw"
            sufixo = None
        resultado = _renderizar_seed(
            cena,
            config,
            raiz,
            template,
            url,
            logger,
            intervalo,
            timeout_s,
            timeout_requisicao,
            seed,
            destino_base,
            sufixo,
        )
        avaliacao = avaliar_candidato(resultado["arquivo"], logger, esperado or None)
        registro = {
            "seed": int(seed),
            "arquivo": caminho_relativo(raiz, resultado["arquivo"]),
            "prompt_id": resultado["prompt_id"],
            "valido": avaliacao["valido"],
            "pontuacao": avaliacao["pontuacao"],
            "motivos": avaliacao["motivos"],
            "info": avaliacao.get("info"),
            "metricas": avaliacao.get("metricas"),
        }
        registros.append(registro)
        atualizar_cena(config, cena_id, candidatos=registros)
        salvar_projeto(config, projeto_path)
        logger.info(
            "%s: seed %d válido=%s pontuação=%.3f%s",
            cena_id,
            seed,
            registro["valido"],
            registro["pontuacao"],
            "" if registro["valido"] else f" — {'; '.join(registro['motivos'])}",
        )


    validos = [registro for registro in registros if registro["valido"]]
    if not validos:
        resumo = " | ".join(
            f"seed {r['seed']}: {'; '.join(r['motivos'])}" for r in registros
        )
        raise RuntimeError(
            f"{cena_id}: nenhum dos {len(registros)} candidato(s) passou na "
            f"validação ({resumo})"
        )
    escolhido = max(validos, key=lambda r: (r["pontuacao"], -int(r["seed"])))
    origem = resolver(raiz, escolhido["arquivo"])
    destino = pasta_raw / f"{cena_id}_raw{origem.suffix.lower()}"
    if origem.resolve() != destino.resolve():
        shutil.copy2(origem, destino)
    exigir_arquivo_integro(destino, minimo_bytes=1024, descricao=f"clipe bruto {cena_id}")
    for registro in registros:
        registro["escolhido"] = int(registro["seed"]) == int(escolhido["seed"])

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
        prompt_id=escolhido["prompt_id"],
        seed=escolhido["seed"],
        seed_escolhida=escolhido["seed"],
        candidatos=registros,
        erro=None,
        info_raw=detalhes,
    )
    salvar_projeto(config, projeto_path)
    logger.info(
        "%s gerada: seed %s de %d (%.1f KB) -> %s",
        cena_id,
        escolhido["seed"],
        len(registros),
        destino.stat().st_size / 1024,
        destino.name,
    )
    return destino


def escolher_candidato(
    cena: dict[str, Any],
    seed_alvo: int,
    config: dict[str, Any],
    raiz: Path,
    projeto_path: Path,
    logger: Any,
) -> Path:
    """Promove um candidato já gerado (`temp/candidatos/<cena>/seed_X`) a `_raw`."""
    cena_id = str(cena["id"])
    pastas = diretorios(config, raiz)
    pasta_cand = pastas["temp"] / "candidatos" / cena_id
    encontrados = sorted(
        arquivo
        for arquivo in pasta_cand.glob(f"seed_{int(seed_alvo)}.*")
        if arquivo.is_file()
    )
    if not encontrados:
        raise FileNotFoundError(
            f"{cena_id}: candidato da seed {seed_alvo} não encontrado em {pasta_cand}"
        )
    origem = encontrados[0]
    avaliacao = avaliar_candidato(
        origem, logger, duracao_esperada(cena, config) or None
    )
    if not avaliacao["valido"]:
        logger.warning(
            "%s: seed %s tem ressalvas: %s",
            cena_id,
            seed_alvo,
            "; ".join(avaliacao["motivos"]),
        )

    pasta_raw = pastas["raw"]
    pasta_raw.mkdir(parents=True, exist_ok=True)
    destino = pasta_raw / f"{cena_id}_raw{origem.suffix.lower()}"
    if origem.resolve() != destino.resolve():
        shutil.copy2(origem, destino)
    exigir_arquivo_integro(destino, minimo_bytes=1024, descricao=f"clipe bruto {cena_id}")

    registros = [
        dict(registro)
        for registro in (cena.get("candidatos") or [])
        if int(registro.get("seed", -1)) != int(seed_alvo)
    ]
    registros.append(
        {
            "seed": int(seed_alvo),
            "arquivo": caminho_relativo(raiz, origem),
            "valido": avaliacao["valido"],
            "pontuacao": avaliacao["pontuacao"],
            "motivos": avaliacao["motivos"],
            "info": avaliacao.get("info"),
            "metricas": avaliacao.get("metricas"),
        }
    )
    registros.sort(key=lambda registro: int(registro.get("seed") or 0))
    for registro in registros:
        registro["escolhido"] = int(registro.get("seed") or 0) == int(seed_alvo)

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
        seed=int(seed_alvo),
        seed_escolhida=int(seed_alvo),
        candidatos=registros,
        erro=None,
        info_raw=detalhes,
    )
    salvar_projeto(config, projeto_path)
    logger.info("%s: seed %s promovida para %s", cena_id, seed_alvo, destino.name)
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
    parser.add_argument(
        "--candidatos",
        type=int,
        default=None,
        help="Candidatos por cena (sobrescreve candidatos_por_cena do config)",
    )
    parser.add_argument(
        "--escolher-seed",
        type=int,
        default=None,
        help="Promove um candidato já gerado (temp/candidatos) sem re-renderizar",
    )
    parser.add_argument(
        "--ignorar-limite-vram",
        action="store_true",
        help="Permite configurações acima do orçamento seguro de VRAM",
    )
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
    timeout_requisicao = float(geracao.get("timeout_requisicao_segundos", 120))

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

    if args.escolher_seed is not None:
        if args.dry_run:
            logger.error("--escolher-seed não combina com --dry-run")
            return 2
        if args.cena:
            alvos_cenas = [obter_cena(config, alvo) for alvo in args.cena]
        else:
            pasta_cand = diretorios(config, raiz)["temp"] / "candidatos"
            padrao = f"seed_{int(args.escolher_seed)}.*"
            alvos_cenas = [
                cena
                for cena in config.get("cenas", [])
                if any(
                    arquivo.is_file()
                    for arquivo in (pasta_cand / str(cena["id"])).glob(padrao)
                )
            ]
            if not alvos_cenas:
                logger.error(
                    "Nenhuma cena tem candidato da seed %s em temp/candidatos. "
                    "Use --cena para trocar por uma seed existente.",
                    args.escolher_seed,
                )
                return 1
            logger.info(
                "Seed %s encontrada em: %s",
                args.escolher_seed,
                ", ".join(str(cena["id"]) for cena in alvos_cenas),
            )
        if not alvos_cenas:
            logger.error(
                "Nenhum candidato em temp/candidatos para promover. "
                "Rode com --candidatos N primeiro."
            )
            return 1
        falhas_escolha = 0
        for cena in alvos_cenas:
            try:
                escolher_candidato(
                    cena, args.escolher_seed, config, raiz, projeto_path, logger
                )
            except (RuntimeError, ValueError, FileNotFoundError) as erro:
                falhas_escolha += 1
                logger.error("%s: %s", cena.get("id"), erro)
        return 1 if falhas_escolha else 0

    cenas = selecionar_cenas(config, args.cena, args.forcar)
    if not cenas:
        logger.info("Nenhuma cena para gerar (todas já estão além de 'pendente').")
        return 0

    logger.info("%d cena(s) na fila: %s", len(cenas), ", ".join(c["id"] for c in cenas))

    if args.dry_run:
        geracao_dry = config.get("geracao") or {}
        for cena in cenas:
            nome_ref = nome_previsto_referencia(cena)
            grafo = injetar_parametros(
                template, cena, config, nome_imagem_ref=nome_ref
            )
            amostradores = _nos_por_classe(grafo, *CLASSES_AMOSTRADOR)
            seeds = [
                grafo[no]["inputs"].get("noise_seed", grafo[no]["inputs"].get("seed"))
                for no in amostradores
            ]
            referencias = _nos_por_classe(grafo, "LTXVImgToVideo")
            if referencias:
                entradas = grafo[referencias[0]]["inputs"]
            else:
                latentes = _nos_por_classe(grafo, "EmptyLTXVLatentVideo")
                entradas = (
                    grafo[latentes[0]]["inputs"]
                    if latentes
                    else {}
                )
            dimensoes = (
                f"{entradas.get('width')}x{entradas.get('height')}"
                f"@{entradas.get('length')}f"
            )
            avisos = avaliar_orcamento_vram(
                int(entradas.get("width") or 0),
                int(entradas.get("height") or 0),
                int(entradas.get("length") or 0),
                int(cena.get("steps") or geracao_dry.get("steps", 25)),
            )
            quantidade = max(
                1,
                int(
                    args.candidatos
                    or cena.get("candidatos_por_cena")
                    or geracao_dry.get("candidatos_por_cena")
                    or 1
                ),
            )
            logger.info(
                "[dry-run] %s | %s | modo=%s | ref=%s | seeds=%s | "
                "força=%.2f | reactor=%s | candidatos=%d",
                cena["id"],
                dimensoes,
                "img2video" if nome_ref else "text2video",
                cena.get("imagem_referencia") or "-",
                seeds,
                float(cena.get("forca_referencia") or geracao_dry.get("forca_referencia", 0.65)),
                bool(cena.get("aplicar_reactor")),
                quantidade,
            )
            for aviso in avisos:
                logger.warning("[dry-run] %s: %s", cena["id"], aviso)
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
                timeout_requisicao,
                candidatos=args.candidatos,
                ignorar_limite_vram=args.ignorar_limite_vram,
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

