"""Orquestrador do pipeline de vídeo (máquina de estados por cena).

Etapas:
    gerar    -> pendente/erro -> gerado  (ComfyUI, N candidatos por cena)
    escolher -> promove um candidato de `temp/candidatos/{cena}/seed_X` a `_raw`
    polir    -> gerado -> polido  (RIFE/upscale)
    montar   -> polido -> outputs/{nome}_final.mp4 (passe único FFmpeg)

`gerar` já promove automaticamente o melhor candidato; `escolher` é a via
manual (`--escolher-seed N`) para trocar a escolha sem re-renderizar.

A lógica de montagem é exposta como funções puras (`construir_comando_montagem`,
`planejar_montagem`) para reuso futuro na camada FastAPI.

Uso:
    python3 scripts/orchestrator.py --etapa tudo
    python3 scripts/orchestrator.py --etapa gerar --candidatos 3
    python3 scripts/orchestrator.py --etapa escolher --escolher-seed 445 --refazer-cena cena_04
    python3 scripts/orchestrator.py --etapa montar --transicao fade
    python3 scripts/orchestrator.py --refazer-cena cena_03 --nova-seed 777
    python3 scripts/orchestrator.py --etapa status
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gerador_sequencia  # noqa: E402
import polimento  # noqa: E402
from common import (  # noqa: E402
    STATUS_APROVADO,
    STATUS_PENDENTE,
    STATUS_POLIDO,
    atualizar_cena,
    caminho_relativo,
    carregar_projeto,
    configurar_logging,
    diretorios,
    exigir_arquivo_integro,
    executar,
    ffprobe_info,
    obter_cena,
    preparar_diretorio,
    resolver,
    salvar_projeto,
)

ETAPAS = ("tudo", "gerar", "escolher", "polir", "montar", "status")


def planejar_montagem(
    config: dict[str, Any],
    raiz: Path,
    transicao: str | None = None,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    """Seleciona os clipes polidos e devolve (arquivos, metadados, montagem)."""
    montagem = dict(config.get("montagem") or {})
    if transicao:
        montagem["transicao"] = transicao
    modo = str(montagem.get("transicao", "corte"))
    if modo not in ("corte", "fade"):
        raise ValueError(f"transição inválida: {modo} (use 'corte' ou 'fade')")

    arquivos: list[Path] = []
    metadados: list[dict[str, Any]] = []
    faltando: list[str] = []
    for cena in config.get("cenas", []):
        if cena.get("status") not in (STATUS_POLIDO, STATUS_APROVADO):
            faltando.append(f"{cena.get('id')} (status={cena.get('status')})")
            continue
        caminho = cena.get("arquivo_polido")
        if not caminho:
            faltando.append(f"{cena.get('id')} (sem arquivo_polido)")
            continue
        clipe = exigir_arquivo_integro(
            resolver(raiz, caminho), 4096, f"clipe polido {cena.get('id')}"
        )
        arquivos.append(clipe)
        metadados.append({"id": str(cena.get("id")), "caminho": str(clipe)})
    if faltando:
        raise RuntimeError(
            "Cenas sem clipe polido: " + ", ".join(faltando)
            + ". Rode a etapa 'polir' (ou 'tudo') antes de montar."
        )
    if not arquivos:
        raise RuntimeError("Nenhum clipe polido disponível para montagem")
    return arquivos, metadados, montagem


def construir_comando_montagem(
    clipes: list[Path],
    destino: Path,
    montagem: dict[str, Any],
    infos: list[dict[str, Any]],
) -> list[str]:
    """Monta o comando FFmpeg de passe único (normaliza escala/fps/SAR)."""
    largura = int(montagem.get("largura", 1024))
    altura = int(montagem.get("altura", 640))
    if largura % 2 or altura % 2:
        raise ValueError(f"Resolução final deve ser par (recebido {largura}x{altura})")
    fps = float(montagem.get("fps", 30))
    modo = str(montagem.get("transicao", "corte"))

    partes: list[str] = []
    for indice, info in enumerate(infos):
        filtros = [
            f"scale={largura}:{altura}:force_original_aspect_ratio=decrease",
            f"pad={largura}:{altura}:(ow-iw)/2:(oh-ih)/2:color=black",
            f"fps={fps}",
            "setsar=1",
            "format=yuv420p",
        ]
        duracao = float(info.get("duracao") or 0.0)
        if modo == "fade" and duracao > 0.5:
            if indice > 0:
                filtros.append("fade=t=in:st=0:d=0.2")
            if indice < len(infos) - 1:
                filtros.append(f"fade=t=out:st={duracao - 0.2:.3f}:d=0.2")
        partes.append(f"[{indice}:v]{','.join(filtros)}[v{indice}]")

    entradas = "".join(f"[v{indice}]" for indice in range(len(infos)))
    partes.append(f"{entradas}concat=n={len(infos)}:v=1:a=0[vfinal]")
    grafo = ";".join(partes)

    comando = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    for clipe in clipes:
        comando.extend(["-i", str(clipe)])
    comando.extend(
        [
            "-filter_complex",
            grafo,
            "-map",
            "[vfinal]",
            "-c:v",
            "libx264",
            "-crf",
            str(int(montagem.get("crf", 18))),
            "-preset",
            str(montagem.get("preset", "medium")),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(destino),
        ]
    )
    return comando


def _nome_arquivo(nome_projeto: str) -> str:
    limpo = re.sub(r"[^A-Za-z0-9._-]+", "_", nome_projeto).strip("_") or "projeto"
    return f"{limpo}_final.mp4"


def montar_video_final(
    config: dict[str, Any],
    raiz: Path,
    projeto_path: Path,
    logger: Any,
    transicao: str | None = None,
) -> Path:
    """Monta o vídeo final consolidado a partir das cenas polidas."""
    clipes, metadados, montagem = planejar_montagem(config, raiz, transicao)
    infos = [ffprobe_info(clipe, logger) for clipe in clipes]
    for meta, info in zip(metadados, infos):
        meta.update(info)

    destino_dir = preparar_diretorio(diretorios(config, raiz)["outputs"])
    destino = destino_dir / _nome_arquivo(str(config.get("nome_projeto", "projeto")))
    comando = construir_comando_montagem(clipes, destino, montagem, infos)

    logger.info("Montando %d clipe(s) -> %s", len(clipes), destino.name)
    executar(comando, descricao="montagem final", logger=logger)
    final = exigir_arquivo_integro(destino, 8192, "vídeo final")
    resultado = ffprobe_info(final, logger)

    esperado = round(sum(info.get("duracao") or 0.0 for info in infos), 2)
    logger.info(
        "Vídeo final: %dx%d @ %.2f fps, %.2fs (esperado ~%.2fs), %.1f MB",
        resultado["largura"],
        resultado["altura"],
        resultado["fps"],
        resultado["duracao"],
        esperado,
        resultado["tamanho"] / (1024 * 1024),
    )
    if resultado["largura"] != int(montagem.get("largura", 1024)):
        logger.warning("Largura final diferente do configurado (%d px)", resultado["largura"])

    config["ultimo_render"] = {
        "arquivo": caminho_relativo(raiz, final),
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "transicao": montagem.get("transicao", "corte"),
        "cenas": [meta["id"] for meta in metadados],
        "duracao_segundos": resultado["duracao"],
        "resolucao": f"{resultado['largura']}x{resultado['altura']}",
        "fps": resultado["fps"],
    }
    salvar_projeto(config, projeto_path)
    return final


def exibir_status(config: dict[str, Any]) -> None:
    """Tabela simples do estado de cada cena."""
    print(f"\nProjeto: {config.get('nome_projeto')}")
    print(f"{'cena':<10} {'status':<10} {'bruto':<26} {'polido':<26}")
    for cena in config.get("cenas", []):
        print(
            f"{str(cena.get('id')):<10} {str(cena.get('status')):<10} "
            f"{str(cena.get('arquivo_raw') or '-'):<26} "
            f"{str(cena.get('arquivo_polido') or '-'):<26}"
        )
    ultimo = config.get("ultimo_render")
    if ultimo:
        print(f"\nÚltimo render: {ultimo.get('arquivo')} ({ultimo.get('duracao_segundos')}s)")
    print()



def preparar_refazer(
    config: dict[str, Any],
    projeto_path: Path,
    raiz: Path,
    cena_ids: list[str],
    nova_seed: int | None,
    logger: Any,
) -> None:
    """Marca cenas como 'pendente', descarta artefatos e troca a seed."""
    descartes = (
        diretorios(config, raiz)["temp"]
        / "_descartados"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    for cena_id in cena_ids:
        cena = obter_cena(config, cena_id)
        for chave in ("arquivo_raw", "arquivo_polido"):
            valor = cena.get(chave)
            if not valor:
                continue
            origem = resolver(raiz, valor)
            if origem.is_file():
                descartes.mkdir(parents=True, exist_ok=True)
                shutil.move(str(origem), str(descartes / f"{cena_id}_{origem.name}"))
        seed = nova_seed if nova_seed is not None else random.randint(1, 2**31 - 1)
        atualizar_cena(
            config,
            cena_id,
            status=STATUS_PENDENTE,
            seed=seed,
            arquivo_raw=None,
            arquivo_polido=None,
            prompt_id=None,
            info_raw=None,
            info_polido=None,
            erro=None,
        )
        logger.info("Cena %s marcada para refazer (seed=%s)", cena_id, seed)
    salvar_projeto(config, projeto_path)


def executar_etapa(etapa: str, args: argparse.Namespace, logger: Any) -> int:
    """Delega a etapa para o módulo responsável, preservando o estado em disco."""
    alvos = list(args.refazer_cena or [])
    if etapa in ("gerar", "escolher", "polir"):
        argv = ["--projeto", args.projeto, "--log-level", args.log_level]
        if not alvos and args.forcar:
            argv.append("--forcar")
        for alvo in alvos:
            argv.extend(["--cena", alvo])
        if etapa == "gerar":
            if args.candidatos is not None:
                argv.extend(["--candidatos", str(int(args.candidatos))])
            return gerador_sequencia.main(argv)
        if etapa == "escolher":
            argv.extend(["--escolher-seed", str(int(args.escolher_seed))])
            return gerador_sequencia.main(argv)
        return polimento.main(argv)

    config, projeto_path, raiz = carregar_projeto(args.projeto)
    montar_video_final(config, raiz, projeto_path, logger, args.transicao)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orquestrador do pipeline (gerar -> polir -> montar)"
    )
    parser.add_argument("--projeto", default="config/projeto.json")
    parser.add_argument("--etapa", choices=ETAPAS, default="tudo")
    parser.add_argument(
        "--refazer-cena",
        action="append",
        default=[],
        help="Id de cena a regenerar (repetível), preservando as demais",
    )
    parser.add_argument("--nova-seed", type=int, default=None)
    parser.add_argument(
        "--candidatos",
        type=int,
        default=None,
        help="Quantos candidatos por cena gerar (sobrescreve candidatos_por_cena)",
    )
    parser.add_argument(
        "--escolher-seed",
        type=int,
        default=None,
        help="Seed já gerada em temp/candidatos que será promovida a '_raw'",
    )
    parser.add_argument("--forcar", action="store_true", help="Refaz cenas já concluídas")
    parser.add_argument("--transicao", choices=("corte", "fade"), default=None)
    parser.add_argument("--continuar-em-erro", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logger = configurar_logging(args.log_level)
    if args.nova_seed is not None and not args.refazer_cena:
        parser.error("--nova-seed exige --refazer-cena")
    if args.escolher_seed is not None and args.etapa != "escolher":
        parser.error("--escolher-seed exige --etapa escolher")
    if args.etapa == "escolher" and args.escolher_seed is None:
        parser.error("--etapa escolher exige --escolher-seed N")

    config, projeto_path, raiz = carregar_projeto(args.projeto)

    if args.refazer_cena and args.etapa != "escolher":
        try:
            preparar_refazer(
                config, projeto_path, raiz, args.refazer_cena, args.nova_seed, logger
            )
        except KeyError as erro:
            logger.error("%s", erro.args[0] if erro.args else erro)
            return 1
        if args.etapa in ("montar", "status"):
            logger.warning(
                "Cenas marcadas como pendentes; rode --etapa gerar (ou tudo) para refazê-las."
            )

    if args.etapa == "status":
        exibir_status(config)
        return 0

    etapas = ["gerar", "polir", "montar"] if args.etapa == "tudo" else [args.etapa]
    for etapa in etapas:
        logger.info("== Etapa: %s ==", etapa)
        try:
            codigo = executar_etapa(etapa, args, logger)
        except (RuntimeError, ValueError, FileNotFoundError) as erro:
            logger.error("Etapa %s falhou: %s", etapa, erro)
            codigo = 1
        if codigo != 0:
            if not args.continuar_em_erro:
                logger.error("Pipeline interrompido na etapa '%s'.", etapa)
                return codigo
            logger.warning("Etapa %s retornou código %d; seguindo.", etapa, codigo)

    if "montar" in etapas:
        estado, _, _ = carregar_projeto(args.projeto)
        ultimo = estado.get("ultimo_render") or {}
        if ultimo.get("arquivo"):
            logger.info("Vídeo final: %s", ultimo["arquivo"])
    logger.info("Pipeline concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

