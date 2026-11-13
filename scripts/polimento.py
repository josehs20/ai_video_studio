"""Polimento dos clipes brutos: interpolação temporal + upscale + encode final.

Estratégia (ordem do spec 03):
1. Extrai frames do clipe bruto na taxa nativa (`geracao.fps`).
2. Interpola: `rife-ncnn-vulkan` quando instalado; senão FFmpeg `minterpolate`.
3. Escala: `realesrgan-ncnn-vulkan` quando instalado; senão `scale` lanczos.
4. Encoda H.264 (`yuv420p`, `fps_alvo`) em `temp/{cena}_polido.mp4`.

Uso:
    python3 scripts/polimento.py
    python3 scripts/polimento.py --cena cena_02 --forcar --interpolador minterpolate
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    STATUS_ERRO,
    STATUS_GERADO,
    STATUS_POLIDO,
    atualizar_cena,
    binario_disponivel,
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

BINARIOS_EXTRA = (Path(__file__).resolve().parent.parent / "bin",)
FILTRO_MINTERPOLATE = (
    "minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
)


def _frames(pasta: Path) -> list[Path]:
    return sorted(pasta.glob("*.png"))


def _numero_inicial(pasta: Path) -> int:
    """Primeiro número da sequência `%05d.png` (RIFE pode começar em 1)."""
    arquivos = _frames(pasta)
    if not arquivos:
        return 1
    try:
        return int(arquivos[0].stem)
    except ValueError:
        return 1


def extrair_frames(video: Path, destino: Path, fps: float, logger: Any) -> int:
    """Extrai frames PNG em taxa constante e devolve a quantidade."""
    preparar_diretorio(destino, limpar=True)
    executar(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={fps}",
            str(destino / "%05d.png"),
        ],
        descricao=f"extração de frames de {video.name}",
        logger=logger,
    )
    quantidade = len(_frames(destino))
    if quantidade < 2:
        raise RuntimeError(f"{video.name}: só {quantidade} frame(s) extraído(s)")
    return quantidade


def montar_video(frames_dir: Path, fps: float, saida: Path, logger: Any) -> Path:
    """Encoda uma sequência de PNGs em H.264 intermediário."""
    executar(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-framerate",
            f"{fps}",
            "-start_number",
            str(_numero_inicial(frames_dir)),
            "-i",
            str(frames_dir / "%05d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(saida),
        ],
        descricao=f"montagem de {saida.name}",
        logger=logger,
    )
    return exigir_arquivo_integro(saida, 4096, f"intermediário {saida.name}")


def interpolar_com_minterpolate(
    video: Path, saida: Path, fps_alvo: float, logger: Any
) -> Path:
    """Fallback FFmpeg de interpolação por movimento (sem binário externo)."""
    executar(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            FILTRO_MINTERPOLATE.format(fps=fps_alvo),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(saida),
        ],
        descricao="minterpolate",
        logger=logger,
    )
    return exigir_arquivo_integro(saida, 4096, "vídeo interpolado")


def interpolar_com_rife(
    frames_dir: Path,
    workdir: Path,
    fps_origem: float,
    fps_alvo: float,
    binario: Path,
    logger: Any,
) -> tuple[Path, float] | None:
    """Aplica RIFE (2x por passada) até atingir/exceder o fps alvo."""
    passadas = max(1, math.ceil(math.log2(max(fps_alvo / fps_origem, 2.0))))
    passadas = min(passadas, 2)
    atual = frames_dir
    for indice in range(passadas):
        destino = workdir / f"frames_rife_{indice + 1}"
        preparar_diretorio(destino, limpar=True)
        try:
            executar(
                [str(binario), "-i", str(atual), "-o", str(destino)],
                descricao=f"RIFE passada {indice + 1}",
                logger=logger,
            )
        except RuntimeError as erro:
            logger.warning("RIFE falhou (%s); usando minterpolate.", erro)
            return None
        if len(_frames(destino)) <= len(_frames(atual)):
            logger.warning("RIFE não gerou frames extras; usando minterpolate.")
            return None
        atual = destino
    return atual, fps_origem * (2**passadas)


def escalar_com_realesrgan(
    frames_dir: Path,
    destino: Path,
    escala: int,
    modelo: str,
    binario: Path,
    logger: Any,
) -> Path | None:
    """Upscale quadro a quadro com Real-ESRGAN (Vulkan)."""
    preparar_diretorio(destino, limpar=True)
    try:
        executar(
            [
                str(binario),
                "-i",
                str(frames_dir),
                "-o",
                str(destino),
                "-n",
                modelo,
                "-s",
                str(escala),
                "-f",
                "png",
            ],
            descricao="Real-ESRGAN",
            logger=logger,
        )
    except RuntimeError as erro:
        logger.warning("Real-ESRGAN falhou (%s); usando scale lanczos.", erro)
        return None
    if len(_frames(destino)) < len(_frames(frames_dir)):
        logger.warning("Real-ESRGAN devolveu menos frames; usando scale lanczos.")
        return None
    return destino


def _encodar_final(
    entrada: Path,
    saida: Path,
    fps_alvo: float,
    escala_pendente: int,
    crf: int,
    logger: Any,
) -> Path:
    """Encode final H.264 normalizando fps, escala e pix_fmt."""
    filtros: list[str] = []
    if escala_pendente > 1:
        filtros.append(f"scale=iw*{escala_pendente}:ih*{escala_pendente}:flags=lanczos")
    filtros.append(f"fps={fps_alvo}")
    filtros.append("format=yuv420p")
    executar(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(entrada),
            "-vf",
            ",".join(filtros),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(saida),
        ],
        descricao=f"encode final de {saida.name}",
        logger=logger,
    )
    return exigir_arquivo_integro(saida, 4096, f"clipe polido {saida.name}")


def processar_cena(
    cena: dict[str, Any],
    config: dict[str, Any],
    raiz: Path,
    projeto_path: Path,
    opcoes: argparse.Namespace,
    logger: Any,
) -> Path:
    """Interpola, escala e encoda uma cena; atualiza o status para 'polido'."""
    cena_id = str(cena["id"])
    bruto = cena.get("arquivo_raw")
    if not bruto:
        raise ValueError(f"{cena_id}: sem 'arquivo_raw' (rode o gerador primeiro)")
    origem = exigir_arquivo_integro(
        resolver(raiz, bruto), minimo_bytes=1024, descricao=f"clipe bruto {cena_id}"
    )

    geracao = config.get("geracao") or {}
    polimento = config.get("polimento") or {}
    fps_origem = float(geracao.get("fps", 12))
    fps_alvo = float(opcoes.fps or polimento.get("fps_alvo", 30))
    escala = int(opcoes.escala or polimento.get("escala_upscale", 2))
    modelo_escala = str(polimento.get("modelo_escala", "realesrgan-x4plus"))
    interpolador = str(opcoes.interpolador or polimento.get("interpolador", "auto"))

    diretorios_projeto = diretorios(config, raiz)
    bruto_dir = diretorios_projeto["temp"]
    workdir = preparar_diretorio(bruto_dir / f"_work_{cena_id}", limpar=True)

    bin_rife = binario_disponivel("rife-ncnn-vulkan", BINARIOS_EXTRA)
    bin_esrgan = binario_disponivel("realesrgan-ncnn-vulkan", BINARIOS_EXTRA)

    frames_origem = workdir / "frames_in"
    try:
        total = extrair_frames(origem, frames_origem, fps_origem, logger)
    except RuntimeError as erro:
        if origem.suffix.lower() == ".webp":
            raise RuntimeError(
                f"{cena_id}: o FFmpeg deste ambiente não decodifica WebP animado "
                f"({erro}). Regenere a cena com o workflow que salva MP4 "
                "(nós CreateVideo + SaveVideo)."
            ) from erro
        raise
    logger.info("%s: %d frames a %.0f fps extraídos", cena_id, total, fps_origem)

    fonte_frames = frames_origem
    fps_base = fps_origem
    resultado_rife = None
    if interpolador in ("auto", "rife") and bin_rife:
        resultado_rife = interpolar_com_rife(
            frames_origem, workdir, fps_origem, fps_alvo, bin_rife, logger
        )
    if resultado_rife:
        fonte_frames, fps_base = resultado_rife
        logger.info("%s: RIFE -> %.0f fps (%d frames)", cena_id, fps_base, len(_frames(fonte_frames)))
    else:
        if interpolador == "rife" and not bin_rife:
            logger.warning("rife-ncnn-vulkan ausente; usando minterpolate.")
        interp = interpolar_com_minterpolate(origem, workdir / "interp.mp4", fps_alvo, logger)
        frames_interp = workdir / "frames_interp"
        extrair_frames(interp, frames_interp, fps_alvo, logger)
        fonte_frames, fps_base = frames_interp, fps_alvo
        logger.info(
            "%s: minterpolate -> %.0f fps (%d frames)", cena_id, fps_base, len(_frames(fonte_frames))
        )

    escala_pendente = escala
    if escala > 1 and bin_esrgan:
        destino_up = workdir / "frames_up"
        if escalar_com_realesrgan(
            fonte_frames, destino_up, escala, modelo_escala, bin_esrgan, logger
        ):
            fonte_frames = destino_up
            escala_pendente = 1
            logger.info("%s: Real-ESRGAN x%d aplicado", cena_id, escala)
    if escala_pendente > 1:
        logger.info("%s: upscale por scale lanczos x%d (sem modelo dedicado)", cena_id, escala)

    intermediario = montar_video(fonte_frames, fps_base, workdir / "pre_final.mp4", logger)
    destino = bruto_dir / f"{cena_id}_polido.mp4"
    _encodar_final(intermediario, destino, fps_alvo, escala_pendente, int(opcoes.crf), logger)

    info = ffprobe_info(destino, logger)
    if info["duracao"] <= 0:
        raise RuntimeError(f"{cena_id}: vídeo polido sem duração válida")

    atualizar_cena(
        config,
        cena_id,
        status=STATUS_POLIDO,
        arquivo_polido=caminho_relativo(raiz, destino),
        info_polido=info,
        erro=None,
    )
    salvar_projeto(config, projeto_path)

    if not opcoes.manter_temp:
        shutil.rmtree(workdir, ignore_errors=True)

    logger.info(
        "%s polida: %s (%dx%d @ %.0f fps, %.2fs)",
        cena_id,
        destino.name,
        info["largura"],
        info["altura"],
        info["fps"],
        info["duracao"],
    )
    return destino



def selecionar_cenas(
    config: dict[str, Any],
    alvos: list[str],
    forcar: bool,
) -> list[dict[str, Any]]:
    """Cenas a polir: por id, todas as geradas, ou todas com clipe bruto."""
    if alvos:
        return [obter_cena(config, alvo) for alvo in alvos]
    if forcar:
        return [cena for cena in config.get("cenas", []) if cena.get("arquivo_raw")]
    return [
        cena for cena in config.get("cenas", []) if cena.get("status") == STATUS_GERADO
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polimento (RIFE/upscale) dos clipes")
    parser.add_argument("--projeto", default="config/projeto.json")
    parser.add_argument("--cena", action="append", default=[], help="Id da cena (repetível)")
    parser.add_argument("--forcar", action="store_true", help="Repolir cenas já polidas")
    parser.add_argument(
        "--interpolador",
        choices=("auto", "rife", "minterpolate"),
        default=None,
        help="auto = RIFE se instalado, senão minterpolate",
    )
    parser.add_argument("--escala", type=int, default=None, help="Fator de upscale")
    parser.add_argument("--fps", type=float, default=None, help="fps final")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--manter-temp", action="store_true", help="Não apaga frames temporários")
    parser.add_argument("--parar-no-erro", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logger = configurar_logging(args.log_level)
    if not binario_disponivel("ffmpeg"):
        logger.error("ffmpeg não encontrado no PATH.")
        return 1

    config, projeto_path, raiz = carregar_projeto(args.projeto)
    cenas = selecionar_cenas(config, args.cena, args.forcar)
    if not cenas:
        logger.info("Nenhuma cena com status 'gerado' para polir.")
        return 0

    logger.info("Polindo %d cena(s): %s", len(cenas), ", ".join(c["id"] for c in cenas))
    falhas = 0
    for cena in cenas:
        try:
            processar_cena(cena, config, raiz, projeto_path, args, logger)
        except (RuntimeError, ValueError, FileNotFoundError, KeyError) as erro:
            falhas += 1
            logger.error("Falha ao polir %s: %s", cena.get("id"), erro)
            atualizar_cena(
                config, str(cena["id"]), status=STATUS_ERRO, erro=str(erro)[:1000]
            )
            salvar_projeto(config, projeto_path)
            if args.parar_no_erro:
                break

    if falhas:
        logger.warning("Polimento finalizado com %d falha(s).", falhas)
        return 1
    logger.info("Polimento concluído com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

