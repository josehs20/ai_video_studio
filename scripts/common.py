"""Utilitários compartilhados do ai_video_studio.

Concentra logging, persistência atômica do JSON de projeto, resolução de
caminhos, validação de integridade de arquivos e wrappers de subprocesso
(FFmpeg/ffprobe) usados por todos os scripts do pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

STATUS_PENDENTE = "pendente"
STATUS_GERADO = "gerado"
STATUS_POLIDO = "polido"
STATUS_APROVADO = "aprovado"
STATUS_ERRO = "erro"

STATUS_VALIDOS = (
    STATUS_PENDENTE,
    STATUS_GERADO,
    STATUS_POLIDO,
    STATUS_APROVADO,
    STATUS_ERRO,
)

# Extensões de vídeo/imagem animada aceitas como clipe bruto do ComfyUI.
EXTENSOES_RAW = (".webp", ".mp4", ".webm", ".mov", ".mkv", ".gif")

# Extensões de imagem estática aceitas como referência visual de cena.
EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Orçamento conservador para LTX-Video 2B em 6 GB de VRAM com `--lowvram`.
# Subir resolução/frames além disso tende a estourar a VRAM antes de os steps
# extras compensarem a perda de qualidade.
LIMITE_VRAM_LARGURA = 512
LIMITE_VRAM_ALTURA = 320
LIMITE_VRAM_FRAMES = 49
LIMITE_VRAM_PIXELS_FRAMES = LIMITE_VRAM_LARGURA * LIMITE_VRAM_ALTURA * LIMITE_VRAM_FRAMES

# Faixa de luma (8 bits) considerada "tela preta" no signalstats.
LUMA_QUASE_PRETO = 24.0
# Diferença média mínima entre frames para considerar o clipe "vivo".
YDIF_MINIMO = 0.05


def configurar_logging(nivel: str = "INFO", nome: str = "ai_video_studio") -> logging.Logger:
    """Configura e devolve um logger estruturado (sem duplicar handlers)."""
    logger = logging.getLogger(nome)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, str(nivel).upper(), logging.INFO))
    return logger


def agora_iso() -> str:
    """Timestamp UTC em ISO-8601 (segundos), usado no estado do projeto."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def carregar_json(caminho: Path) -> dict[str, Any]:
    """Lê um JSON UTF-8 e garante que o conteúdo seja um objeto."""
    with caminho.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    if not isinstance(dados, dict):
        raise ValueError(f"JSON inválido em {caminho}: esperado um objeto no topo")
    return dados


def salvar_json_atomico(caminho: Path, dados: dict[str, Any]) -> None:
    """Grava JSON de forma atômica (arquivo temporário + os.replace)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f".{caminho.name}.tmp")
    with temporario.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
        arquivo.flush()
        os.fsync(arquivo.fileno())
    os.replace(temporario, caminho)


def raiz_projeto(config_path: Path) -> Path:
    """Raiz do projeto: o diretório acima de `config/` (ou o próprio diretório)."""
    caminho = config_path.expanduser().resolve()
    if caminho.parent.name == "config":
        return caminho.parent.parent
    return caminho.parent


def resolver(raiz: Path, valor: str | os.PathLike[str]) -> Path:
    """Resolve um caminho do JSON (absoluto, `~` ou relativo à raiz do projeto)."""
    caminho = Path(str(valor)).expanduser()
    if caminho.is_absolute():
        return caminho
    return (raiz / caminho).resolve()


def caminho_relativo(raiz: Path, caminho: Path) -> str:
    """Serializa um caminho no JSON: relativo à raiz quando possível."""
    try:
        return caminho.resolve().relative_to(raiz.resolve()).as_posix()
    except ValueError:
        return caminho.resolve().as_posix()


def exigir_arquivo_integro(
    caminho: Path,
    minimo_bytes: int = 1024,
    descricao: str | None = None,
) -> Path:
    """Valida existência, tipo e tamanho mínimo (rejeita arquivos de 0 bytes)."""
    rotulo = descricao or caminho.name
    if not caminho.exists():
        raise FileNotFoundError(f"{rotulo} não encontrado: {caminho}")
    if not caminho.is_file():
        raise ValueError(f"{rotulo} não é um arquivo: {caminho}")
    tamanho = caminho.stat().st_size
    if tamanho < minimo_bytes:
        raise ValueError(
            f"{rotulo} parece corrompido/ponteiro LFS vazio "
            f"({tamanho} bytes < {minimo_bytes}): {caminho}"
        )
    return caminho


def preparar_diretorio(caminho: Path, limpar: bool = False) -> Path:
    """Cria o diretório; opcionalmente remove conteúdo anterior."""
    if limpar and caminho.exists():
        shutil.rmtree(caminho)
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def executar(
    comando: Sequence[str],
    descricao: str,
    logger: logging.Logger | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Executa um processo externo sem shell, com log e erro explícito."""
    log = logger or logging.getLogger("ai_video_studio")
    log.debug("Comando (%s): %s", descricao, " ".join(str(parte) for parte in comando))
    try:
        return subprocess.run(
            [str(parte) for parte in comando],
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as erro:
        raise RuntimeError(
            f"{descricao}: executável não encontrado ({comando[0]}). "
            "Instale a dependência ou ajuste o caminho em config/projeto.json."
        ) from erro
    except subprocess.CalledProcessError as erro:
        detalhe = (erro.stderr or erro.stdout or "").strip()
        ultimas = "\n".join(detalhe.splitlines()[-12:])
        raise RuntimeError(f"{descricao} falhou (código {erro.returncode}):\n{ultimas}") from erro
    except subprocess.TimeoutExpired as erro:
        raise RuntimeError(f"{descricao} excedeu o timeout de {timeout}s") from erro
def binario_disponivel(nome: str, extras: Iterable[Path] = ()) -> Path | None:
    """Localiza um binário no PATH ou em diretórios extras (ex: ./bin)."""
    candidatos: list[Path] = [Path(extra) / nome for extra in extras]
    candidatos.append(Path(nome))
    for candidato in candidatos:
        achado = shutil.which(str(candidato))
        if achado:
            return Path(achado).resolve()
    return None


def _fps_de_fracao(valor: str | None) -> float:
    """Converte '30000/1001' em float; devolve 0.0 quando indisponível."""
    if not valor:
        return 0.0
    if "/" not in valor:
        try:
            return float(valor)
        except ValueError:
            return 0.0
    numerador, denominador = valor.split("/", 1)
    try:
        return float(numerador) / float(denominador)
    except (ValueError, ZeroDivisionError):
        return 0.0


def ffprobe_info(caminho: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    """Lê metadados essenciais do vídeo via ffprobe."""
    log = logger or logging.getLogger("ai_video_studio")
    resultado = executar(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(caminho),
        ],
        descricao=f"ffprobe {caminho.name}",
        logger=log,
    )
    dados = json.loads(resultado.stdout or "{}")
    streams = dados.get("streams") or [{}]
    stream = streams[0]
    formato = dados.get("format") or {}
    fps = _fps_de_fracao(stream.get("avg_frame_rate")) or _fps_de_fracao(
        stream.get("r_frame_rate")
    )
    return {
        "largura": int(stream.get("width") or 0),
        "altura": int(stream.get("height") or 0),
        "fps": round(fps, 4),
        "frames": int(stream.get("nb_frames") or 0),
        "codec": stream.get("codec_name"),
        "duracao": round(float(formato.get("duration") or 0.0), 3),
        "tamanho": int(formato.get("size") or caminho.stat().st_size),
    }


def ajustar_frames_ltx(frames: int, minimo: int = 9, maximo: int = 97) -> int:
    """Arredonda `frames` para o múltiplo de 8 + 1 mais próximo (exigência do LTX).

    O VAE do LTX-Video comprime o tempo em blocos de 8, então a contagem de
    frames precisa obedecer `frames = 8 * k + 1` (25, 33, 41, 49, ...). Pedir
    `31` no grafo, por exemplo, faz o ComfyUI entregar 25 frames — o ajuste
    explícito evita essa diferença silenciosa entre config e saída.
    """
    desejado = min(max(int(frames), int(minimo)), int(maximo))
    return int(round((desejado - 1) / 8.0)) * 8 + 1


def decompor_orcamento() -> str:
    """Texto curto do orçamento de VRAM usado nas mensagens de diagnóstico."""
    return (
        f"{LIMITE_VRAM_LARGURA}x{LIMITE_VRAM_ALTURA}, "
        f"ate {LIMITE_VRAM_FRAMES} frames"
    )


def avaliar_orcamento_vram(
    largura: int,
    altura: int,
    frames: int,
    steps: int | None = None,
) -> list[str]:
    """Aponta configurações que tendem a estourar 6 GB de VRAM no `--lowvram`.

    Devolve uma lista de avisos legíveis (vazia quando o orçamento é seguro).
    """
    avisos: list[str] = []
    largura, altura, frames = int(largura), int(altura), int(frames)
    if largura > LIMITE_VRAM_LARGURA:
        avisos.append(
            f"largura {largura}px acima do limite seguro para 6 GB "
            f"(<= {LIMITE_VRAM_LARGURA}px)"
        )
    if altura > LIMITE_VRAM_ALTURA:
        avisos.append(
            f"altura {altura}px acima do limite seguro para 6 GB "
            f"(<= {LIMITE_VRAM_ALTURA}px)"
        )
    if frames > LIMITE_VRAM_FRAMES:
        avisos.append(
            f"frames {frames} acima do limite seguro para 6 GB "
            f"(<= {LIMITE_VRAM_FRAMES} frames)"
        )
    custo = largura * altura * max(frames, 1)
    if custo > LIMITE_VRAM_PIXELS_FRAMES:
        fator = custo / float(LIMITE_VRAM_PIXELS_FRAMES)
        avisos.append(
            f"custo pixels×frames em {fator:.2f}x do orçamento seguro "
            f"({decompor_orcamento()}); reduza resolução/frames antes de subir steps"
        )
    if steps is not None and int(steps) > 40:
        avisos.append(
            f"steps {int(steps)} > 40: em 6 GB cada step extra soma risco de OOM; "
            "mantenha 30–40 com resolução <= 512x320"
        )
    return avisos


def nome_referencia_no_input(cena_id: str, origem: Path) -> str:
    """Nome estável (e sem colisão entre cenas) da referência em `ComfyUI/input`."""
    sufixo = origem.suffix.lower() if origem.suffix else ".png"
    return f"avs_ref_{cena_id}{sufixo}"


def sincronizar_para_input(origem: Path, destino_dir: Path, nome: str) -> Path:
    """Copia `origem` para `destino_dir/nome` quando o conteúdo ainda não está lá."""
    try:
        destino_dir.mkdir(parents=True, exist_ok=True)
    except OSError as erro:
        raise RuntimeError(f"Não foi possível preparar {destino_dir}: {erro}") from erro
    destino = destino_dir / nome
    if not destino.exists() or destino.stat().st_size != origem.stat().st_size:
        shutil.copy2(origem, destino)
    return destino


def inspecionar_imagem(
    caminho: Path,
    logger: logging.Logger | None = None,
    minimo_bytes: int = 512,
) -> dict[str, Any]:
    """Valida uma imagem de referência e devolve dimensões/codec via ffprobe."""
    log = logger or logging.getLogger("ai_video_studio")
    if caminho.suffix.lower() not in EXTENSOES_IMAGEM:
        raise ValueError(
            f"imagem de referência com extensão não suportada ({caminho.suffix}): {caminho}"
        )
    exigir_arquivo_integro(caminho, minimo_bytes=minimo_bytes, descricao="imagem de referência")
    resultado = executar(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(caminho),
        ],
        descricao=f"ffprobe imagem {caminho.name}",
        logger=log,
    )
    dados = json.loads(resultado.stdout or "{}")
    stream = (dados.get("streams") or [{}])[0]
    largura = int(stream.get("width") or 0)
    altura = int(stream.get("height") or 0)
    if largura <= 0 or altura <= 0:
        raise ValueError(f"imagem de referência inválida (sem dimensões): {caminho}")
    return {
        "largura": largura,
        "altura": altura,
        "codec": stream.get("codec_name"),
        "tamanho": caminho.stat().st_size,
    }


def _escapar_filtro_movie(caminho: Path) -> str:
    """Escapa o caminho para uso em `movie=<path>` (sintaxe de filtros do FFmpeg)."""
    texto = str(caminho)
    for caractere in ("\\", "'", ":", ","):
        texto = texto.replace(caractere, f"\\{caractere}")
    return texto


def metricas_video(
    caminho: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Mede luma média e diferença entre frames (YAVG/YDIF) via ffprobe+signalstats.

    Usado para descartar clipes pretos/corrompidos/congelados e para pontuar
    candidatos multi-seed sem precisar de OpenCV.
    """
    log = logger or logging.getLogger("ai_video_studio")
    resultado = executar(
        [
            "ffprobe",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"movie={_escapar_filtro_movie(caminho)},signalstats",
            "-show_entries",
            "frame_tags=lavfi.signalstats.YAVG,lavfi.signalstats.YDIF",
            "-of",
            "csv=p=0",
        ],
        descricao=f"métricas de frames de {caminho.name}",
        logger=log,
    )
    luma: list[float] = []
    diferencas: list[float] = []
    for linha in (resultado.stdout or "").splitlines():
        partes = [parte for parte in linha.strip().split(",") if parte]
        if len(partes) < 2:
            continue
        try:
            luma.append(float(partes[0]))
            diferencas.append(float(partes[1]))
        except ValueError:
            continue
    if len(luma) < 2:
        raise RuntimeError(f"{caminho.name}: signalstats devolveu {len(luma)} frame(s)")

    movimento = diferencas[1:] or [0.0]
    fracao_preta = sum(1 for valor in luma if valor < LUMA_QUASE_PRETO) / len(luma)
    fracao_congelada = sum(1 for valor in movimento if valor < YDIF_MINIMO) / len(movimento)
    return {
        "frames_analisados": len(luma),
        "luma_media": round(statistics.fmean(luma), 3),
        "luma_desvio": round(statistics.pstdev(luma), 3) if len(luma) > 1 else 0.0,
        # O YDIF do primeiro frame é sempre 0; ignora para não mascarar a média.
        "movimento_medio": round(statistics.fmean(movimento), 4),
        "movimento_maximo": round(max(movimento), 4),
        "fracao_quase_preta": round(fracao_preta, 4),
        "fracao_congelada": round(fracao_congelada, 4),
    }


def avaliar_candidato(
    caminho: Path,
    logger: logging.Logger | None = None,
    duracao_esperada: float | None = None,
    tolerancia: float = 0.35,
) -> dict[str, Any]:
    """Valida integridade/duração/movimento de um clipe e devolve uma pontuação.

    Critérios de reprovação: ffprobe falha, codec ausente, vídeo praticamente
    todo preto, frames congelados (YDIF ~ 0) ou duração muito fora do esperado.
    """
    log = logger or logging.getLogger("ai_video_studio")
    avaliacao: dict[str, Any] = {
        "arquivo": str(caminho),
        "valido": False,
        "motivos": [],
        "pontuacao": 0.0,
    }
    try:
        info = ffprobe_info(caminho, log)
    except (RuntimeError, ValueError) as erro:
        avaliacao["motivos"].append(f"ffprobe falhou: {erro}")
        return avaliacao
    avaliacao["info"] = info
    if not info.get("codec") or info.get("duracao", 0) <= 0:
        avaliacao["motivos"].append("clipe sem codec/duração válidos (possível corrupção)")
    if info.get("frames", 0) < 2:
        avaliacao["motivos"].append(f"apenas {info.get('frames', 0)} frame(s)")

    if duracao_esperada and duracao_esperada > 0:
        razao = float(info.get("duracao") or 0.0) / float(duracao_esperada)
        avaliacao["duracao_esperada"] = round(float(duracao_esperada), 3)
        avaliacao["razao_duracao"] = round(razao, 3)
        if razao < (1.0 - tolerancia):
            avaliacao["motivos"].append(
                f"duração {info.get('duracao')}s muito curta "
                f"(esperado ~{duracao_esperada:.2f}s)"
            )
        elif razao > (1.0 + tolerancia) + (1.0 / max(duracao_esperada, 1.0)):
            avaliacao["motivos"].append(
                f"duração {info.get('duracao')}s muito longa "
                f"(esperado ~{duracao_esperada:.2f}s)"
            )

    try:
        metricas = metricas_video(caminho, log)
    except (RuntimeError, ValueError) as erro:
        avaliacao["motivos"].append(f"métricas de frames falharam: {erro}")
        metricas = {}
    if metricas:
        avaliacao["metricas"] = metricas
        if metricas["fracao_quase_preta"] >= 0.9:
            avaliacao["motivos"].append(
                f"{metricas['fracao_quase_preta'] * 100:.0f}% dos frames quase pretos"
            )
        if (
            metricas["movimento_medio"] < YDIF_MINIMO
            and metricas["luma_desvio"] < YDIF_MINIMO
        ):
            avaliacao["motivos"].append(
                "frames congelados (variação temporal ~0 entre frames)"
            )

        movimento = min(metricas["movimento_medio"], 3.0) / 3.0
        detalhe = min(metricas["luma_desvio"], 15.0) / 15.0
        estabilidade = 1.0 - min(metricas["fracao_quase_preta"], 1.0)
        avaliacao["pontuacao"] = round(
            0.6 * movimento + 0.3 * detalhe + 0.1 * estabilidade, 4
        )

    avaliacao["valido"] = not avaliacao["motivos"]
    if not avaliacao["valido"]:
        avaliacao["pontuacao"] = 0.0
    return avaliacao


def obter_cena(config: dict[str, Any], cena_id: str) -> dict[str, Any]:
    """Devolve a cena pelo id, com erro claro quando não existir."""
    for cena in config.get("cenas", []):
        if cena.get("id") == cena_id:
            return cena
    disponiveis = ", ".join(str(c.get("id")) for c in config.get("cenas", [])) or "nenhuma"
    raise KeyError(f"Cena '{cena_id}' não existe no projeto (disponíveis: {disponiveis})")


def atualizar_cena(config: dict[str, Any], cena_id: str, **campos: Any) -> dict[str, Any]:
    """Atualiza campos de uma cena (mutação controlada) e carimba a data."""
    cena = obter_cena(config, cena_id)
    cena.update(campos)
    cena["atualizado_em"] = agora_iso()
    return cena


def diretorios(config: dict[str, Any], raiz: Path) -> dict[str, Path]:
    """Resolve os diretórios de trabalho declarados no JSON do projeto."""
    mapa = config.get("diretorios") or {}
    return {
        "raw": resolver(raiz, mapa.get("raw", "inputs/projeto_atual")),
        "temp": resolver(raiz, mapa.get("temp", "temp")),
        "outputs": resolver(raiz, mapa.get("outputs", "outputs")),
    }


def carregar_projeto(
    caminho: str | os.PathLike[str],
) -> tuple[dict[str, Any], Path, Path]:
    """Carrega o projeto e devolve (config, caminho_resolvido, raiz)."""
    projeto_path = Path(caminho).expanduser().resolve()
    config = carregar_json(projeto_path)
    if not config.get("cenas"):
        raise ValueError(f"Projeto sem cenas: {projeto_path}")
    return config, projeto_path, raiz_projeto(projeto_path)


def salvar_projeto(config: dict[str, Any], projeto_path: Path) -> None:
    """Persiste o estado do projeto de forma atômica."""
    salvar_json_atomico(projeto_path, config)

