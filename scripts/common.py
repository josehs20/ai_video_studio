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

