"""Diagnóstico do ambiente ComfyUI e geração/validação do `workflow_api.json`.

O grafo gerado segue o template oficial do ComfyUI para LTX-Video 0.9.x
(`ltxv_text_to_video`): checkpoint apenas para MODEL/VAE, CLIPLoader `ltxv`
para o T5-XXL, `EmptyLTXVLatentVideo` + `LTXVConditioning` + `LTXVScheduler`
+ `SamplerCustom` para a amostragem.

Uso:
    python3 scripts/validar_ambiente.py
    python3 scripts/validar_ambiente.py --escrever-workflow
    python3 scripts/validar_ambiente.py --url http://127.0.0.1:8188 --log-level DEBUG
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    carregar_json,
    carregar_projeto,
    configurar_logging,
    exigir_arquivo_integro,
    resolver,
    salvar_json_atomico,
)

EXTENSOES_MODELO = (".safetensors", ".onnx", ".pth", ".ckpt", ".pt", ".bin")

NODE_CLASSES_OBRIGATORIOS = (
    "CLIPLoader",
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLTXVLatentVideo",
    "LTXVConditioning",
    "LTXVScheduler",
    "KSamplerSelect",
    "SamplerCustom",
    "VAEDecode",
    "CreateVideo",
    "SaveVideo",
    "LoadImage",
    "ReActorFaceSwap",
)


def consultar_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """GET simples com timeout, usado nas rotas de diagnóstico do ComfyUI."""
    with request.urlopen(url, timeout=timeout) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def auditar_modelos(models_dir: Path) -> tuple[list[Path], list[Path]]:
    """Varre o diretório de modelos e separa arquivos íntegros de 0 bytes."""
    integros: list[Path] = []
    vazios: list[Path] = []
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Diretório de modelos não encontrado: {models_dir}")
    for arquivo in sorted(models_dir.rglob("*")):
        if not arquivo.is_file() or arquivo.suffix.lower() not in EXTENSOES_MODELO:
            continue
        if arquivo.stat().st_size == 0:
            vazios.append(arquivo)
        else:
            integros.append(arquivo)
    return integros, vazios


def localizar_modelo(models_dir: Path, nome_arquivo: str) -> Path | None:
    """Encontra um modelo pelo nome em qualquer subpasta de `models/`."""
    for candidato in models_dir.rglob(nome_arquivo):
        if candidato.is_file():
            return candidato
    return None



def construir_workflow_template(config: dict[str, Any]) -> dict[str, Any]:
    """Monta o grafo LTX-Video em formato API a partir do JSON do projeto."""
    geracao = config.get("geracao") or {}
    modelos = config.get("modelos") or {}
    primeira = config["cenas"][0]

    largura = int(geracao.get("largura", 512))
    altura = int(geracao.get("altura", 320))
    frames = int(primeira.get("frames") or geracao.get("frames", 33))
    fps = float(geracao.get("fps", 12))
    steps = int(geracao.get("steps", 25))
    cfg = float(geracao.get("cfg", 3.0))
    sampler = str(geracao.get("sampler", "euler"))
    negativo = (
        primeira.get("negative_prompt")
        or geracao.get("prompt_negativo_padrao")
        or "low quality, worst quality, deformed, distorted, blurry"
    )
    referencia = Path(str(config.get("referencia_facial", "personagem_base.png"))).name

    return {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": modelos.get("text_encoder", "t5xxl_fp8_e4m3fn.safetensors"),
                "type": "ltxv",
                "device": "default",
            },
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": modelos.get("checkpoint", "ltx-video-2b-v0.9.1.safetensors")
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": str(primeira.get("prompt", "")), "clip": ["1", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": str(negativo), "clip": ["1", 0]},
        },
        "5": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": largura,
                "height": altura,
                "length": frames,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": fps},
        },
        "7": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["5", 0],
            },
        },
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "9": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["2", 0],
                "add_noise": True,
                "noise_seed": int(primeira.get("seed", 42)),
                "cfg": cfg,
                "positive": ["6", 0],
                "negative": ["6", 1],
                "sampler": ["8", 0],
                "sigmas": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "10": {"class_type": "LoadImage", "inputs": {"image": referencia}},
        "11": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": bool(primeira.get("aplicar_reactor", False)),
                "input_image": ["12", 0],
                "swap_model": modelos.get("reactor_swap", "inswapper_128.onnx"),
                "facedetection": "retinaface_resnet50",
                "face_restore_model": modelos.get("reactor_restore", "GFPGANv1.4.pth"),
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
                "source_image": ["10", 0],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["2", 2]},
        },
        "13": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["11", 0], "fps": fps},
        },
        "14": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["13", 0],
                "filename_prefix": str(primeira.get("id", "cena_01")),
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def _opcoes_combo(opcoes: Any) -> list[str] | None:
    """Extrai a lista de opções de um input combo do `/object_info`."""
    if not isinstance(opcoes, list) or not opcoes:
        return None
    if isinstance(opcoes[0], list):
        return [str(item) for item in opcoes[0]]
    if (
        opcoes[0] == "COMBO"
        and len(opcoes) > 1
        and isinstance(opcoes[1], dict)
        and isinstance(opcoes[1].get("options"), list)
    ):
        return [str(item) for item in opcoes[1]["options"]]
    return None


def validar_workflow(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
) -> list[str]:
    """Compara o grafo com o `/object_info` real do servidor ComfyUI."""
    problemas: list[str] = []
    for node_id, no in workflow.items():
        classe = str(no.get("class_type"))
        definicao = object_info.get(classe)
        if definicao is None:
            problemas.append(f"nó {node_id}: classe '{classe}' não existe no ComfyUI")
            continue
        schema = definicao.get("input") or {}
        obrigatorias = schema.get("required") or {}
        declaradas = {**obrigatorias, **(schema.get("optional") or {})}
        informadas = no.get("inputs") or {}

        for entrada, valor in informadas.items():
            if entrada not in declaradas:
                problemas.append(
                    f"nó {node_id} ({classe}): entrada '{entrada}' não existe no schema"
                )
                continue
            opcoes = _opcoes_combo(declaradas[entrada])
            if opcoes is not None and isinstance(valor, str) and valor not in opcoes:
                amostra = ", ".join(opcoes[:5]) or "nenhuma"
                problemas.append(
                    f"nó {node_id} ({classe}): valor '{valor}' indisponível para "
                    f"'{entrada}' (primeiras opções: {amostra})"
                )

        for entrada in obrigatorias:
            if entrada not in informadas:
                problemas.append(
                    f"nó {node_id} ({classe}): entrada obrigatória '{entrada}' ausente"
                )
    return problemas


def _imprimir_modelos(integros: list[Path], vazios: list[Path], models_dir: Path) -> None:
    print(f"\n== Modelos em {models_dir} ==")
    print(f"Íntegros: {len(integros)} | Corrompidos (0 bytes): {len(vazios)}")
    for arquivo in vazios:
        print(f"  [VAZIO] {arquivo.relative_to(models_dir)}")
    principais = [a for a in integros if a.stat().st_size >= 100 * 1024 * 1024]
    for arquivo in principais[:25]:
        tamanho_gb = arquivo.stat().st_size / (1024**3)
        print(f"  [OK]    {arquivo.relative_to(models_dir)} ({tamanho_gb:.2f} GB)")


def _verificar_criticos(config: dict[str, Any], models_dir: Path) -> list[str]:
    """Confere se os modelos exigidos pelo workflow existem e não estão vazios."""
    problemas: list[str] = []
    exigidos = config.get("modelos") or {}
    for chave in ("checkpoint", "text_encoder", "reactor_swap", "reactor_restore"):
        nome = exigidos.get(chave)
        if not nome:
            continue
        caminho = localizar_modelo(models_dir, str(nome))
        if caminho is None:
            problemas.append(f"modelo '{nome}' ({chave}) não encontrado em {models_dir}")
            continue
        try:
            exigir_arquivo_integro(caminho, minimo_bytes=1024 * 1024, descricao=chave)
        except (FileNotFoundError, ValueError) as erro:
            problemas.append(str(erro))
        else:
            print(f"  [OK]    {chave}: {caminho.name}")
    return problemas



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnóstico de modelos/ComfyUI e geração do workflow_api.json"
    )
    parser.add_argument("--projeto", default="config/projeto.json")
    parser.add_argument(
        "--models-dir",
        default="~/Documentos/projetos/IAs/ComfyUI/models",
        help="Diretório models/ do ComfyUI",
    )
    parser.add_argument("--workflow", default=None, help="Sobrescreve config.workflow_api")
    parser.add_argument("--url", default=None, help="Sobrescreve config.comfyui.url")
    parser.add_argument(
        "--escrever-workflow",
        action="store_true",
        help="Regrava o workflow_api.json a partir do config (mesmo se já existir)",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    log = configurar_logging(args.log_level)
    config, projeto_path, raiz = carregar_projeto(args.projeto)
    workflow_padrao = config.get("workflow_api", "scripts/workflow_api.json")
    workflow_path = resolver(raiz, args.workflow or workflow_padrao)
    models_dir = Path(args.models_dir).expanduser().resolve()

    print("== ai_video_studio :: diagnóstico de ambiente ==")
    print(f"Projeto : {projeto_path}")
    print(f"Workflow: {workflow_path}")

    problemas: list[str] = []

    try:
        integros, vazios = auditar_modelos(models_dir)
    except FileNotFoundError as erro:
        log.error("%s", erro)
        return 1
    _imprimir_modelos(integros, vazios, models_dir)

    print("\n== Modelos exigidos pelo projeto ==")
    problemas.extend(_verificar_criticos(config, models_dir))

    if args.escrever_workflow or not workflow_path.exists():
        workflow = construir_workflow_template(config)
        salvar_json_atomico(workflow_path, workflow)
        log.info("workflow_api.json gravado (%d nós) em %s", len(workflow), workflow_path)
    else:
        workflow = carregar_json(workflow_path)
        log.info("workflow_api.json existente carregado (%d nós)", len(workflow))

    referencia = str(config.get("referencia_facial") or "")
    caminho_ref = resolver(raiz, referencia) if referencia else None
    if caminho_ref:
        if caminho_ref.is_file() and caminho_ref.stat().st_size > 0:
            print(f"\nReferência facial: {caminho_ref}")
            pasta_input = (config.get("comfyui") or {}).get("input_dir")
            if pasta_input:
                destino_dir = resolver(raiz, pasta_input)
                if destino_dir.is_dir():
                    destino = destino_dir / caminho_ref.name
                    if (
                        not destino.exists()
                        or destino.stat().st_size != caminho_ref.stat().st_size
                    ):
                        shutil.copy2(caminho_ref, destino)
                        log.info("Referência facial sincronizada em %s", destino)
                else:
                    log.warning(
                        "Pasta input do ComfyUI inexistente (%s): o ReActor não "
                        "encontrará a referência.",
                        destino_dir,
                    )
        else:
            problemas.append(f"referência facial ausente/vazia: {caminho_ref}")

    url = str(
        args.url or (config.get("comfyui") or {}).get("url") or "http://127.0.0.1:8188"
    ).rstrip("/")
    try:
        stats = consultar_json(f"{url}/system_stats", args.timeout)
        object_info = consultar_json(f"{url}/object_info", args.timeout)
        dispositivo = (stats.get("devices") or [{}])[0]
        vram_gb = float(dispositivo.get("vram_total") or 0) / (1024**3)
        log.info(
            "ComfyUI online em %s | ComfyUI %s | GPU %s (VRAM %.1f GB)",
            url,
            (stats.get("system") or {}).get("comfyui_version", "?"),
            dispositivo.get("name", "?"),
            vram_gb,
        )
        faltantes = [c for c in NODE_CLASSES_OBRIGATORIOS if c not in object_info]
        if faltantes:
            problemas.append(
                "nós ausentes no ComfyUI: " + ", ".join(faltantes)
                + " (instale ComfyUI-ReActor ou atualize o ComfyUI)"
            )
        problemas.extend(validar_workflow(workflow, object_info))
    except (error.URLError, TimeoutError, OSError, ValueError) as erro:
        log.warning(
            "ComfyUI indisponível em %s (%s). Executando apenas auditoria offline.",
            url,
            erro,
        )
        print(
            "\nDica: inicie o servidor com\n"
            "  cd ~/Documentos/projetos/IAs/ComfyUI && ./venv/bin/python main.py --lowvram"
        )

    print("\n== Resultado ==")
    if problemas:
        for problema in problemas:
            print(f"  [PROBLEMA] {problema}")
        print(f"\n{len(problemas)} problema(s) encontrado(s).")
        return 1
    print("  Ambiente consistente para gerar as cenas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

