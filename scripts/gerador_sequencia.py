import os
import json
import urllib.request
import time
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188/prompt"

ROTEIRO_ANUNCIO = [
    {
        "id": "cena_01",
        "prompt": "Close-up portrait of a tech developer talking to the camera, modern office background, cinematic lighting",
    },
    {
        "id": "cena_02",
        "prompt": "Medium shot of the same developer pointing at a glowing monitor displaying software code",
    },
    {
        "id": "cena_03",
        "prompt": "Close-up of hands typing rapidly on a mechanical keyboard, dark aesthetic, RGB lights",
    },
    {
        "id": "cena_04",
        "prompt": "Wide shot of the developer smiling, giving a thumbs up, bright modern office",
    }
]

def enviar_prompt_api(prompt_texto):
    # Payload estruturado para utilizar o LTX-Video (que já está íntegro na sua máquina)
    payload = {
        "prompt": {
            "3": {
                "inputs": {
                    "seed": 42,
                    "steps": 25,
                    "cfg": 3.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "ltx-video-2b-v0.9.1.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 320,  # Resolução paisagem otimizada para os 6 GB de VRAM
                    "length": 33,   # Quantidade de quadros por clipe
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage" # Ou nó de latente de vídeo do LTX se instalado
            },
            "6": {
                "inputs": {
                    "text": prompt_texto,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "worst quality, inconsistent motion, blurry, jittery",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "AnuncioLTX",
                    "fps": 24,
                    "lossless": False,
                    "quality": 80,
                    "method": "default",
                    "images": ["8", 0]
                },
                "class_type": "SaveAnimatedWEBP"
            }
        }
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(COMFYUI_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode('utf-8'))
        return result.get("prompt_id")
    except urllib.error.HTTPError as e:
        erro_detalhe = e.read().decode('utf-8')
        print(f"[ERRO HTTP] Detalhes do ComfyUI: {erro_detalhe}")
        return None
    except Exception as e:
        print(f"[ERRO] Falha ao comunicar com a API: {e}")
        return None

def main():
    pasta_destino = "inputs/projeto_atual"
    os.makedirs(pasta_destino, exist_ok=True)

    print("=== Iniciando Geração Sequencial com LTX-Video via API ===")
    
    for cena in ROTEIRO_ANUNCIO:
        print(f"\n[Processando] {cena['id']}...")
        print(f"-> Prompt: {cena['prompt']}")
        
        prompt_id = enviar_prompt_api(cena['prompt'])
        if prompt_id:
            print(f"[Enviado com Sucesso] ID da tarefa: {prompt_id}")
        
        time.sleep(3)

    print("\n[SUCESSO] Fila enviada para o ComfyUI!")

if __name__ == "__main__":
    main()