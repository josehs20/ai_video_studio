import os
import json
import argparse
import subprocess
from pathlib import Path

def executar_cmd(cmd):
    print(f"\n[EXECUTANDO]: {cmd}")
    resultado = subprocess.run(cmd, shell=True)
    if resultado.returncode != 0:
        print(f"[ERRO CRÍTICO] Falha ao executar o comando:\n{cmd}")
        exit(1)

def buscar_videos_pasta(pasta_entrada):
    extensoes_validas = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
    pasta = Path(pasta_entrada)
    
    if not pasta.exists():
        print(f"[ERRO] A pasta de entrada '{pasta_entrada}' não foi encontrada.")
        exit(1)

    # Busca todos os vídeos compatíveis e ordena alfabeticamente (ex: cena_01, cena_02...)
    arquivos = sorted([str(f.resolve()) for f in pasta.iterdir() if f.suffix.lower() in extensoes_validas])
    
    return arquivos

def montar_lista_ffmpeg(clipes, arquivo_txt_saida):
    with open(arquivo_txt_saida, 'w', encoding='utf-8') as f:
        for clipe in clipes:
            f.write(f"file '{clipe}'\n")

def main():
    parser = argparse.ArgumentParser(description="Orquestrador de Vídeo IA - Estúdio Local")
    parser.add_argument("--input", required=True, help="Caminho da pasta contendo os clipes brutos (ex: inputs/projeto_01)")
    parser.add_argument("--nome", default="projeto_output", help="Nome base para os arquivos temporários e finais")
    parser.add_argument("--escala", type=int, default=2, help="Fator de upscale do Real-ESRGAN (ex: 2)")
    parser.add_argument("--modelo", default="realesr-animevideov3", help="Modelo do Real-ESRGAN")
    
    args = parser.parse_args()

    nome = args.nome
    pasta_temp = Path(f"temp/{nome}")
    pasta_out = Path("outputs")
    pasta_temp.mkdir(parents=True, exist_ok=True)
    pasta_out.mkdir(parents=True, exist_ok=True)

    print(f"=== Iniciando Pipeline para o projeto: {nome} ===")
    print(f"Buscando clipes na pasta: {args.input}")

    # ETAPA 1: Varre a pasta informada e pega os vídeos automaticamente
    clipes = buscar_videos_pasta(args.input)
    
    if not clipes:
        print(f"[ERRO] Nenhum vídeo compatível encontrado em '{args.input}'.")
        return

    print(f"Clipes encontrados ({len(clipes)}):")
    for c in clipes:
        print(f"  - {os.path.basename(c)}")

    lista_txt = pasta_temp / "lista_clips.txt"
    video_concatenado = pasta_temp / "1_concatenado.mp4"
    montar_lista_ffmpeg(clipes, lista_txt)
    
    cmd_concat = f"ffmpeg -y -f concat -safe 0 -i {lista_txt} -c copy {video_concatenado}"
    executar_cmd(cmd_concat)

    # ETAPA 2: Upscaling com Real-ESRGAN (NCNN)
    video_upscaled = pasta_temp / "2_upscaled.mp4"
    realesrgan_bin = "realesrgan-ncnn-vulkan" 
    
    cmd_upscale = (
        f"{realesrgan_bin} -i {video_concatenado} -o {video_upscaled} "
        f"-n {args.modelo} -s {args.escala}"
    )
    executar_cmd(cmd_upscale)

    # ETAPA 3: Interpolação de Quadros com RIFE (NCNN)
    video_interpolated = pasta_temp / "3_interpolated.mp4"
    rife_bin = "rife-ncnn-vulkan"
    
    cmd_rife = f"{rife_bin} -i {video_upscaled} -o {video_interpolated}"
    executar_cmd(cmd_rife)

    # ETAPA 4: Ajuste Final com FFmpeg (H.264)
    video_final = pasta_out / f"{nome}_final.mp4"
    cmd_final = (
        f"ffmpeg -y -i {video_interpolated} "
        f"-c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p {video_final}"
    )
    executar_cmd(cmd_final)

    print(f"\n[SUCESSO] Vídeo final gerado com sucesso em: {video_final}")

if __name__ == "__main__":
    main()