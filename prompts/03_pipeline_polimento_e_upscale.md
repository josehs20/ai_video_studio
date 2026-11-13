# Tarefa 03: Algoritmo de Lapidação e Polimento (`scripts/polimento.py`)

## Objetivo
Criar o módulo responsável por transformar os clipes brutos de baixa resolução e baixo FPS (`inputs/projeto_atual/*_raw.*`) em clipes fluidos e nítidos em `temp/` sem estourar a VRAM.

## Etapas do Fluxo Encadeado:
1. **Conversão e Extração de Frames (FFmpeg):**
   - Converter o arquivo bruto `.webp` ou `.mp4` para sequência de frames PNG ou MP4 intermediário padronizado.
2. **Interpolação de Quadros (RIFE):**
   - Verificar se há binário `rife-ncnn-vulkan` em `bin/` ou pacote Python instalado.
   - Dobrar a taxa de quadros (ex: de 12/15 fps para 24/30 fps) para eliminar engasgos visuais.
   - Caso o RIFE não esteja disponível no momento, implementar fallback automático usando filtro `minterpolate` do FFmpeg.
3. **Super-Resolução / Upscale (Real-ESRGAN):**
   - Verificar se há binário `realesrgan-ncnn-vulkan` em `bin/` (ou usar nó de upscale via ComfyUI com `GFPGANv1.4.pth` / `codeformer-v0.1.0.pth` já presentes na pasta `facerestore_models`).
   - Escalonar de 480p para 720p/1080p mantendo nitidez facial e de cenário.
4. **Atualização de Estado:**
   - Salvar o clipe lapidado em `temp/{cena_id}_polido.mp4` e marcar `"status": "polido"` no `config/projeto.json`.
