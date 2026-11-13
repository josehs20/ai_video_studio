# ai_video_studio

Pipeline local de geração de vídeo por IA com **ComfyUI + LTX-Video 2B**, pensado
para GPUs de 6 GB (RTX 3050). O estado vive em JSON, cada etapa valida o resultado
e o fluxo é orquestrado por uma máquina de estados por cena.

```
config/projeto.json          estado do projeto (cenas, seeds, status, caminhos)
scripts/common.py            utilitários: logging, JSON atômico, ffprobe, subprocesso
scripts/validar_ambiente.py  auditoria de modelos/ComfyUI + gera workflow_api.json
scripts/workflow_api.json    grafo LTX-Video em formato API (gerado/validado)
scripts/gerador_sequencia.py gera cenas no ComfyUI (polling real de /history)
scripts/polimento.py         interpolação + upscale + encode H.264
scripts/orchestrator.py      máquina de estados + montagem final
inputs/projeto_atual/        clipes brutos ({cena}_raw.mp4)
temp/                        intermediários e {cena}_polido.mp4
outputs/                     {nome_projeto}_final.mp4
bin/                         binários opcionais (rife/Real-ESRGAN)
```

## Pré-requisitos

1. **ComfyUI** (0.28+) com `ComfyUI-ReActor` em `~/Documentos/projetos/IAs/ComfyUI`,
   iniciado com modo de baixa VRAM:

   ```bash
   cd ~/Documentos/projetos/IAs/ComfyUI
   ./venv/bin/python main.py --lowvram
   ```

2. **Modelos** já presentes: `checkpoints/ltx-video-2b-v0.9.1.safetensors`,
   `clip/t5xxl_fp8_e4m3fn.safetensors`, `reactor/inswapper_128.onnx`,
   `facerestore_models/GFPGANv1.4.pth`.
3. **ffmpeg/ffprobe** no `PATH`.
4. Opcional: `rife-ncnn-vulkan` e `realesrgan-ncnn-vulkan` no `PATH` ou em `bin/`
   (sem eles o pipeline usa os fallbacks do FFmpeg).

## Uso

```bash
# 1. Diagnóstico (modelos, nós, referência facial, ComfyUI online/offline)
python3 scripts/validar_ambiente.py

# 2. Gerar cenas pendentes
python3 scripts/gerador_sequencia.py

# 3. Polir cenas geradas (RIFE/upscale -> temp/{cena}_polido.mp4)
python3 scripts/polimento.py

# 4. Montar o vídeo final
python3 scripts/orchestrator.py --etapa montar

# Tudo de uma vez
python3 scripts/orchestrator.py --etapa tudo

# Situação de cada cena
python3 scripts/orchestrator.py --etapa status
```

### Refazer apenas uma cena (preserva as demais)

```bash
python3 scripts/orchestrator.py --refazer-cena cena_03 --nova-seed 4242
```

A cena volta para `pendente`, os arquivos antigos vão para
`temp/_descartados/<data>/` e só ela é regenerada; depois o vídeo é remontado.

### Flags úteis

| Comando | Flag | Efeito |
|---|---|---|
| `validar_ambiente.py` | `--escrever-workflow` | Regrava `workflow_api.json` a partir do config |
| `gerador_sequencia.py` | `--dry-run` | Mostra seed/resolução por cena sem enviar |
| `gerador_sequencia.py` | `--cena cena_03 --forcar` | Regera uma cena específica |
| `polimento.py` | `--interpolador minterpolate` | Força fallback FFmpeg |
| `polimento.py` | `--escala`, `--fps`, `--crf` | Sobrescreve parâmetros do config |
| `orchestrator.py` | `--transicao fade` | Fade-in/out de 0,2 s por clipe |
| `orchestrator.py` | `--continuar-em-erro` | Segue para a próxima etapa mesmo com falha |

## Estados

```
pendente --gerar--> gerado --polir--> polido --(revisão manual)--> aprovado
   ^                   |                  |
   +------- erro <-----+------------------+   (erro guarda a mensagem em "erro")
```

A montagem aceita cenas `polido` ou `aprovado`; se faltar alguma, o erro lista
exatamente quais cenas estão pendentes.

## Configuração (`config/projeto.json`)

- `comfyui.url`, `comfyui.output_dir`, `comfyui.input_dir`
- `modelos.*`: nomes exatos dos arquivos usados pelo workflow
- `geracao`: `largura`/`altura` (múltiplos de 32), `frames` (25–33), `fps`, `steps`,
  `cfg`, `timeout_segundos`
- `polimento`: `fps_alvo`, `escala_upscale`, `modelo_escala`, `interpolador`
- `montagem`: `largura`/`altura` finais (pares), `fps`, `transicao`, `crf`, `preset`
- `cenas[]`: `id`, `prompt`, `negative_prompt`, `seed`, `frames`,
  `duracao_segundos`, `aplicar_reactor`, `status`

Caminhos relativos são resolvidos a partir da raiz do projeto (diretório acima de
`config/`); `~` é expandido. O JSON é gravado de forma atômica, então um
`Ctrl+C` no meio da execução não corrompe o estado.

## Notas técnicas e limitações

- **Saída do ComfyUI em MP4**: o template usa `CreateVideo` + `SaveVideo`
  (H.264). O FFmpeg 8.0.1 deste ambiente **não decodifica WebP animado**
  (`image data not found`), então `SaveAnimatedWEBP` não serve como bruto; se uma
  cena antiga em `.webp` for encontrada, o polimento falha com mensagem explícita.
- **Grafo LTX-Video** segue o template oficial do ComfyUI
  (`ltxv_text_to_video`): `CLIPLoader` (`type=ltxv`) para o T5-XXL,
  `CheckpointLoaderSimple` para MODEL/VAE, `EmptyLTXVLatentVideo`,
  `LTXVConditioning`, `LTXVScheduler`, `KSamplerSelect`, `SamplerCustom`,
  `VAEDecode`. O payload antigo (`EmptyLatentImage` + `length`) era inválido.
- **Fallbacks obrigatórios**: sem `rife-ncnn-vulkan` usa-se `minterpolate`; sem
  `realesrgan-ncnn-vulkan` usa-se `scale` lanczos (o upscale fica por conta do
  FFmpeg). O `validar_ambiente.py` reporta qual caminho está ativo.
- **GFPGAN não é super-resolução**: ele restaura rostos dentro do ReActor. Não é
  usado como upscale.
- **Transição `fade`**: fade-in/out para preto em cada clipe (mantém a duração).
  Não é crossfade (`xfade`), que exigiria recalcular offsets por clipe.
- **Sem áudio**: o vídeo final é mudo (`-an`).
- **VRAM**: se houver OOM, reduza `geracao.frames` (ex.: 25) ou a resolução
  (ex.: 448x288) antes de aumentar `steps`.

## Migração do config antigo

O `config/projeto.json` anterior (`clipes`, `fps_origem`, `escala_upscale`)
apontava para arquivos inexistentes e foi substituído pelo schema por cena. A
versão antiga continua acessível no histórico Git:

```bash
git show bcd6185:config/projeto.json
```
