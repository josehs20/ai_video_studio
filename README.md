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
assets/referencias/          imagens de referência (facial e por cena)
inputs/projeto_atual/        clipes brutos ({cena}_raw.mp4)
temp/candidatos/{cena}/      variações multi-seed (seed_<n>.mp4)
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
# 1. Diagnóstico (modelos, nós, referências por cena, ComfyUI online/offline)
python3 scripts/validar_ambiente.py

# 1b. Regravar o grafo a partir do config (img2video se houver referência)
python3 scripts/validar_ambiente.py --escrever-workflow

# 2. Gerar cenas pendentes (N candidatos por cena + escolha automática)
python3 scripts/gerador_sequencia.py

# 2b. Trocar a escolha sem re-renderizar (promove temp/candidatos/{cena}/seed_N)
python3 scripts/gerador_sequencia.py --escolher-seed 445 --cena cena_04

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

### Cena com imagem de referência (img2video)

Basta apontar `imagem_referencia` na cena — o gerador converte o grafo para o
template `ltxv_image_to_video` (`LoadImage` -> `LTXVImgToVideo`) apenas naquela
cena e mantém as demais em text2video:

```json
{
  "id": "cena_04",
  "prompt": "Close-up angle on a dark mode laptop screen ...",
  "seed": 444,
  "frames": 33,
  "imagem_referencia": "assets/referencias/image.png",
  "forca_referencia": 0.7,
  "aplicar_reactor": false
}
```

- A imagem é validada com `ffprobe` e copiada para `ComfyUI/input/` como
  `avs_ref_{cena_id}.png` (nome estável, sem colisão entre cenas).
- `forca_referencia` (0–1) é o `strength` do `LTXVImgToVideo`: valores baixos
  deixam o modelo inventar mais, altos prendem o clipe ao print.
- `--t2v` no `validar_ambiente.py` força o grafo puro text2video (ignora as
  referências); `--i2v` faz o contrário e exige ao menos uma cena com imagem.

### Múltiplos candidatos por cena

```bash
python3 scripts/gerador_sequencia.py --cena cena_04 --candidatos 3
python3 scripts/gerador_sequencia.py --escolher-seed 446 --cena cena_04
```

Cada seed vira `temp/candidatos/cena_04/seed_444.mp4` (444, 445, 446...), é
validada (duração, movimento, detalhe, estabilidade) e registrada em
`cenas[].candidatos`. O melhor score é promovido automaticamente para
`cena_04_raw.mp4`. Para trocar a escolha depois, sem re-renderizar:

- `--escolher-seed N --cena cena_04` promove exatamente aquela cena;
- `--escolher-seed N` sem `--cena` promove apenas as cenas que realmente têm
  `seed_N` em `temp/candidatos` (e falha com mensagem clara se nenhuma tiver).

### Flags úteis

| Comando | Flag | Efeito |
|---|---|---|
| `validar_ambiente.py` | `--escrever-workflow` | Regrava `workflow_api.json` a partir do config |
| `validar_ambiente.py` | `--i2v` / `--t2v` | Força o grafo img2video ou text2video |
| `gerador_sequencia.py` | `--dry-run` | Mostra modo/seed/resolução/força por cena sem enviar |
| `gerador_sequencia.py` | `--cena cena_03 --forcar` | Regera uma cena específica |
| `gerador_sequencia.py` | `--candidatos N` | Variações por cena (sobrescreve `candidatos_por_cena`) |
| `gerador_sequencia.py` | `--escolher-seed N` | Promove `temp/candidatos/{cena}/seed_N` a `_raw` |
| `gerador_sequencia.py` | `--ignorar-limite-vram` | Permite config acima do orçamento seguro |
| `polimento.py` | `--interpolador minterpolate` | Força fallback FFmpeg |
| `polimento.py` | `--escala`, `--fps`, `--crf` | Sobrescreve parâmetros do config |
| `polimento.py` | `--nitidez`, `--reducao-ruido`, `--sem-filtros` | Ajusta `unsharp`/`hqdn3d`/micro-contraste |
| `orchestrator.py` | `--etapa escolher --escolher-seed N` | Troca o clipe escolhido sem regerar |
| `orchestrator.py` | `--candidatos N` | Repassa o número de candidatos para a etapa `gerar` |
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

`escolher` não cria estado novo: ao promover outro candidato a `_raw` a cena volta
para `gerado` (o `arquivo_polido` antigo fica desatualizado), então rode
`--etapa polir` em seguida.

## Configuração (`config/projeto.json`)

- `comfyui.url`, `comfyui.output_dir`, `comfyui.input_dir`
- `modelos.*`: nomes exatos dos arquivos usados pelo workflow
- `geracao`: `largura`/`altura` (múltiplos de 32, ≤512×320 no 6 GB), `frames`
  (normalizados para `8k+1`; padrão 33), `fps`, `steps` (30–40), `cfg`,
  `forca_referencia` (0–1), `candidatos_por_cena`, `timeout_segundos` (7200),
  `timeout_requisicao_segundos` (180), `intervalo_polling_segundos`
- `polimento`: `fps_alvo`, `escala_upscale`, `modelo_escala`, `interpolador`,
  `nitidez` (unsharp, 0 desliga), `reducao_ruido` (hqdn3d, 0 desliga),
  `micro_contraste` (eq de contraste/saturação)
- `montagem`: `largura`/`altura` finais (pares), `fps`, `transicao`, `crf`, `preset`
- `cenas[]`: `id`, `prompt`, `negative_prompt`, `seed`, `frames`,
  `imagem_referencia`, `forca_referencia`, `aplicar_reactor`, `candidatos_por_cena`,
  `duracao_segundos`, `status`

Caminhos relativos são resolvidos a partir da raiz do projeto (diretório acima de
`config/`); `~` é expandido. O JSON é gravado de forma atômica, então um
`Ctrl+C` no meio da execução não corrompe o estado.

## Notas técnicas e limitações

- **Saída do ComfyUI em MP4**: o template usa `CreateVideo` + `SaveVideo`
  (H.264). O FFmpeg 8.0.1 deste ambiente **não decodifica WebP animado**
  (`image data not found`), então `SaveAnimatedWEBP` não serve como bruto; se uma
  cena antiga em `.webp` for encontrada, o polimento falha com mensagem explícita.
- **Grafo LTX-Video** segue os templates oficiais do ComfyUI. Text2Video
  (`ltxv_text_to_video`): `CLIPLoader` (`type=ltxv`) para o T5-XXL,
  `CheckpointLoaderSimple` para MODEL/VAE, `EmptyLTXVLatentVideo`,
  `LTXVConditioning`, `LTXVScheduler`, `KSamplerSelect`, `SamplerCustom`,
  `VAEDecode`. Img2Video (`ltxv_image_to_video`): `LoadImage` ->
  `LTXVImgToVideo`, cujos outputs 0/1/2 alimentam `LTXVConditioning`
  (positive/negative), `LTXVScheduler` e `SamplerCustom.latent_image`. O payload
  antigo (`EmptyLatentImage` + `length`) era inválido.
- **`frames` sempre `8k+1`**: o VAE do LTX comprime o tempo em blocos de 8, então
  `frames: 31` sai como 25. O pipeline normaliza para o múltiplo mais próximo
  (31 -> 33) antes de injetar, evitando divergência entre config e saída.
- **Loop multi-seed**: `geracao.candidatos_por_cena` (ou `--candidatos N`) renderiza
  `seed`, `seed+1`, ... em `temp/candidatos/{cena}/`, mede movimento (YAVG/YDIF via
  `signalstats`), detalhe e estabilidade, e promove o maior score — desempate pela
  menor seed. Os registros ficam em `cenas[].candidatos[]` (`pontuacao`, `valido`,
  `motivos`, `escolhido`), com `seed_escolhida` no nível da cena.
- **Travas de VRAM**: `geracao` acima de 512×320, 49 frames ou `steps > 40` gera
  aviso e o gerador aborta a cena com mensagem explícita, a menos que se use
  `--ignorar-limite-vram`.
- **Fallbacks obrigatórios**: sem `rife-ncnn-vulkan` usa-se `minterpolate`; sem
  `realesrgan-ncnn-vulkan` usa-se `scale` lanczos (o upscale fica por conta do
  FFmpeg). O `validar_ambiente.py` reporta qual caminho está ativo.
- **GFPGAN não é super-resolução**: ele restaura rostos dentro do ReActor. Não é
  usado como upscale.
- **Transição `fade`**: fade-in/out para preto em cada clipe (mantém a duração).
  Não é crossfade (`xfade`), que exigiria recalcular offsets por clipe.
- **Sem áudio**: o vídeo final é mudo (`-an`).
- **ReActor é opcional por cena**: `aplicar_reactor: false` remove o nó do grafo e
  religa os consumidores ao `VAEDecode` (bypass real, sem tocar nos frames), e a
  `LoadImage` da referência facial órfã é descartada. A `referencia_facial` só é
  cobrada na validação quando alguma cena tem `aplicar_reactor: true`.
- **ReActor v0.7.0-b1 tinha um bug** que zerava o lote de frames
  (`if False and ...` + `pil_images = pil_images_sfw`), devolvendo **1 frame
  preto 512×512**. Foi aplicado um patch local de 1 linha em
  `ComfyUI/custom_nodes/ComfyUI-ReActor/nodes.py` (backup em
  `nodes.py.bak_ai_video_studio`). Se o ReActor for atualizado, reaplique o
  patch, senão as cenas com `aplicar_reactor: true` sairão com 1 frame.
- **Polling resiliente**: o ComfyUI bloqueia o event loop durante nós pesados
  (ReActor/GFPGAN), então `/history` pode ficar sem responder por dezenas de
  segundos. O gerador tolera timeouts transitórios
  (`geracao.timeout_requisicao_segundos`, padrão 120) sem abortar a cena.
- **Cache do ComfyUI**: repetir o mesmo grafo/seed retorna o resultado em
  milissegundos. Use `--nova-seed` (via `orchestrator.py`) para forçar um novo
  render.

## Desempenho medido (RTX 3050 6 GB, `--lowvram`)

| Cenário | Tempo |
|---|---|
| LTX-Video 448×288, 25 frames, 20 steps | ~6 s de sampling (≈70 s com load inicial) |
| ReActor (25 frames) + GFPGAN | ~65–90 s por cena |
| Polimento (`minterpolate` + lanczos) | ~3 s por cena |
| Montagem final (3 cenas, 1024×640) | ~1 s |

- **VRAM**: se houver OOM, reduza `geracao.frames` (ex.: 25) ou a resolução
  (ex.: 448x288) antes de aumentar `steps`.

## Migração do config antigo

O `config/projeto.json` anterior (`clipes`, `fps_origem`, `escala_upscale`)
apontava para arquivos inexistentes e foi substituído pelo schema por cena. A
versão antiga continua acessível no histórico Git:

```bash
git show bcd6185:config/projeto.json
```
# ai_video_studio
