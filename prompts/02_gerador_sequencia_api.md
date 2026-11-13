# Tarefa 02: Refatoração do `scripts/gerador_sequencia.py` (Geração Iterativa via API)

## Objetivo
Transformar `scripts/gerador_sequencia.py` em um cliente resiliente da API do ComfyUI que lê `config/projeto.json`, injeta prompts dinamicamente no `scripts/workflow_api.json` e aguarda a conclusão real da renderização antes de avançar.

## Requisitos Técnicos:
1. **Leitura de Configuração:**
   - Ler `config/projeto.json` e processar apenas cenas com `"status": "pendente"` (permitindo reprocessar apenas cenas rejeitadas sem refazer o vídeo todo).
2. **Injeção Dinâmica no Grafo (`workflow_api.json`):**
   - Localizar automaticamente os nós `CLIPTextEncode` (positivo e negativo), `KSampler` (para injetar a `seed` e parâmetros de passos/CFG), nó de latente (para garantir resolução leve compatível com 6 GB VRAM) e nó de saída (`SaveAnimatedWEBP` ou `VHS_VideoCombine`).
   - Se `aplicar_reactor` estiver ativo na cena, configurar o nó do ReActor apontando para `assets/referencias/personagem_base.png`.
3. **Monitoramento Real de Fila (Sem `time.sleep` cego):**
   - Enviar o payload para `POST http://127.0.0.1:8188/prompt`.
   - Fazer polling inteligente em `GET http://127.0.0.1:8188/history/{prompt_id}` (ou conectar via `websocket-client`) até a tarefa concluir ou falhar.
   - Baixar/mover o arquivo gerado da pasta `output/` do ComfyUI para `inputs/projeto_atual/{cena_id}_raw.webp` (ou `.mp4`).
   - Atualizar o status da cena no `config/projeto.json` para `"gerado"`.
