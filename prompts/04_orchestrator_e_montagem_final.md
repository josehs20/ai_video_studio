# Tarefa 04: Orquestrador Mestre e Montagem Final (`scripts/orchestrator.py`)

## Objetivo
Implementar a máquina de estados completa em `scripts/orchestrator.py` para executar o ciclo ponta a ponta (Geração -> Lapidação -> Montagem Final) e permitir o refinamento seletivo de cenas.

## Requisitos Técnicos:
1. **CLI Interativo / Argumentos (`argparse`):**
   - `--projeto`: Caminho para o JSON do projeto (padrão: `config/projeto.json`).
   - `--etapa`: Permite rodar `tudo`, `gerar`, `polir` ou `montar`.
   - `--refazer-cena`: Ex: `--refazer-cena cena_02 --nova-seed 9999` para lapidar apenas um trecho específico que não ficou bom, preservando os demais clipes aprovados.
2. **Montagem Final com FFmpeg (`outputs/`):**
   - Ler todos os clipes em `temp/*_polido.mp4` na ordem exata definida em `config/projeto.json`.
   - Normalizar resolução, codec (`libx264`, `yuv420p`, `crf 18`), framerate e aplicar transições suaves ou corte seco preciso.
   - Exportar o vídeo final consolidado de ~20 segundos em `outputs/{nome_projeto}_final.mp4`.
3. **Preparação para Futura Plataforma SaaS:**
   - Manter a lógica de negócio desacoplada da interface CLI (funções puras que recebem dicionários de configuração e retornam caminhos/status), facilitando a futura exposição via FastAPI + Redis Queue.
