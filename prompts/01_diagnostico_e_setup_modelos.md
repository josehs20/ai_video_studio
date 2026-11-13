# Tarefa 01: Diagnóstico de Modelos e Preparação do Workflow API

## Objetivo
Garantir que os modelos do ComfyUI utilizados pelo pipeline estejam íntegros (não sejam ponteiros Git LFS de 0 bytes) e configurar a leitura dinâmica do `scripts/workflow_api.json`.

## Instruções para o Cline executar:
1. **Auditoria de Modelos:**
   - Verifique o diretório `~/Documentos/projetos/IAs/ComfyUI/models/` em busca de arquivos `.safetensors`, `.onnx` ou `.pth` com tamanho igual a `0` bytes (como ocorreu com `wan2.1_t2v_1.3b.safetensors` e `wan2.1_t2v_1.3B_fp8.safetensors`).
   - Liste quais modelos estão íntegros (ex: `ltx-video-2b-v0.9.1.safetensors`, `umt5_xxl_fp8_e4m3fn.safetensors`, `t5xxl_fp8_e4m3fn.safetensors`, `wan_2.1_vae.safetensors`, `inswapper_128.onnx`, `GFPGANv1.4.pth`).
2. **Validação do `workflow_api.json`:**
   - Caso `scripts/workflow_api.json` ainda não exista, crie um script auxiliar `scripts/validar_ambiente.py` que:
     - Testa a conexão com `http://127.0.0.1:8188/system_stats`.
     - Consulta `/object_info` do ComfyUI para verificar os nós exatos disponíveis para LTX-Video ou Wan 2.1 e ReActor.
     - Gera um template válido de `scripts/workflow_api.json` compatível com os nós instalados e os arquivos íntegros encontrados.
3. **Atualização do `config/projeto.json`:**
   - Estruture `config/projeto.json` para suportar controle de estado por cena (`status`: `"pendente"`, `"gerado"`, `"polido"`, `"aprovado"`), `seed`, `duracao_segundos`, `prompt`, `negative_prompt` e flag `aplicar_reactor`.
