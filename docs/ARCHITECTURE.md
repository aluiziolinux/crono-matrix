# Arquitetura

```text
CustomTkinter ─┐
               ├─ LauncherWebState ── leitura GGUF / hardware / autotune
FastAPI/HTMX ──┘          │
                          ├─ OptimalParams ── comando validado
                          ├─ monitor C99 / cgroup / swap NVMe
                          └─ llama-server ── /health /props /tools /v1
```

`launch_model_core.py` concentra metadados, inventário de hardware e cálculo.
`web/services.py` mantém estado, valida formulários, gerencia processos e
confirma o runtime real. A interface Web e `launch_model_ctk.py` chamam esse
mesmo serviço; não possuem um segundo algoritmo de parâmetros.

O valor mostrado como efetivo só deve ser promovido depois de `/health` e
`/props`. O comando de preview é uma projeção; o snapshot de runtime é a fonte
de verdade após o carregamento.

O checkout `llama.cpp/` é dependência reconstruível e não vendorizada. A
revisão, patch e arquivos novos ficam sob `third_party/` e `patches/`.

O servidor MCP privado do Crono Matrix não integra esta edição. Configurações
genéricas de MCP externo podem ser repassadas ao recurso upstream do
`llama-server`, mas permanecem desligadas por padrão. Ferramentas nativas como
leitura, shell e `browser_playwright` não dependem de MCP.
