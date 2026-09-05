---
name: Teste, desempenho ou problema / Test report
about: Relate um teste reproduzível e ajude a ampliar a cobertura de hardware/modelos.
title: "[Teste] "
labels: ""
assignees: ""
---

## O que aconteceu / What happened

Resultado esperado e observado. Informe se é bug, teste de compatibilidade ou medição.

## Ambiente / Environment

- Sistema e versão / OS:
- CPU, núcleos e threads:
- RAM total e disponível:
- GPU, VRAM livre e driver:
- Disco/swap, se relevante:
- Commit Crono Matrix:
- Saída de `llama-server --version`:
- Interface: desktop / Web / WebUI llama.cpp / harness (nome/versão):

## Modelo e configuração / Model and configuration

- Repositório de origem e nome do GGUF (não anexe pesos):
- Quantização e MMProj, se usado:
- Contexto solicitado e efetivo:
- K/V, offload, CPU MoE, batch/ubatch e threads:
- Sampling, seed, esforço/budget de raciocínio:
- Preview/comando efetivo **sem tokens, chaves ou caminhos pessoais**:

## Reprodução / Reproduction

1.
2.
3.

Prompt sintético ou exemplo público mínimo. Não publique conversas privadas.

## Evidência / Evidence

Logs sanitizados, erros completos e, se mediu desempenho: tamanho do prompt,
tokens gerados, cache frio/quente, repetições e outras cargas ativas.
Separe resultado medido de estimativa. Compare qualidade além de tokens/s.
