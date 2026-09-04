# Contribuição

Contribuições são bem-vindas por issues e pull requests. Nos termos da seção 5
da Apache License 2.0, contribuições enviadas intencionalmente para inclusão no
projeto são licenciadas sob os mesmos termos, salvo declaração explícita em
contrário.

Antes de enviar uma mudança:

1. não adicione modelos, builds ou dados de usuário;
2. preserve a separação entre cálculo, estado/processos e interfaces;
3. prove mudanças de desempenho com medições reproduzíveis;
4. execute `make test` e `make release-check`;
5. registre qualquer alteração no patch de `llama.cpp`, incluindo revisão-base.

Mudanças no autotuning precisam informar hardware, modelo, comando efetivo e
medição antes/depois. Não inclua GGUF, MMProj, credenciais, histórico local ou
capturas contendo dados pessoais.
