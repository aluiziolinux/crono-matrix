# llama.cpp fixado

- Repositório: `https://github.com/ggml-org/llama.cpp.git`
- Revisão-base upstream: `d230ddd763ffe27781c7ffd237ea78b639b36b6d`
- Data da revisão-base: `2026-09-03T23:53:04+02:00`
- Patch local: `patches/llama.cpp/crono-matrix.patch`
- SHA-256 do patch: `3a71b9d6d957377556f0c24d29ee46aa561ff54d022fb0d90fa91815a731a22e`

O patch contém tanto os commits locais posteriores/divergentes quanto as
alterações ainda não consolidadas do checkout de desenvolvimento na data de
montagem. A base escolhida pertence ao histórico público do upstream; o commit
de merge local não foi usado como dependência remota. Arquivos não rastreados
ficam em `patches/llama.cpp/overlay/` e são copiados depois da aplicação.

Não atualize a revisão sem primeiro regenerar o patch, executar `git apply
--check` contra um checkout limpo e validar geração, ferramentas, visão,
raciocínio, contexto longo, RAM e VRAM no hardware-alvo.
