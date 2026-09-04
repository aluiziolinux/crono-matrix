# Integração com llama.cpp

O produto fixa uma revisão exata porque a integração usa APIs experimentais do
servidor e patches na WebUI/ferramentas. O fluxo reproduzível é:

1. clonar `ggml-org/llama.cpp`;
2. fazer checkout da revisão em `third_party/llama.cpp/REVISION`;
3. validar/aplicar `patches/llama.cpp/crono-matrix.patch`;
4. copiar `patches/llama.cpp/overlay/`;
5. compilar no diretório `build-crono`.

O patch local inclui:

- ferramenta nativa `browser_playwright`;
- diretório de trabalho Unicode enviado no corpo da chamada de ferramenta;
- controles e sincronização de modelo/raciocínio na WebUI;
- contagem de `reasoning_tokens` nas respostas OpenAI-compatible;
- fontes experimentais de MetaHead preservadas no checkout local;
- falha de alocação MMProj reportada sem abortar o processo inteiro;
- testes upstream ampliados para esses comportamentos.

Uma atualização upstream exige merge manual. Nunca aplique `git pull` sobre o
checkout modificado como atualização de produto. Primeiro teste o patch contra
o novo commit, resolva o delta, regenere o SHA-256 e execute regressão real.
