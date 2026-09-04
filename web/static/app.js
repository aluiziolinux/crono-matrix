(() => {
  const maxLogLines = 1200;
  const baseDocumentTitle = document.title;
  let hfRadarUnread = null;
  const parameterHelp = {
    ctx: "Janela total de contexto em tokens, compartilhada pelos slots. Valores maiores consomem mais KV cache.",
    ngl: "Camadas do modelo mantidas na GPU. 'all' força offload completo; 'auto' permite ao llama.cpp ajustar.",
    cache_k: "Precisão do cache de chaves (K). Menos bits aumentam o contexto disponível, com possível perda pequena de qualidade.",
    cache_v: "Precisão do cache de valores (V). Normalmente deve acompanhar K; V costuma ser mais sensível à quantização.",
    flash: "Flash Attention reduz tráfego de memória e acelera atenção na GPU. 'on' é recomendado para CUDA.",
    parallel: "Quantidade de slots simultâneos. O contexto é dividido entre eles e o consumo de memória aumenta.",
    kv_offload: "Mantém o cache KV na GPU. Na CPU economiza VRAM, mas reduz fortemente a velocidade.",
    kv_unified: "Compartilha um buffer KV entre slots. Útil com múltiplos slots; desnecessário com apenas um.",
    load_mode: "Estratégia de leitura dos pesos: mmap, memória travada, cópia normal ou Direct I/O.",
    cont_batching: "Agrupa continuamente requisições de slots diferentes para melhorar throughput do servidor.",
    cache_prompt: "Reutiliza prefixos de prompts já processados, reduzindo o tempo até o primeiro token.",
    split_mode: "Como distribuir tensors entre GPUs. Em uma GPU, layer é o modo simples e recomendado.",
    device: "Backend/dispositivo usado no offload, por exemplo CUDA0. Vazio deixa o llama.cpp escolher.",
    poll: "Nível de busy-wait dos workers CPU (0–100). Maior pode reduzir latência, mas aumenta uso aparente de CPU.",
    repack: "Reorganiza pesos quantizados para kernels mais eficientes no hardware atual.",
    no_host: "Evita buffers host para alguns tensors. Pode poupar RAM, mas exige mais buffers de backend.",
    numa: "Política de memória para sistemas com vários nós NUMA: distribuir ou isolar por nó.",
    swa_full: "Reserva cache completo para Sliding Window Attention. Necessário apenas em arquiteturas compatíveis.",
    cache_reuse: "Quantidade mínima de tokens para reutilização/deslocamento de KV entre prompts semelhantes.",
    cache_idle_slots: "Mantém caches de slots ociosos disponíveis para reutilização posterior.",
    threads: "Threads CPU usadas na geração e em operações não offloadadas. Em GPU, poucas threads costumam bastar.",
    threads_batch: "Threads CPU usadas no processamento de prompt/batch. Pode aproveitar todos os threads lógicos.",
    batch: "Batch lógico máximo do prompt. Maior melhora prefill, mas aumenta workspace e reduz contexto disponível.",
    ubatch: "Micro-batch físico enviado ao backend. Maior acelera prefill e consome mais VRAM temporária.",
    cpu_moe: "Mantém todos os especialistas MoE na CPU para economizar VRAM, com grande custo de velocidade.",
    n_cpu_moe: "Calculado automaticamente por modelo, VRAM e contexto; altere somente para benchmark.",
    temp: "Temperatura do sampling. Menor é mais determinístico; maior aumenta diversidade.",
    top_k: "Restringe cada escolha aos K tokens mais prováveis. Zero desativa.",
    top_p: "Nucleus sampling: mantém o menor conjunto cuja probabilidade acumulada alcança este valor.",
    min_p: "Remove tokens muito improváveis em relação ao token mais provável.",
    repeat_penalty: "Penaliza tokens repetidos. 1.0 desativa; valores altos podem degradar código.",
    presence_penalty: "Penaliza tokens que já apareceram, independentemente da frequência.",
    frequency_penalty: "Penaliza tokens proporcionalmente ao número de ocorrências anteriores.",
    repeat_last_n: "Janela de tokens usada para calcular penalidade de repetição.",
    seed: "Semente aleatória. -1 escolhe uma semente nova; valor fixo melhora reprodutibilidade.",
    ignore_eos: "Ignora o token de fim. Pode gerar indefinidamente até atingir outro limite.",
    sampler_seq: "Ordem dos samplers do llama.cpp; alterar a sequência muda a distribuição final.",
    dry_multiplier: "Força antirrepetição DRY. Zero desativa.",
    dry_base: "Base exponencial da penalidade DRY para sequências repetidas.",
    dry_allowed_length: "Comprimento repetido permitido antes de aplicar DRY.",
    dry_penalty_last_n: "Janela analisada pelo DRY. -1 usa todo o contexto disponível.",
    top_nsigma: "Filtra logits por distância em desvios padrão. Valor negativo desativa.",
    typical_p: "Typical sampling; valores abaixo de 1 filtram tokens semanticamente atípicos.",
    xtc_probability: "Probabilidade de aplicar XTC, que remove alternativas excessivamente prováveis.",
    xtc_threshold: "Limiar mínimo de probabilidade considerado pelo XTC.",
    dynatemp_range: "Amplitude da temperatura dinâmica. Zero mantém temperatura fixa.",
    dynatemp_exp: "Expoente que controla como a temperatura dinâmica reage à incerteza.",
    mirostat: "Controlador de perplexidade Mirostat: 0 desliga, 1 ou 2 selecionam a versão.",
    mirostat_lr: "Taxa de aprendizado do controlador Mirostat.",
    mirostat_ent: "Entropia/perplexidade alvo do Mirostat.",
    adaptive_target: "Alvo do sampler adaptativo. Valor negativo desativa.",
    adaptive_decay: "Velocidade de adaptação do sampler adaptativo.",
    reasoning: "Controla a geração de raciocínio: automático pelo template, ligado ou desligado.",
    reasoning_format: "Formato usado para separar raciocínio e resposta final na API.",
    reasoning_budget: "Limite de tokens de raciocínio. -1 deixa sem limite explícito.",
    reasoning_budget_message: "Mensagem inserida quando o orçamento de raciocínio é esgotado.",
    reasoning_preserve: "Preserva reasoning_content no histórico de conversas multiturno quando o template suporta.",
    chat_template_kwargs: "JSON passado a --chat-template-kwargs. Único caminho inequívoco para reasoning_effort low/medium/high do GPT-OSS — o campo OAI reasoning_effort só trata \"none\".",
    spec_type: "Método de speculative decoding. draft-mtp usa a cabeça MTP integrada ao modelo.",
    spec_draft_n_max: "Máximo de tokens propostos em cada etapa especulativa.",
    spec_draft_n_min: "Mínimo de tokens que o draft tenta propor.",
    spec_draft_p_min: "Probabilidade mínima para aceitar candidatos do draft.",
    spec_draft_p_split: "Limiar para dividir candidatos especulativos.",
    spec_ngram_mod_n_min: "Comprimento mínimo das sequências usadas pelo modo n-gram mod.",
    spec_ngram_mod_n_max: "Comprimento máximo das sequências usadas pelo modo n-gram.",
    spec_ngram_mod_n_match: "Quantidade de tokens exigida para considerar um n-gram correspondente.",
    spec_ngram_min_hits: "Número mínimo de ocorrências antes de usar uma previsão n-gram.",
    rope_scaling_type: "Método de extensão posicional RoPE: original, linear ou YaRN.",
    rope_scale: "Fator geral de escala do contexto RoPE.",
    rope_freq_base: "Frequência base RoPE; sobrescreve a metadata do modelo.",
    rope_freq_scale: "Escala inversa de frequência RoPE para extensão de contexto.",
    yarn_orig_ctx: "Contexto original de treinamento usado como referência pelo YaRN.",
    yarn_ext_factor: "Fator de extrapolação YaRN. -1 usa o padrão do modelo.",
    yarn_attn_factor: "Correção de magnitude da atenção no YaRN.",
    yarn_beta_slow: "Parâmetro beta lento da interpolação YaRN.",
    yarn_beta_fast: "Parâmetro beta rápido da interpolação YaRN.",
    omni: "Ativa o projetor multimodal. Desligado libera VRAM para uma janela textual maior.",
    mmproj_offload: "Executa o projetor multimodal na GPU. Na CPU é mais lento, mas usa menos VRAM.",
    no_mmproj_auto: "Desabilita descoberta automática de projetor pelo llama.cpp.",
    audio: "Ativa o vocoder/modelo de áudio quando um companion compatível existe.",
    image_min_tokens: "Número mínimo de tokens visuais reservado por imagem.",
    image_max_tokens: "Limite máximo de tokens visuais por imagem; zero usa o padrão.",
    mtmd_batch_max: "Máximo de tokens multimodais processados por batch.",
    host: "Endereço de escuta. 127.0.0.1 limita acesso à máquina local.",
    port: "Porta TCP da API e interface do llama-server.",
    agentic: "Ativa o agente e ferramentas integradas. Use somente em loopback/confiável.",
    tools: "Conjunto de ferramentas permitido: todas, somente leitura ou nenhuma.",
    fit: "Permite ao llama.cpp ajustar parâmetros não fixados para caber na VRAM.",
    fit_target: "VRAM que deve permanecer livre após o fit, em MiB.",
    fit_ctx: "Menor contexto que o fit pode escolher ao reduzir memória.",
    sleep_idle: "Coloca o modelo em estado de economia após inatividade. -1 desativa.",
    timeout: "Tempo máximo, em segundos, para operações/requisições do servidor.",
    threads_http: "Threads destinadas ao servidor HTTP. -1 usa cálculo automático.",
    sse_ping_interval: "Intervalo do heartbeat SSE em segundos.",
    reuse_port: "Solicita reutilização da porta pelo socket do servidor.",
    offline: "Impede acesso de rede do llama-server para buscar recursos externos.",
    cache_ram: "Limite de RAM para cache de prompts, em MiB. Zero desativa.",
    ctx_checkpoints: "Quantidade de checkpoints internos usados para restauração de contexto.",
    checkpoint_min_step: "Distância mínima em tokens entre checkpoints de contexto.",
    context_shift: "Desloca o contexto quando cheio para continuar gerando sem reiniciar.",
    warmup: "Executa uma passagem inicial para aquecer kernels e buffers antes de servir.",
    log_verbosity: "Nível de detalhe dos logs do llama.cpp.",
    log_file: "Arquivo opcional para persistir logs do servidor.",
    log_colors: "Controla sequências de cor ANSI nos logs.",
    log_prefix: "Inclui prefixos de componente/nível nas linhas de log.",
    log_timestamps: "Inclui timestamps nos logs.",
    metrics: "Expõe endpoint de métricas para monitoramento.",
    perf: "Inclui contadores e informações extras de desempenho.",
    check_tensors: "Valida tensors durante a carga; aumenta o tempo de inicialização.",
    spm_infill: "Usa formato de infill compatível com tokenizadores SentencePiece.",
    jinja: "Usa o chat template Jinja armazenado no GGUF.",
    slot_similarity: "Limiar para reutilizar slots com prompts semelhantes.",
    agentic_max_turns: "Máximo de ciclos agente↔ferramenta por solicitação.",
    agentic_max_tool_preview_lines: "Máximo de linhas exibidas na prévia do resultado de ferramenta.",
    ui_config_file: "Arquivo JSON de configuração avançada da UI do llama-server.",
    override_kv: "Sobrescreve uma chave de metadata GGUF em tempo de execução.",
    op_offload: "Move operações auxiliares do graph para o dispositivo acelerador.",
    slot_save_path: "Diretório usado para salvar/restaurar estados de slots KV.",
    api_key: "Chave exigida nas requisições à API. Não deve aparecer em logs ou previews.",
    api_key_file: "Arquivo contendo chaves válidas da API.",
    ssl_key_file: "Chave privada TLS usada pelo servidor HTTPS.",
    ssl_cert_file: "Certificado TLS usado pelo servidor HTTPS.",
    cors_origins: "Origens web autorizadas pelo CORS.",
    cors_methods: "Métodos HTTP permitidos pelo CORS.",
    cors_headers: "Cabeçalhos permitidos pelo CORS.",
    cors_credentials: "Permite credenciais/cookies em requisições CORS.",
    mcp_config_file: "Arquivo JSON com servidores MCP que o agente pode iniciar.",
    mcp_config_json: "Configuração MCP inline. Pode executar processos; use apenas conteúdo confiável.",
    mcp_native: "Acopla o MCP Crono Matrix diretamente ao llama-server por stdio. O ciclo de vida acompanha o servidor.",
    mcp_policy: "Segura expõe apenas ferramentas ALLOW. Completa habilita ferramentas ASK de escrita e shell e exige confiança/aprovação do cliente. Ferramentas SYSTEM continuam bloqueadas.",
    mcp_workspace: "Diretório de trabalho usado pelas ferramentas MCP para caminhos relativos e comandos.",
    mcp_snn_threads: "Threads OpenMP usadas pelo núcleo SNN Asael. Duas mantêm baixo uso de CPU sem bloquear a inferência principal.",
    mcp_snn_steps: "Passos neurais executados por estímulo. 64 é o equilíbrio recomendado; valores maiores aumentam CPU e latência.",
    mcp_repeat_limit: "Bloqueia chamadas MCP consecutivas com ferramenta e argumentos idênticos. Após sucesso, 2 impede a primeira repetição.",
    alias: "Nome alternativo exposto como identificador do modelo na API.",
    tags: "Tags informativas associadas ao modelo servido."
  };

  function bindModelFilter(root = document) {
    const input = root.querySelector("[data-model-filter]");
    if (!input || input.dataset.bound) return;
    input.dataset.bound = "1";
    input.addEventListener("input", () => {
      const query = input.value.trim().toLowerCase();
      document.querySelectorAll("[data-model-name]").forEach((row) => {
        row.hidden = query && !row.dataset.modelName.includes(query);
      });
    });
  }

  function bindParameterHelp(root = document) {
    const panel = root.matches?.("#parameters-panel") ? root : root.querySelector?.("#parameters-panel");
    if (!panel) return;
    panel.querySelectorAll("[name]").forEach((control) => {
      if (control.dataset.helpBound) return;
      const help = parameterHelp[control.name] || `Parâmetro ${control.name} enviado ao llama-server.`;
      const label = control.closest("label");
      control.dataset.helpBound = "1";
      control.title = help;
      control.setAttribute("aria-description", help);
      if (!label) return;
      label.title = help;
      label.classList.add("has-param-help");
      const marker = document.createElement("span");
      marker.className = "param-help";
      marker.textContent = "?";
      marker.title = help;
      marker.tabIndex = 0;
      marker.setAttribute("aria-label", help);
      label.append(marker);
    });
  }

  function bindCtkPreset(root = document) {
    const panel = root.matches?.("#parameters-panel") ? root : root.querySelector?.("#parameters-panel");
    if (!panel) return;
    panel.querySelectorAll("[data-ctk-preset]").forEach((select) => {
      if (select.dataset.bound) return;
      select.dataset.bound = "1";
      select.addEventListener("change", () => {
        const target = panel.querySelector(`[name="${select.dataset.ctkTarget}"]`);
        if (!target) return;
        target.value = select.value;
        target.dispatchEvent(new Event("input", { bubbles: true }));
      });
    });
  }

  function trimLogs() {
    const stream = document.querySelector("#log-stream");
    if (!stream) return;
    while (stream.children.length > maxLogLines) stream.firstElementChild.remove();
    if (!stream.classList.contains("paused")) stream.scrollTop = stream.scrollHeight;
  }

  function notifyHfRadar(root = document) {
    const panel = root.matches?.("#hf-radar") ? root : root.querySelector?.("#hf-radar") || document.querySelector("#hf-radar");
    const content = panel?.querySelector?.("[data-hf-radar-unread]");
    if (!content) return;
    const unread = Number(content.dataset.hfRadarUnread || "0");
    if (!Number.isFinite(unread)) return;
    document.title = unread > 0 ? `(${unread}) lançamentos · ${baseDocumentTitle}` : baseDocumentTitle;
    if (unread > 0 && unread !== hfRadarUnread) {
      const region = document.querySelector("#notice-region");
      if (region) {
        const notice = document.createElement("div");
        notice.className = "notice success";
        notice.textContent = `RADAR HF: ${unread} lançamento${unread === 1 ? "" : "s"} ou atualização(ões) aguardando revisão.`;
        region.append(notice);
        window.setTimeout(() => notice.remove(), 7500);
      }
    }
    hfRadarUnread = unread;
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindModelFilter();
    bindParameterHelp();
    bindCtkPreset();
    notifyHfRadar();
    document.querySelector("[data-log-pause]")?.addEventListener("click", (event) => {
      const stream = document.querySelector("#log-stream");
      stream.classList.toggle("paused");
      event.currentTarget.textContent = stream.classList.contains("paused") ? "SEGUIR" : "PAUSAR";
      trimLogs();
    });
    document.querySelector("[data-log-clear]")?.addEventListener("click", () => {
      const stream = document.querySelector("#log-stream");
      stream.replaceChildren();
    });
  });

  document.addEventListener("htmx:afterSwap", (event) => {
    bindModelFilter(event.detail.target);
    bindParameterHelp(event.detail.target);
    bindCtkPreset(event.detail.target);
    notifyHfRadar(event.detail.target);
    if (event.detail.target?.id === "log-stream") trimLogs();
    const notice = document.querySelector(".notice");
    if (notice) window.setTimeout(() => notice.remove(), 4500);
  });

  document.addEventListener("htmx:sseMessage", trimLogs);
})();
