(function () {
  'use strict';
  var data = null;
  var models = [];
  var state = { a: null, b: null, axis: '', evalSearch: '', mode: localStorage.getItem('eval-dashboard-mode') || 'simple' };

  function $(id) { return document.getElementById(id); }
  function node(tag, cls, text) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }
  function escapeHtml(value) {
    return String(value == null ? 'N/D' : value).replace(/[&<>"']/g, function (char) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char];
    });
  }
  function pct(v) { return v == null ? 'N/D' : Math.round(v * 100) + '%'; }
  function score(v) { return v == null ? 'N/D' : Number(v).toFixed(2); }
  function speed(v) { return v == null ? 'N/D' : Number(v).toFixed(1) + ' tok/s'; }
  function seconds(ms) { return ms == null ? 'N/D' : ms >= 60000 ? (ms / 60000).toFixed(1) + ' min' : (ms / 1000).toFixed(1) + ' s'; }
  function dateLabel(v) { if (!v) return 'sem data'; var d = new Date(v); return isNaN(d) ? 'sem data' : d.toLocaleString('pt-BR'); }
  function latestByModel(checkpoints) {
    var map = new Map();
    checkpoints.forEach(function (cp) {
      var cur = map.get(cp.model);
      if (!cur || String(cp.timestamp || '') > String(cur.timestamp || '')) map.set(cp.model, cp);
    });
    return Array.from(map.values());
  }
  function getCheckpoint(id) { return models.find(function (c) { return c.id === id; }); }
  function axisRows(cp) { return new Map((cp.summary.axes || []).map(function (a) { return [a.axisNumber, a]; })); }
  function capabilityName(n) { var ax = (data.catalog.axes || []).find(function (a) { return a.number === n; }); return ax ? ax.label : 'Eixo ' + n; }

  function metric(label, val, sub, hero) {
    var card = node('article', 'metric' + (hero ? ' hero-model' : ''));
    card.appendChild(node('span', 'metric-label', label));
    card.appendChild(node('strong', 'metric-value', val));
    card.appendChild(node('span', 'metric-sub', sub));
    return card;
  }

  function renderHero() {
    var cp = getCheckpoint(state.a); if (!cp) return;
    var hero = $('eval-hero'); hero.replaceChildren();
    hero.appendChild(metric('Modelo em foco', cp.model, cp.suiteGeneration + ' · ' + dateLabel(cp.timestamp), true));
    hero.appendChild(metric('Score médio', score(cp.summary.meanScore), 'de 10 pontos', false));
    hero.appendChild(metric('Taxa de acerto', pct(cp.summary.passRate), cp.summary.passed + ' de ' + cp.summary.evaluated + ' avaliados', false));
    hero.appendChild(metric('Cobertura', pct(cp.summary.coverage), cp.summary.available + ' de ' + cp.expectedTests + ' itens da geração', false));
  }

  function renderVerdict() {
    var cp = getCheckpoint(state.a); if (!cp) return;
    var axes = (cp.summary.axes || []).filter(function (a) { return a.meanScore != null; }).sort(function (a, b) { return b.meanScore - a.meanScore; });
    var strongest = axes.slice(0, 3).map(function (a) { return a.label; }).join(', ') || 'sem dados';
    var weakest = axes.slice(-3).reverse().map(function (a) { return a.label; }).join(', ') || 'sem dados';
    var box = $('eval-verdict'); box.replaceChildren();
    var title = node('h2', '', cp.summary.passRate >= .9 ? 'Ótimo candidato para uso principal' : cp.summary.passRate >= .75 ? 'Bom, mas precisa de supervisão' : 'Especialista com lacunas importantes');
    var simple = node('p', 'simple-only'); simple.append('Ele se destaca em '); simple.appendChild(node('strong', '', strongest)); simple.append(' e merece cuidado em '); simple.appendChild(node('strong', '', weakest)); simple.append('.');
    var academic = node('p', 'academic-only', 'Interpretação descritiva: score micro ' + score(cp.summary.meanScore) + ', cobertura ' + pct(cp.summary.coverage) + ', pass rate ' + pct(cp.summary.passRate) + '. Não há repetições independentes suficientes nos checkpoints legados para inferência causal.');
    box.append(title, simple, academic);
  }

  function runtimeEntries(runtime) {
    var p = runtime && runtime.parameters || {};
    var m = runtime && runtime.model || {};
    var h = runtime && runtime.hardware || {};
    var b = runtime && runtime.benchmark || {};
    var server = runtime && runtime.server_properties || {};
    var reported = server.reported || {};
    var vision = runtime && runtime.capabilities && runtime.capabilities.vision || {};
    var skippedAxes = b.skipped_axes || [];
    return [
      ['Modelo', m.name], ['Arquitetura', m.arch], ['Quantização', m.quant],
      ['Camadas', m.layers], ['Contexto', p.ctx], ['GPU layers', p.ngl],
      ['KV cache K', p.cache_k], ['KV cache V', p.cache_v],
      ['KV offload', p.kv_offload], ['KV unificado', p.kv_unified],
      ['Flash attention', p.flash], ['Paralelismo', p.parallel],
      ['Batch / ubatch', (p.batch != null ? p.batch : 'N/D') + ' / ' + (p.ubatch != null ? p.ubatch : 'N/D')],
      ['Threads / batch', (p.threads != null ? p.threads : 'N/D') + ' / ' + (p.threads_batch != null ? p.threads_batch : 'N/D')],
      ['Device', p.device], ['Split mode', p.split_mode],
      ['Reasoning', p.reasoning], ['Especulação', p.spec_type],
      ['Temperatura', p.temp], ['Top K / Top P / Min P', [p.top_k, p.top_p, p.min_p].join(' / ')],
      ['GPU', h.gpu_model || 'N/D'], ['VRAM total / livre', h.gpu_vram_gb != null ? h.gpu_vram_gb + ' / ' + h.gpu_vram_free_gb + ' GB' : 'N/D'],
      ['CPU', h.cpu_model || h.cpu || 'N/D'], ['Eixos', b.axes],
      ['Repetições / seed', (b.repeats != null ? b.repeats : 'N/D') + ' / ' + (b.seed != null ? b.seed : 'N/D')],
      ['Escala / reasoning', (b.scale || 'N/D') + ' / ' + (b.reasoning_mode || 'N/D')],
      ['Esforço / budget', (b.reasoning_effort || 'N/D') + ' / ' + (b.reasoning_budget != null ? b.reasoning_budget : 'N/D')],
      ['Sampling da avaliação', b.sampling || 'N/D'],
      ['Sampling fixo', 'T=' + (b.temperature != null ? b.temperature : 'N/D') + ' · K=' + (b.top_k != null ? b.top_k : 'N/D') + ' · P=' + (b.top_p != null ? b.top_p : 'N/D') + ' · minP=' + (b.min_p != null ? b.min_p : 'N/D') + ' · RP=' + (b.repeat_penalty != null ? b.repeat_penalty : 'N/D')],
      ['Saída / timeout / xctx', (b.max_tokens != null ? b.max_tokens : 'N/D') + ' / ' + (b.timeout != null ? b.timeout + ' s' : 'N/D') + ' / ' + (b.xctx_scale != null ? b.xctx_scale : 'N/D')],
      ['SO / juiz', (b.os_filter || 'N/D') + ' / ' + (b.judge_model || b.judge_url || 'mesma API')],
      ['API', b.api_url], ['Servidor /props', server.error ? 'indisponível: ' + server.error : 'capturado'],
      ['Modelo reportado', reported.model_alias || reported.model_path || reported.model || 'N/D'],
      ['Visão multimodal', vision.enabled ? 'ativa' : 'inativa'],
      ['MMProj', vision.mmproj || 'N/D'],
      ['Eixos executados', b.effective_axes || b.axes],
      ['Eixos pulados', skippedAxes.length ? skippedAxes.map(function (item) { return item.axis + ': ' + item.reason; }).join('; ') : 'nenhum'],
      ['Capturado em', runtime && runtime.captured_at],
    ];
  }

  function runtimeMarkup(runtime) {
    if (!runtime || !runtime.parameters) return '<p class="muted">Contexto de execução não registrado neste checkpoint.</p>';
    return '<div class="runtime-grid">' + runtimeEntries(runtime).map(function (entry) {
      return '<div><span>' + escapeHtml(entry[0]) + '</span><strong>' + escapeHtml(entry[1]) + '</strong></div>';
    }).join('') + '</div>';
  }

  function renderRuntime() {
    var cp = getCheckpoint(state.a);
    var box = $('eval-runtime');
    if (box) box.innerHTML = runtimeMarkup(cp && cp.runtimeContext);
  }

  function makeTable(headers, rows) {
    var tbl = node('table'); var thead = node('thead'); var tr = node('tr');
    headers.forEach(function (h) { tr.appendChild(node('th', '', h)); }); thead.appendChild(tr); tbl.appendChild(thead);
    var tbody = node('tbody'); rows.forEach(function (cells) { var row = node('tr'); cells.forEach(function (cell) { var td = node('td'); if (cell instanceof Node) td.appendChild(cell); else td.textContent = String(cell); row.appendChild(td); }); tbody.appendChild(row); }); tbl.appendChild(tbody); return tbl;
  }

  function renderLeaderboard() {
    var rows = models.map(function (cp, i) {
      var nameBtn = node('button', 'model-link', cp.model); nameBtn.type = 'button';
      nameBtn.addEventListener('click', function () { state.a = cp.id; $('model-a').value = state.a; render(); });
      return [node('span', 'rank', i + 1), nameBtn, score(cp.summary.meanScore), pct(cp.summary.passRate), speed(cp.summary.completionTokensPerSecond), pct(cp.summary.coverage), cp.summary.passed + ' / ' + cp.summary.failed, seconds(cp.summary.medianLatencyMs), cp.suiteGeneration];
    });
    var lb = $('eval-leaderboard'); if (lb) lb.replaceChildren(makeTable(['#', 'Modelo / configuração', 'Score', 'Acerto', 'Velocidade', 'Cobertura', '✓ / ✗', 'Mediana', 'Geração'], rows));
  }

  function renderOutcomes() {
    var c = $('eval-outcome-chart'); if (!c) return; c.replaceChildren();
    models.forEach(function (cp) {
      var row = node('div', 'bar-row'); row.appendChild(node('strong', '', cp.model));
      var stack = node('div', 'stack');
      var expected = Math.max(cp.expectedTests, cp.summary.available, 1);
      [['p', cp.summary.passed, 'aprovados'], ['f', cp.summary.failed, 'falhas'], ['s', cp.summary.skipped, 'erros/skips'], ['m', Math.max(0, expected - cp.summary.available), 'não presentes']].forEach(function (part) {
        var seg = node('span', part[0]); seg.style.width = (part[1] / expected * 100) + '%'; seg.title = part[1] + ' ' + part[2]; stack.appendChild(seg);
      });
      row.appendChild(stack); row.appendChild(node('span', 'muted', pct(cp.summary.passRate))); c.appendChild(row);
    });
  }

  function heatColor(val) {
    if (val == null) return '';
    var hue = Math.max(0, Math.min(115, val / 10 * 115));
    return 'hsl(' + hue + ' 58% 32%)';
  }

  function renderHeatmap() {
    var wrap = $('eval-heatmap'); if (!wrap) return; wrap.replaceChildren();
    var grid = node('div', 'heatmap');
    grid.appendChild(node('div', 'heat-cell', 'Modelo'));
    (data.catalog.axes || []).forEach(function (ax) {
      var cell = node('div', 'heat-cell axis', ax.number + ' ' + ax.label);
      cell.title = ax.label; grid.appendChild(cell);
    });
    models.forEach(function (cp) {
      grid.appendChild(node('div', 'heat-cell model', cp.model));
      var map = axisRows(cp);
      (data.catalog.axes || []).forEach(function (ax) {
        var result = map.get(ax.number);
        var cell = node('div', 'heat-cell' + (!result || result.meanScore == null ? ' na' : ''), result && result.meanScore != null ? result.meanScore.toFixed(1) : '—');
        if (result && result.meanScore != null) { cell.style.background = heatColor(result.meanScore); cell.title = ax.label + ': ' + result.meanScore.toFixed(2) + '/10 · ' + result.passed + '/' + result.evaluated; }
        grid.appendChild(cell);
      });
    });
    wrap.appendChild(grid);
  }

  function commonTests(a, b) {
    if (!a || !b || a.suiteGeneration !== b.suiteGeneration) return [];
    var right = new Map(b.tests.map(function (t) { return [t.id + '|' + t.name, t]; }));
    return a.tests.map(function (left) { var other = right.get(left.id + '|' + left.name); return other ? [left, other] : null; }).filter(Boolean).filter(function (pair) { return pair[0].score != null && pair[1].score != null; });
  }

  function renderPairwise() {
    var a = getCheckpoint(state.a), b = getCheckpoint(state.b), pairs = commonTests(a, b), box = $('eval-pairwise');
    if (!box) return; box.replaceChildren();
    if (!pairs.length) { box.appendChild(node('p', 'muted', 'Sem conjunto compatível. Escolha modelos da mesma geração da suíte.')); return; }
    var wins = pairs.filter(function (p) { return p[0].score > p[1].score; }).length;
    var losses = pairs.filter(function (p) { return p[0].score < p[1].score; }).length;
    var ties = pairs.length - wins - losses;
    var duel = node('div', 'duel'); duel.append(node('strong', 'win', wins), node('strong', 'tie', ties), node('strong', 'loss', losses)); box.appendChild(duel);
    box.appendChild(node('p', 'muted', 'vitórias de A · empates · vitórias de B, em ' + pairs.length + ' testes comuns'));
    var scoreDelta = (a.summary.meanScore || 0) - (b.summary.meanScore || 0);
    var passDelta = ((a.summary.passRate || 0) - (b.summary.passRate || 0)) * 100;
    var latencyDelta = (a.summary.medianLatencyMs || 0) - (b.summary.medianLatencyMs || 0);
    var speedDelta = (a.summary.completionTokensPerSecond || 0) - (b.summary.completionTokensPerSecond || 0);
    var sameModel = a.baseModel && b.baseModel && a.baseModel === b.baseModel;
    var qualityWinner = scoreDelta > 0 ? 'A' : scoreDelta < 0 ? 'B' : 'empate';
    var speedWinner = speedDelta > 0 ? 'A' : speedDelta < 0 ? 'B' : 'empate';
    var latencyWinner = latencyDelta < 0 ? 'A' : latencyDelta > 0 ? 'B' : 'empate';
    var summary = node('div', 'comparison-summary');
    summary.appendChild(node('strong', '', sameModel ? 'Comparação de configurações do mesmo modelo' : 'Comparação entre modelos/configurações'));
    summary.appendChild(node('p', '', 'Qualidade: ' + qualityWinner + ' · Δ score ' + (scoreDelta >= 0 ? '+' : '') + scoreDelta.toFixed(2)
      + ' · Δ acerto ' + (passDelta >= 0 ? '+' : '') + passDelta.toFixed(1) + ' pp'));
    summary.appendChild(node('p', '', 'Desempenho: ' + speedWinner + ' · Δ velocidade ' + (speedDelta >= 0 ? '+' : '') + speedDelta.toFixed(1)
      + ' tok/s · Δ latência ' + (latencyDelta >= 0 ? '+' : '') + seconds(Math.abs(latencyDelta))));
    summary.appendChild(node('p', 'comparison-conclusion', 'Resultado: ' + qualityWinner + ' venceu em qualidade; '
      + speedWinner + ' venceu em velocidade; ' + latencyWinner + ' teve menor latência.'));
    box.appendChild(summary);
    var axisNums = Array.from(new Set(pairs.map(function (p) { return p[0].axisNumber; }))).sort(function (x, y) { return x - y; });
    axisNums.forEach(function (n) {
      var rel = pairs.filter(function (p) { return p[0].axisNumber === n; });
      var delta = rel.reduce(function (s, p) { return s + p[0].score - p[1].score; }, 0) / rel.length;
      var row = node('div', 'axis-delta');
      row.appendChild(node('span', '', capabilityName(n)));
      var track = node('div', 'delta-track'); var fill = node('span', 'delta-fill');
      var w = Math.min(50, Math.abs(delta) / 10 * 50); fill.style.width = w + '%';
      fill.style.left = delta >= 0 ? '50%' : (50 - w) + '%';
      fill.style.background = delta >= 0 ? 'var(--phosphor)' : 'var(--danger)';
      track.appendChild(fill);
      row.append(track, node('strong', delta >= 0 ? 'pass' : 'fail', (delta > 0 ? '+' : '') + delta.toFixed(1)));
      box.appendChild(row);
    });
  }

  function renderLatency() {
    var box = $('eval-latency'); if (!box) return; box.replaceChildren();
    [getCheckpoint(state.a), getCheckpoint(state.b)].filter(Boolean).forEach(function (cp) {
      var card = node('div', 'latency-card'); var left = node('div');
      left.append(node('strong', '', cp.model), node('p', 'muted', 'Q1 ' + seconds(cp.summary.q1LatencyMs) + ' · Q3 ' + seconds(cp.summary.q3LatencyMs) + ' · P90 ' + seconds(cp.summary.p90LatencyMs)
        + ' · prompt ' + (cp.summary.promptTokens || 0) + ' tok · resposta ' + (cp.summary.completionTokens || 0) + ' tok'));
      var right = node('div', 'latency-numbers');
      right.append(node('strong', '', seconds(cp.summary.medianLatencyMs)), node('span', 'pass', speed(cp.summary.completionTokensPerSecond)));
      card.append(left, right); box.appendChild(card);
    });
    box.appendChild(node('p', 'muted', 'A mediana inclui todo o teste, não apenas inferência. Outliers podem conter retries, ferramentas e LLM judge.'));
  }

  function renderProvenance() {
    var box = $('eval-provenance'); if (!box) return; box.replaceChildren();
    box.appendChild(makeTable(['Indicador', 'Valor', 'Interpretação'], [
      ['Checkpoints importados', data.checkpoints.length, 'Snapshots disponíveis, incluindo gerações antigas'],
      ['Modelos únicos exibidos', models.length, 'Apenas o checkpoint mais recente de cada modelo'],
      ['Duplicatas removidas', (data.duplicates || []).length, 'Arquivos idênticos não contam duas vezes'],
      ['Avisos de validação', (data.validation || []).length, 'Inconsistências de contagem ou parsing'],
      ['Catálogo atual', data.catalog.tests.length + ' testes', 'Eixos 1–23 da suíte v4'],
    ]));
  }

  function detailCell(test) {
    var d = node('details'); d.appendChild(node('summary', '', 'ver evidência'));
    d.appendChild(node('p', '', test.details || test.error || 'Sem detalhes no checkpoint legado.'));
    if (test.response) d.appendChild(node('pre', '', String(test.response).slice(0, 1500)));
    return d;
  }

  function renderExplorer() {
    var cp = getCheckpoint(state.a); if (!cp) return;
    var q = state.evalSearch.toLowerCase();
    var tests = cp.tests.filter(function (t) { return (!state.axis || String(t.axisNumber) === state.axis) && (!q || (t.id + ' ' + t.name + ' ' + (t.details || '')).toLowerCase().includes(q)); });
    var rc = $('eval-result-count'); if (rc) rc.textContent = tests.length + ' testes';
    var rows = tests.map(function (t) {
      var st = node('span', t.status === 'pass' ? 'pass' : t.status === 'fail' ? 'fail' : 'skip-state', t.status === 'pass' ? '✓ passou' : t.status === 'fail' ? '✗ falhou' : '⚠ sem resultado');
      return [t.id, capabilityName(t.axisNumber), t.name, t.difficulty, st, t.score == null ? '—' : t.score.toFixed(1), seconds(t.latencyMs), detailCell(t)];
    });
    var te = $('eval-test-explorer'); if (te) te.replaceChildren(makeTable(['ID', 'Eixo', 'Teste', 'Nível', 'Status', 'Score', 'Tempo', 'Evidência'], rows));
  }

  function render() { renderHero(); renderVerdict(); renderRuntime(); renderLeaderboard(); renderOutcomes(); renderHeatmap(); renderPairwise(); renderLatency(); renderProvenance(); renderExplorer(); }

  function setMode(mode) {
    state.mode = mode;
    document.body.classList.toggle('eval-academic', mode === 'academic');
    document.querySelectorAll('[data-eval-mode]').forEach(function (btn) { btn.classList.toggle('active', btn.dataset.evalMode === mode); });
    localStorage.setItem('eval-dashboard-mode', mode);
  }

  function showEmptyState(msg) {
    var wrap = $('eval-dashboard-content');
    if (!wrap) return;
    wrap.innerHTML = '<div class="eval-panel empty-state"><p style="text-align:center;padding:3rem 1rem;color:var(--muted);font-family:var(--mono);font-size:.9rem;">'
      + msg + '</p></div>';
  }

  function showDashboardShell() {
    var wrap = $('eval-dashboard-content');
    if (!wrap) return;
    wrap.innerHTML = '<div id="eval-db-root">'
      + '<section id="eval-hero" class="hero-grid" aria-live="polite"></section>'
      + '<section class="eval-panel verdict" id="eval-verdict"></section>'
      + '<section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Reprodutibilidade</p><h2>Configuração da inferência</h2></div></div><div id="eval-runtime"></div></section>'
      + '<section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Panorama</p><h2>Modelos e configurações</h2></div><p class="academic-only">Cada configuração de inferência possui identidade própria; execuções repetidas da mesma variante usam a mais recente.</p></div><div id="eval-leaderboard" class="table-wrap"></div></section>'
      + '<section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Resultado</p><h2>Acertos, falhas e lacunas</h2></div></div><div id="eval-outcome-chart" class="chart-list"></div></section>'
      + '<section class="eval-panel" style="border:none;padding:0;"><div class="section-head" style="padding:0 1.6rem;padding-top:1.6rem;"><div><p class="eyebrow">Especialidades</p><h2>Mapa de competências</h2></div><p class="simple-only">Quanto mais cheio e verde, melhor naquela área.</p><p class="academic-only">Média por eixo; células hachuradas indicam ausência de dados, não reprovação.</p></div><div id="eval-heatmap" class="heatmap-wrap" style="padding:0 1.6rem 1.6rem;"></div></section>'
      + '<div class="split"><section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Duelo direto</p><h2>Onde cada modelo vence</h2></div></div><div id="eval-pairwise"></div></section><section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Tempo</p><h2>Latência ponta a ponta</h2></div></div><div id="eval-latency"></div></section></div>'
      + '<section class="eval-panel academic-only"><div class="section-head"><div><p class="eyebrow">Proveniência</p><h2>Qualidade dos dados</h2></div></div><div id="eval-provenance"></div></section>'
      + '<section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Evidências</p><h2>Explorador de testes</h2></div><span id="eval-result-count" aria-live="polite"></span></div><div id="eval-test-explorer" class="table-wrap"></div></section>'
      + '</div>';
  }

  function initFromData() {
    models = latestByModel(data.checkpoints || []).sort(function (a, b) { return (b.summary.meanScore || 0) - (a.summary.meanScore || 0); });
    var ctrlWrap = $('eval-controls-wrapper');
    if (ctrlWrap) ctrlWrap.style.display = models.length ? '' : 'none';

    if (!models.length) {
      showEmptyState('Nenhum modelo avaliado ainda.<br><br>Use o painel acima para executar a suite de avaliação em um servidor llama.cpp ativo.');
      return;
    }

    showDashboardShell();

    var ma = $('model-a'), mb = $('model-b'), af = $('axis-filter');
    if (ma) { ma.replaceChildren(); models.forEach(function (cp) { var opt = node('option', '', cp.model + ' · ' + cp.summary.available + ' testes'); opt.value = cp.id; ma.appendChild(opt); }); }
    if (mb) { mb.replaceChildren(); models.forEach(function (cp) { var opt = node('option', '', cp.model + ' · ' + cp.summary.available + ' testes'); opt.value = cp.id; mb.appendChild(opt); }); }
    if (af) { af.replaceChildren(); var allOpt = node('option', '', 'Todas'); allOpt.value = ''; af.appendChild(allOpt); (data.catalog.axes || []).forEach(function (ax) { var opt = node('option', '', ax.number + ' · ' + ax.label); opt.value = String(ax.number); af.appendChild(opt); }); }

    state.a = models[0] && models[0].id;
    state.b = models[1] && models[1].id;
    if (ma) ma.value = state.a || '';
    if (mb) mb.value = state.b || state.a || '';

    render();
  }

  function loadData() {
    fetch('/eval/data')
      .then(function (r) { return r.json(); })
      .then(function (json) {
        if (!json || !Array.isArray(json.checkpoints)) {
          showEmptyState('Nenhum dado de avaliação encontrado.<br><br>Execute a suite ou gere os dados com:<br><code style="color:var(--phosphor);">node eval_dashboard/tools/generate-dashboard-data.mjs</code>');
          var ctrlWrap = $('eval-controls-wrapper');
          if (ctrlWrap) ctrlWrap.style.display = 'none';
          return;
        }
        data = json;
        initFromData();
      })
      .catch(function () {
        showEmptyState('Erro ao carregar dados de avaliação.');
        var ctrlWrap = $('eval-controls-wrapper');
        if (ctrlWrap) ctrlWrap.style.display = 'none';
      });
  }

  function bindEvents() {
    document.querySelectorAll('[data-eval-mode]').forEach(function (btn) { btn.addEventListener('click', function () { setMode(btn.dataset.evalMode); }); });
    var ma = $('model-a'); if (ma) ma.addEventListener('change', function (e) { state.a = e.target.value; render(); });
    var mb = $('model-b'); if (mb) mb.addEventListener('change', function (e) { state.b = e.target.value; renderPairwise(); renderLatency(); });
    var af = $('axis-filter'); if (af) af.addEventListener('change', function (e) { state.axis = e.target.value; renderExplorer(); });
    var sr = $('eval-search'); if (sr) sr.addEventListener('input', function (e) { state.evalSearch = e.target.value; renderExplorer(); });
  }

  setMode(state.mode);
  bindEvents();
  loadData();

  var logEventSource = null;
  var finalDashboardLoaded = false;

  function liveMetric(label, value, detail) {
    return '<article class="metric"><span class="metric-label">' + label + '</span>'
      + '<strong class="metric-value">' + value + '</strong>'
      + '<span class="metric-sub">' + detail + '</span></article>';
  }

  function renderLiveProgress(snapshot) {
    if (!snapshot || !snapshot.progress) return;
    var progress = snapshot.progress;
    var axes = Object.keys(progress.axes || {}).sort(function (a, b) { return Number(a) - Number(b); });
    var target = document.getElementById('eval-dashboard-content');
    if (!target || snapshot.state === 'idle') return;
    var controls = document.getElementById('eval-controls-wrapper');
    if (controls) {
      controls.style.display = (snapshot.state === 'running' || snapshot.state === 'stopping')
        ? 'none' : models.length ? '' : 'none';
    }

    if (snapshot.state === 'done') {
      if (finalDashboardLoaded) return;
      finalDashboardLoaded = true;
      fetch('/partials/eval-dashboard')
        .then(function (response) { return response.text(); })
        .then(function (markup) {
          target.innerHTML = markup;
          loadData();
        });
      return;
    }

    finalDashboardLoaded = false;
    var completed = Number(progress.current || 0);
    var total = Number(progress.total || 0);
    var overall = total ? Math.min(100, completed / total * 100) : 0;
    var evaluated = Number(progress.passed || 0) + Number(progress.failed || 0);
    var passRate = evaluated ? Math.round(Number(progress.passed || 0) / evaluated * 100) : 0;
    var rows = axes.map(function (axis) {
      var item = progress.axes[axis];
      var count = Number(item.completed || 0) || 1;
      var passWidth = Number(item.passed || 0) / count * 100;
      var failWidth = Number(item.failed || 0) / count * 100;
      return '<div class="live-dashboard-axis">'
        + '<div><strong>Eixo ' + axis + '</strong><small>último: ' + (item.latest_test || '—') + '</small></div>'
        + '<div class="live-dashboard-track"><span class="live-dashboard-pass" style="width:' + passWidth + '%"></span>'
        + '<span class="live-dashboard-fail" style="width:' + failWidth + '%"></span></div>'
        + '<strong>' + Number(item.mean_score || 0).toFixed(1) + '/10</strong>'
        + '<span>' + item.passed + '✓ ' + item.failed + '✗ · ' + item.completed + ' testes</span>'
        + '</div>';
    }).join('');

    target.innerHTML = '<div class="live-dashboard">'
      + '<div class="section-head"><div><p class="eyebrow">Execução em andamento</p><h2>Métricas ao vivo por eixo</h2></div>'
      + '<span class="live-state">' + snapshot.state.toUpperCase() + '</span></div>'
      + '<section class="hero-grid live-hero">'
      + liveMetric('Eixo atual', progress.axis || '—', 'teste ' + (progress.test || 'aguardando'))
      + liveMetric('Concluídos', completed + ' / ' + total, overall.toFixed(1) + '% da suíte')
      + liveMetric('Aprovados', progress.passed || 0, passRate + '% dos avaliados')
      + liveMetric('Falhas / erros', (progress.failed || 0) + ' / ' + (progress.skipped || 0), 'falhas de qualidade / sem resultado')
      + '</section>'
      + '<section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Reprodutibilidade</p><h2>Configuração usada neste teste</h2></div></div>'
      + runtimeMarkup(snapshot.runtime) + '</section>'
      + '<section class="eval-panel"><div class="live-overall-label"><span>Progresso total</span><strong>' + overall.toFixed(1) + '%</strong></div>'
      + '<div class="live-overall-track"><span style="width:' + overall + '%"></span></div></section>'
      + '<section class="eval-panel"><div class="section-head"><div><p class="eyebrow">Comparativo parcial</p><h2>Desempenho dos eixos concluídos</h2></div></div>'
      + (rows || '<p class="empty-state">Aguardando o primeiro teste concluído.</p>') + '</section></div>';
  }

  function connectLogStream() {
    if (logEventSource) { logEventSource.close(); }
    logEventSource = new EventSource('/eval/events');
    logEventSource.addEventListener('log', function (evt) {
      var el = document.getElementById('eval-log-stream');
      if (el) {
        el.insertAdjacentHTML('beforeend', evt.data);
        el.scrollTop = el.scrollHeight;
      }
    });
    logEventSource.addEventListener('status', function (evt) {
      var el = document.getElementById('eval-status');
      if (el) {
        el.innerHTML = evt.data;
        var status = el.querySelector('[data-eval-state]');
        var startButton = document.querySelector('#eval-runner-form button[type="submit"]');
        if (startButton && status) {
          startButton.disabled = status.dataset.evalState === 'running' || status.dataset.evalState === 'stopping';
        }
      }
    });
    logEventSource.addEventListener('progress', function (evt) {
      try { renderLiveProgress(JSON.parse(evt.data)); } catch (_) {}
    });
    logEventSource.onerror = function () {};
  }

  connectLogStream();

  document.addEventListener('click', function (evt) {
    var stopButton = evt.target.closest('[data-eval-stop]');
    if (stopButton && !stopButton.disabled) {
      evt.preventDefault();
      stopButton.disabled = true;
      stopButton.textContent = 'PARANDO...';
      fetch('/eval/stop', { method: 'POST' })
        .then(function (response) { return response.text(); })
        .then(function (markup) {
          var target = document.getElementById('eval-runner-status');
          if (target) {
            target.innerHTML = markup;
            if (window.htmx) window.htmx.process(target);
          }
        })
        .catch(function () { stopButton.disabled = false; });
      return;
    }

    var dashboardButton = evt.target.closest('[data-eval-dashboard]');
    if (dashboardButton) {
      evt.preventDefault();
      fetch('/partials/eval-dashboard')
        .then(function (response) { return response.text(); })
        .then(function (markup) {
          var target = document.getElementById('eval-dashboard-content');
          if (target) {
            target.innerHTML = markup;
            loadData();
          }
        });
      return;
    }

    var deleteButton = evt.target.closest('[data-delete-run]');
    if (deleteButton) {
      evt.preventDefault();
      var selected = document.getElementById('model-a');
      var checkpoint = selected && getCheckpoint(selected.value);
      if (!checkpoint) return;
      if (!window.confirm('Excluir esta execução?\n\n' + checkpoint.model + '\n\nO arquivo bruto também será removido.')) return;
      deleteButton.disabled = true;
      fetch('/eval/runs/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ checkpoint_id: checkpoint.id }).toString(),
      })
        .then(function (response) {
          if (!response.ok) return response.text().then(function (text) { throw new Error(text); });
          return response.json();
        })
        .then(function () { loadData(); })
        .catch(function (error) { window.alert(error.message); deleteButton.disabled = false; });
    }
  });

  document.body.addEventListener('htmx:afterSettle', function (evt) {
    if (evt.detail.target && evt.detail.target.id === 'eval-dashboard-content') {
      bindEvents();
      loadData();
    }
    var logEl = document.getElementById('eval-log-stream');
    if (logEl) { logEl.scrollTop = logEl.scrollHeight; }
  });
})();
