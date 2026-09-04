(function () {
  'use strict';
  var data = window.__EVAL_DASHBOARD_DATA__;
  if (!data || !Array.isArray(data.checkpoints)) {
    document.body.textContent = 'Dados não encontrados. Execute: node tools/generate-dashboard-data.mjs';
    return;
  }

  var $ = function (id) { return document.getElementById(id); };
  var state = { a: null, b: null, axis: '', search: '', mode: localStorage.getItem('alpha-dashboard-mode') || 'simple' };

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }
  function pct(value) { return value == null ? 'N/D' : Math.round(value * 100) + '%'; }
  function score(value) { return value == null ? 'N/D' : Number(value).toFixed(2); }
  function seconds(ms) { return ms == null ? 'N/D' : ms >= 60000 ? (ms / 60000).toFixed(1) + ' min' : (ms / 1000).toFixed(1) + ' s'; }
  function dateLabel(value) { if (!value) return 'sem data'; var date = new Date(value); return isNaN(date) ? 'sem data' : date.toLocaleString('pt-BR'); }
  function latestByModel(checkpoints) {
    var map = new Map();
    checkpoints.forEach(function (checkpoint) {
      var current = map.get(checkpoint.model);
      if (!current || String(checkpoint.timestamp || '') > String(current.timestamp || '')) map.set(checkpoint.model, checkpoint);
    });
    return Array.from(map.values());
  }
  var models = latestByModel(data.checkpoints).sort(function (a, b) { return (b.summary.meanScore || 0) - (a.summary.meanScore || 0); });

  function option(select, checkpoint) {
    var item = node('option', '', checkpoint.model + ' · ' + checkpoint.summary.available + ' testes');
    item.value = checkpoint.id;
    select.appendChild(item);
  }
  models.forEach(function (checkpoint) { option($('model-a'), checkpoint); option($('model-b'), checkpoint); });
  data.catalog.axes.forEach(function (axis) {
    var item = node('option', '', axis.number + ' · ' + axis.label); item.value = String(axis.number); $('axis-filter').appendChild(item);
  });
  state.a = models[0] && models[0].id;
  state.b = models[1] && models[1].id;
  $('model-a').value = state.a || '';
  $('model-b').value = state.b || state.a || '';

  function getCheckpoint(id) { return models.find(function (checkpoint) { return checkpoint.id === id; }); }
  function axisRows(checkpoint) { return new Map((checkpoint.summary.axes || []).map(function (axis) { return [axis.axisNumber, axis]; })); }
  function capabilityName(axisNumber) { var axis = data.catalog.axes.find(function (item) { return item.number === axisNumber; }); return axis ? axis.label : 'Eixo ' + axisNumber; }

  function metric(label, value, sub, hero) {
    var card = node('article', 'panel metric' + (hero ? ' hero-model' : ''));
    card.appendChild(node('span', 'metric-label', label));
    card.appendChild(node('strong', 'metric-value', value));
    card.appendChild(node('span', 'metric-sub', sub));
    return card;
  }

  function renderHero() {
    var checkpoint = getCheckpoint(state.a); if (!checkpoint) return;
    var hero = $('hero'); hero.replaceChildren();
    hero.appendChild(metric('Modelo em foco', checkpoint.model, checkpoint.suiteGeneration + ' · ' + dateLabel(checkpoint.timestamp), true));
    hero.appendChild(metric('Score médio', score(checkpoint.summary.meanScore), 'de 10 pontos', false));
    hero.appendChild(metric('Taxa de acerto', pct(checkpoint.summary.passRate), checkpoint.summary.passed + ' de ' + checkpoint.summary.evaluated + ' avaliados', false));
    hero.appendChild(metric('Cobertura', pct(checkpoint.summary.coverage), checkpoint.summary.available + ' de ' + checkpoint.expectedTests + ' itens da geração', false));
  }

  function renderVerdict() {
    var checkpoint = getCheckpoint(state.a); if (!checkpoint) return;
    var axes = (checkpoint.summary.axes || []).filter(function (axis) { return axis.meanScore != null; }).sort(function (a, b) { return b.meanScore - a.meanScore; });
    var strongest = axes.slice(0, 3).map(function (axis) { return axis.label; }).join(', ') || 'sem dados';
    var weakest = axes.slice(-3).reverse().map(function (axis) { return axis.label; }).join(', ') || 'sem dados';
    var box = $('verdict'); box.replaceChildren();
    var title = node('h2', '', checkpoint.summary.passRate >= .9 ? 'Ótimo candidato para uso principal' : checkpoint.summary.passRate >= .75 ? 'Bom, mas precisa de supervisão' : 'Especialista com lacunas importantes');
    var simple = node('p', 'simple-only');
    simple.append('Ele se destaca em '); simple.appendChild(node('strong', '', strongest)); simple.append(' e merece cuidado em '); simple.appendChild(node('strong', '', weakest)); simple.append('.');
    var academic = node('p', 'academic-only', 'Interpretação descritiva: score micro ' + score(checkpoint.summary.meanScore) + ', cobertura ' + pct(checkpoint.summary.coverage) + ', pass rate ' + pct(checkpoint.summary.passRate) + '. Não há repetições independentes suficientes nos checkpoints legados para inferência causal.');
    box.append(title, simple, academic);
  }

  function makeTable(headers, rows) {
    var table = node('table'); var thead = node('thead'); var tr = node('tr');
    headers.forEach(function (header) { tr.appendChild(node('th', '', header)); }); thead.appendChild(tr); table.appendChild(thead);
    var tbody = node('tbody'); rows.forEach(function (cells) { var row = node('tr'); cells.forEach(function (cell) { var td = node('td'); if (cell instanceof Node) td.appendChild(cell); else td.textContent = String(cell); row.appendChild(td); }); tbody.appendChild(row); }); table.appendChild(tbody); return table;
  }

  function renderLeaderboard() {
    var rows = models.map(function (checkpoint, index) {
      var name = node('button', 'model-link', checkpoint.model); name.type = 'button'; name.addEventListener('click', function () { state.a = checkpoint.id; $('model-a').value = state.a; render(); });
      return [node('span', 'rank', index + 1), name, score(checkpoint.summary.meanScore), pct(checkpoint.summary.passRate), pct(checkpoint.summary.coverage), checkpoint.summary.passed + ' / ' + checkpoint.summary.failed, seconds(checkpoint.summary.medianLatencyMs), checkpoint.suiteGeneration];
    });
    $('leaderboard').replaceChildren(makeTable(['#', 'Modelo', 'Score', 'Acerto', 'Cobertura', '✓ / ✗', 'Mediana', 'Geração'], rows));
  }

  function renderOutcomes() {
    var container = $('outcome-chart'); container.replaceChildren();
    models.forEach(function (checkpoint) {
      var row = node('div', 'bar-row'); row.appendChild(node('strong', '', checkpoint.model));
      var stack = node('div', 'stack');
      var expected = Math.max(checkpoint.expectedTests, checkpoint.summary.available, 1);
      [['p', checkpoint.summary.passed, 'aprovados'], ['f', checkpoint.summary.failed, 'falhas'], ['s', checkpoint.summary.skipped, 'erros/skips'], ['m', Math.max(0, expected - checkpoint.summary.available), 'não presentes']].forEach(function (part) {
        var segment = node('span', part[0]); segment.style.width = (part[1] / expected * 100) + '%'; segment.title = part[1] + ' ' + part[2]; stack.appendChild(segment);
      });
      row.appendChild(stack); row.appendChild(node('span', 'muted', pct(checkpoint.summary.passRate))); container.appendChild(row);
    });
  }

  function heatColor(value) {
    if (value == null) return '';
    var hue = Math.max(0, Math.min(115, value / 10 * 115));
    return 'hsl(' + hue + ' 58% 32%)';
  }
  function renderHeatmap() {
    var wrap = $('heatmap'); wrap.replaceChildren(); var grid = node('div', 'heatmap');
    grid.appendChild(node('div', 'heat-cell', 'Modelo'));
    data.catalog.axes.forEach(function (axis) { var cell = node('div', 'heat-cell axis', axis.number + ' ' + axis.label); cell.title = axis.label; grid.appendChild(cell); });
    models.forEach(function (checkpoint) {
      grid.appendChild(node('div', 'heat-cell model', checkpoint.model)); var map = axisRows(checkpoint);
      data.catalog.axes.forEach(function (axis) {
        var result = map.get(axis.number); var cell = node('div', 'heat-cell' + (!result || result.meanScore == null ? ' na' : ''), result && result.meanScore != null ? result.meanScore.toFixed(1) : '—');
        if (result && result.meanScore != null) { cell.style.background = heatColor(result.meanScore); cell.title = axis.label + ': ' + result.meanScore.toFixed(2) + '/10 · ' + result.passed + '/' + result.evaluated; }
        grid.appendChild(cell);
      });
    }); wrap.appendChild(grid);
  }

  function commonTests(a, b) {
    if (!a || !b || a.suiteGeneration !== b.suiteGeneration) return [];
    var right = new Map(b.tests.map(function (test) { return [test.id + '|' + test.name, test]; }));
    return a.tests.map(function (left) { var other = right.get(left.id + '|' + left.name); return other ? [left, other] : null; }).filter(Boolean).filter(function (pair) { return pair[0].score != null && pair[1].score != null; });
  }
  function renderPairwise() {
    var a = getCheckpoint(state.a), b = getCheckpoint(state.b), pairs = commonTests(a, b), box = $('pairwise'); box.replaceChildren();
    if (!pairs.length) { box.appendChild(node('p', 'muted', 'Sem conjunto compatível. Escolha modelos da mesma geração da suíte.')); return; }
    var wins = pairs.filter(function (pair) { return pair[0].score > pair[1].score; }).length;
    var losses = pairs.filter(function (pair) { return pair[0].score < pair[1].score; }).length;
    var ties = pairs.length - wins - losses;
    var duel = node('div', 'duel'); duel.append(node('strong', 'win', wins), node('strong', 'tie', ties), node('strong', 'loss', losses)); box.appendChild(duel);
    box.appendChild(node('p', 'muted', 'vitórias de A · empates · vitórias de B, em ' + pairs.length + ' testes comuns'));
    var axisNumbers = Array.from(new Set(pairs.map(function (pair) { return pair[0].axisNumber; }))).sort(function (x, y) { return x - y; });
    axisNumbers.forEach(function (axisNumber) {
      var relevant = pairs.filter(function (pair) { return pair[0].axisNumber === axisNumber; });
      var delta = relevant.reduce(function (sum, pair) { return sum + pair[0].score - pair[1].score; }, 0) / relevant.length;
      var row = node('div', 'axis-delta'); row.appendChild(node('span', '', capabilityName(axisNumber))); var track = node('div', 'delta-track'); var fill = node('span', 'delta-fill'); var width = Math.min(50, Math.abs(delta) / 10 * 50); fill.style.width = width + '%'; fill.style.left = delta >= 0 ? '50%' : (50 - width) + '%'; fill.style.background = delta >= 0 ? 'var(--lime)' : 'var(--red)'; track.appendChild(fill); row.append(track, node('strong', delta >= 0 ? 'pass' : 'fail', (delta > 0 ? '+' : '') + delta.toFixed(1))); box.appendChild(row);
    });
  }

  function renderLatency() {
    var box = $('latency'); box.replaceChildren(); [getCheckpoint(state.a), getCheckpoint(state.b)].filter(Boolean).forEach(function (checkpoint) {
      var card = node('div', 'latency-card'); var left = node('div'); left.append(node('strong', '', checkpoint.model), node('p', 'muted', 'Q1 ' + seconds(checkpoint.summary.q1LatencyMs) + ' · Q3 ' + seconds(checkpoint.summary.q3LatencyMs) + ' · P90 ' + seconds(checkpoint.summary.p90LatencyMs))); card.append(left, node('strong', '', seconds(checkpoint.summary.medianLatencyMs))); box.appendChild(card);
    }); box.appendChild(node('p', 'muted', 'A mediana inclui todo o teste, não apenas inferência. Outliers podem conter retries, ferramentas e LLM judge.'));
  }

  function renderProvenance() {
    var box = $('provenance'); box.replaceChildren();
    box.appendChild(makeTable(['Indicador', 'Valor', 'Interpretação'], [
      ['Checkpoints importados', data.checkpoints.length, 'Snapshots disponíveis, incluindo gerações antigas'],
      ['Modelos únicos exibidos', models.length, 'Apenas o checkpoint mais recente de cada modelo'],
      ['Duplicatas removidas', data.duplicates.length, 'Arquivos idênticos não contam duas vezes'],
      ['Avisos de validação', data.validation.length, 'Inconsistências de contagem ou parsing'],
      ['Catálogo atual', data.catalog.tests.length + ' testes', 'Eixos 1–23 da suíte v4'],
    ]));
  }

  function detailCell(test) {
    var details = node('details'); details.appendChild(node('summary', '', 'ver evidência'));
    details.appendChild(node('p', '', test.details || test.error || 'Sem detalhes no checkpoint legado.'));
    if (test.response) details.appendChild(node('pre', '', String(test.response).slice(0, 1500)));
    return details;
  }
  function renderExplorer() {
    var checkpoint = getCheckpoint(state.a); if (!checkpoint) return; var query = state.search.toLowerCase();
    var tests = checkpoint.tests.filter(function (test) { return (!state.axis || String(test.axisNumber) === state.axis) && (!query || (test.id + ' ' + test.name + ' ' + (test.details || '')).toLowerCase().includes(query)); });
    $('result-count').textContent = tests.length + ' testes';
    var rows = tests.map(function (test) { var status = node('span', test.status === 'pass' ? 'pass' : test.status === 'fail' ? 'fail' : 'skip-state', test.status === 'pass' ? '✓ passou' : test.status === 'fail' ? '✗ falhou' : '⚠ sem resultado'); return [test.id, capabilityName(test.axisNumber), test.name, test.difficulty, status, test.score == null ? '—' : test.score.toFixed(1), seconds(test.latencyMs), detailCell(test)]; });
    $('test-explorer').replaceChildren(makeTable(['ID', 'Eixo', 'Teste', 'Nível', 'Status', 'Score', 'Tempo', 'Evidência'], rows));
  }

  function render() { renderHero(); renderVerdict(); renderLeaderboard(); renderOutcomes(); renderHeatmap(); renderPairwise(); renderLatency(); renderProvenance(); renderExplorer(); }
  function setMode(mode) { state.mode = mode; document.body.classList.toggle('academic', mode === 'academic'); document.querySelectorAll('[data-mode]').forEach(function (button) { button.classList.toggle('active', button.dataset.mode === mode); }); localStorage.setItem('alpha-dashboard-mode', mode); }
  document.querySelectorAll('[data-mode]').forEach(function (button) { button.addEventListener('click', function () { setMode(button.dataset.mode); }); });
  $('model-a').addEventListener('change', function (event) { state.a = event.target.value; render(); });
  $('model-b').addEventListener('change', function (event) { state.b = event.target.value; renderPairwise(); renderLatency(); });
  $('axis-filter').addEventListener('change', function (event) { state.axis = event.target.value; renderExplorer(); });
  $('search').addEventListener('input', function (event) { state.search = event.target.value; renderExplorer(); });
  setMode(state.mode); render();
}());
