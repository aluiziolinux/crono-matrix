#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dashboardRoot = resolve(here, '..');
const suiteRoot = resolve(process.argv.find(arg => arg.startsWith('--suite-root='))?.slice(13) || join(dashboardRoot, '..'));
const extraRootArg = process.argv.find(arg => arg.startsWith('--extra-root='))?.slice(13);
const extraRoot = extraRootArg ? resolve(extraRootArg) : null;
const outputRoot = resolve(process.argv.find(arg => arg.startsWith('--output='))?.slice(9) || join(dashboardRoot, 'data'));
const inputPath = process.argv.find(arg => arg.startsWith('--input='))?.slice(8);
const inputRoot = process.argv.find(arg => arg.startsWith('--input-root='))?.slice(13);

const axisLabels = {
  1: 'Raciocínio', 2: 'Ferramentas', 3: 'Visão', 4: 'Contexto longo', 5: 'Identidade',
  6: 'Agente / VPS', 7: 'Código Node.js', 8: 'Diagnóstico de sistema', 9: 'Código C',
  10: 'Raciocínio geral', 11: 'Decisão agentic', 12: 'Matemática', 13: 'Multi-hop',
  14: 'Execução de código', 15: 'Calibração', 16: 'Adversarial', 17: 'C99 / SNN',
  18: 'VPS / Rede', 19: 'Metacognição', 20: 'Bugs ocultos', 21: 'Resiliência',
  22: 'Contexto cruzado', 23: 'Generalização procedural',
};

const slug = value => String(value).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const finite = value => Number.isFinite(Number(value)) ? Number(value) : null;
const quantile = (values, q) => {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position), upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
};
const wilson = (successes, n) => {
  if (!n) return null;
  const z = 1.959963984540054, p = successes / n, d = 1 + z * z / n;
  const center = (p + z * z / (2 * n)) / d;
  const half = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d;
  return [Math.max(0, center - half), Math.min(1, center + half)];
};

async function walk(root) {
  const files = [];
  try {
    for (const entry of await readdir(root, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name === 'eval_dashboard') continue;
      const path = join(root, entry.name);
      if (entry.isDirectory()) files.push(...await walk(path));
      else if (/^(checkpoint_.*\.json|history\.json|eval_results\.json)$/.test(entry.name)) files.push(path);
    }
  } catch {}
  return files;
}

async function buildCatalog() {
  const source = await readFile(join(suiteRoot, 'alpha_eval_suite_v3.3.mjs'), 'utf8');
  const tests = [];
  const pattern = /runTest\s*\(\s*\{\s*id:\s*(['"])(.*?)\1\s*,\s*axis:\s*(['"])(.*?)\3\s*,\s*name:\s*(['"])(.*?)\5(?:\s*,\s*difficulty:\s*(['"])(.*?)\7)?/g;
  for (const match of source.matchAll(pattern)) {
    const axisNumber = Number(match[2].match(/^\d+/)?.[0]);
    tests.push({ id: match[2], axis: match[4], axisNumber, axisLabel: axisLabels[axisNumber] || match[4], name: match[6], difficulty: match[8] || 'medium' });
  }
  return tests.sort((a, b) => a.axisNumber - b.axisNumber || a.id.localeCompare(b.id, undefined, { numeric: true }));
}

function generationFor(tests) {
  const maxAxis = Math.max(0, ...tests.map(test => Number(test.id?.match(/^\d+/)?.[0]) || 0));
  if (maxAxis >= 23) return { id: 'alpha-v4-23-axis', expected: 120 };
  if (maxAxis >= 22 || tests.length >= 110) return { id: 'alpha-v3.3-22-axis', expected: 115 };
  if (maxAxis >= 18 || tests.length >= 100) return { id: 'alpha-v3.2-18-axis', expected: 102 };
  return { id: 'alpha-v3-16-axis', expected: 89 };
}

function summarize(tests) {
  const evaluated = tests.filter(test => test.passed === true || test.passed === false);
  const passed = evaluated.filter(test => test.passed).length;
  const scores = tests.map(test => finite(test.score)).filter(Number.isFinite);
  const latencies = tests.map(test => finite(test.latencyMs)).filter(value => value >= 0);
  const candidateMetrics = tests.map(test => test.metrics?.candidate).filter(Boolean);
  const promptTokens = candidateMetrics.reduce((sum, item) => sum + (finite(item.prompt_tokens) || 0), 0);
  const completionTokens = candidateMetrics.reduce((sum, item) => sum + (finite(item.completion_tokens) || 0), 0);
  const candidateWallMs = candidateMetrics.reduce((sum, item) => sum + (finite(item.wall_ms) || 0), 0);
  const axes = {};
  for (const test of tests) {
    if (!axes[test.axisNumber]) axes[test.axisNumber] = { axisNumber: test.axisNumber, label: test.axisLabel, available: 0, evaluated: 0, passed: 0, scores: [] };
    const axis = axes[test.axisNumber]; axis.available++;
    if (test.passed === true || test.passed === false) axis.evaluated++;
    if (test.passed === true) axis.passed++;
    if (Number.isFinite(test.score)) axis.scores.push(test.score);
  }
  return {
    available: tests.length, evaluated: evaluated.length, passed, failed: evaluated.length - passed,
    skipped: tests.length - evaluated.length,
    passRate: evaluated.length ? passed / evaluated.length : null,
    passRateCI95: wilson(passed, evaluated.length),
    meanScore: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null,
    medianScore: quantile(scores, .5),
    medianLatencyMs: quantile(latencies, .5), q1LatencyMs: quantile(latencies, .25),
    q3LatencyMs: quantile(latencies, .75), p90LatencyMs: quantile(latencies, .9), maxLatencyMs: quantile(latencies, 1),
    promptTokens, completionTokens, candidateWallMs,
    completionTokensPerSecond: completionTokens > 0 && candidateWallMs > 0 ? completionTokens / (candidateWallMs / 1000) : null,
    axes: Object.values(axes).map(axis => ({ ...axis, meanScore: axis.scores.length ? axis.scores.reduce((a, b) => a + b, 0) / axis.scores.length : null, passRate: axis.evaluated ? axis.passed / axis.evaluated : null, scores: undefined })),
  };
}

function normalizeCheckpoint(raw, path, catalogMap, warnings) {
  if (!raw || !Array.isArray(raw.tests)) return null;
  const tests = raw.tests.map(source => {
    const metadata = catalogMap.get(String(source.id)) || {};
    const passed = source.passed === true ? true : source.passed === false ? false : null;
    const score = finite(source.score) ?? (passed === true ? 10 : passed === false ? 0 : null);
    const axisNumber = metadata.axisNumber || Number(String(source.id).match(/^\d+/)?.[0]) || 0;
    const name = source.name || metadata.name || `Teste ${source.id}`;
    return {
      id: String(source.id), axis: source.axis || metadata.axis || `axis-${axisNumber}`, axisNumber,
      axisLabel: metadata.axisLabel || axisLabels[axisNumber] || `Eixo ${axisNumber}`,
      name, difficulty: source.difficulty || metadata.difficulty || 'unknown', passed, score,
      status: passed === true ? 'pass' : passed === false ? 'fail' : 'skip',
      latencyMs: finite(source.latency), details: source.details || null, response: source.response || null,
      error: source.error || null, failureType: source.failure_type || (passed === false ? 'model_quality' : null),
      runs: source.runs || null, metrics: source.metrics || null,
      compatibilityKey: `${generationFor(raw.tests).id}|${source.id}|${slug(name)}`,
    };
  });
  const generation = raw.schema_version === '4.0'
    ? { id: 'alpha-v4-23-axis', expected: 120 }
    : generationFor(tests);
  const summary = summarize(tests);
  const stored = { pass: finite(raw.pass), fail: finite(raw.fail), skip: finite(raw.skip) };
  if (stored.pass != null && (stored.pass !== summary.passed || stored.fail !== summary.failed || stored.skip !== summary.skipped)) warnings.push({ source: path, type: 'count_mismatch', message: 'Contagens armazenadas diferem das contagens recalculadas.' });
  return {
    model: raw.runtime_context?.variant?.display_name || raw.model || basename(path),
    baseModel: raw.model || basename(path),
    configFingerprint: raw.runtime_context?.variant?.fingerprint || null,
    variantLabel: raw.runtime_context?.variant?.label || null,
    runId: raw.run_id || null,
    timestamp: raw.timestamp || raw.runtime_context?.captured_at || null,
    sourceType: basename(path) === 'eval_results.json' ? 'complete-run' : 'checkpoint',
    suiteGeneration: generation.id, expectedTests: generation.expected,
    schemaVersion: raw.schema_version || 'legacy', config: raw.config || null,
    runtimeContext: raw.runtime_context || null,
    tests, summary: { ...summary, coverage: generation.expected ? summary.available / generation.expected : null },
    sourceRefs: [path], warnings: [],
  };
}

const catalog = await buildCatalog();
const catalogMap = new Map(catalog.map(test => [test.id, test]));
const candidates = inputPath
  ? [resolve(inputPath)]
  : inputRoot
    ? await walk(resolve(inputRoot))
    : [...new Set([
        ...(await walk(join(suiteRoot, 'eval_history'))),
        ...(extraRoot ? await walk(extraRoot) : []),
        join(suiteRoot, 'eval_results.json'),
      ])];
const hashes = new Map(), checkpoints = [], duplicates = [], warnings = [], history = [];

for (const path of candidates.sort()) {
  try {
    const text = await readFile(path, 'utf8');
    const hash = createHash('sha256').update(text).digest('hex');
    if (hashes.has(hash)) { duplicates.push({ canonical: hashes.get(hash), duplicate: path, hash: hash.slice(0, 12) }); continue; }
    hashes.set(hash, path);
    const raw = JSON.parse(text);
    if (basename(path) === 'history.json' && Array.isArray(raw)) {
      for (const entry of raw) history.push({ ...entry, api: undefined, source: path });
      continue;
    }
    const checkpoint = normalizeCheckpoint(raw, path, catalogMap, warnings);
    if (!checkpoint) continue;
    checkpoint.id = hash.slice(0, 12);
    checkpoints.push(checkpoint);
  } catch (error) {
    warnings.push({ source: path, type: 'parse_error', message: error.message });
  }
}

history.sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));
checkpoints.sort((a, b) => a.model.localeCompare(b.model) || String(b.timestamp).localeCompare(String(a.timestamp)));

const data = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  methodology: 'Resultados descritivos de uma suíte local heterogênea; não equivalem a uma medida absoluta de inteligência.',
  catalog: { generation: 'alpha-v4-23-axis', expectedTests: catalog.length, axes: Object.entries(axisLabels).map(([number, label]) => ({ number: Number(number), label })), tests: catalog },
  checkpoints, history, duplicates, validation: warnings,
};

await mkdir(outputRoot, { recursive: true });
const json = JSON.stringify(data, null, 2);
await writeFile(join(outputRoot, 'dashboard-data.json'), json);
await writeFile(join(outputRoot, 'dashboard-data.js'), `window.__EVAL_DASHBOARD_DATA__ = ${json.replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029')};\n`);
await writeFile(join(outputRoot, 'validation-report.json'), JSON.stringify({ duplicates, warnings }, null, 2));
console.log(`Dashboard: ${checkpoints.length} checkpoints, ${history.length} entradas históricas, ${duplicates.length} duplicatas, ${warnings.length} avisos.`);
