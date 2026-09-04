#!/usr/bin/env node
const https = require("https");
const http = require("http");
const fs = require("fs");
const path = require("path");
const readline = require("readline");

const HF_API_BASE = "https://huggingface.co/api";
const HF_RESOLVE_BASE = "https://huggingface.co";

const TRUSTED_USERS = [
  ["lmstudio-community", 3],
  ["bartowski", 2],
  ["TheBloke", 1],
  ["mradermacher", 1],
  ["MaziyarPanahi", 1],
];

function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith("https") ? https : http;
    proto.get(url, { headers: { "User-Agent": "hf-models-tool/1.0" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return fetchJSON(res.headers.location).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error(`Invalid JSON: ${e.message}`));
        }
      });
    }).on("error", reject);
  });
}

function fetchStream(url, onProgress) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith("https") ? https : http;
    proto.get(url, { headers: { "User-Agent": "hf-models-tool/1.0" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return fetchStream(res.headers.location, onProgress).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      const total = parseInt(res.headers["content-length"] || "0", 10);
      let downloaded = 0;
      const startTime = Date.now();
      resolve({
        stream: res,
        total,
        onData(chunk) {
          downloaded += chunk.length;
          const elapsed = (Date.now() - startTime) / 1000;
          const speed = elapsed > 0 ? downloaded / elapsed : 0;
          onProgress({ downloaded, total, speed });
        },
      });
    }).on("error", reject);
  });
}

function parseHFUrl(input) {
  try {
    const url = new URL(input);
    if (url.hostname !== "huggingface.co") return null;
    const segments = url.pathname.split("/").filter(Boolean);
    const [user, repo, ...rest] = segments;
    if (!user || !repo) return null;
    const fileNamePref = rest.length > 0 ? rest[rest.length - 1] : undefined;
    return { type: "huggingface", user, repo, fileNamePreference: fileNamePref };
  } catch {
    return null;
  }
}

function splitModelName(input) {
  const parts = input.split("@");
  if (parts.length > 2) throw new Error("At most one @ allowed");
  return { name: parts[0].trim(), quant: parts[1]?.trim() };
}

function cleanFileName(name) {
  return name.replace(/-I?Q\d[_0-9A-Za-z]{0,6}/g, "");
}

function findBreakPoints(name) {
  const points = [];
  let inBreak = false;
  for (let i = 0; i < name.length; i++) {
    if (name[i] === "-" || name[i] === ".") {
      if (!inBreak) { points.push(i); inBreak = true; }
    } else {
      inBreak = false;
    }
  }
  return points;
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + " " + units[i];
}

function formatSpeed(bytesPerSec) {
  return formatBytes(bytesPerSec) + "/s";
}

function formatTime(seconds) {
  if (seconds < 60) return Math.round(seconds) + "s";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m " + Math.round(seconds % 60) + "s";
  return Math.floor(seconds / 3600) + "h " + Math.floor((seconds % 3600) / 60) + "m";
}

// ─── Commands ─────────────────────────────────────────────────────────────

async function cmdSearch(term, { limit = 20 } = {}) {
  console.error(`🔍 Searching HuggingFace for "${term}"...`);
  const data = await fetchJSON(`${HF_API_BASE}/models?search=${encodeURIComponent(term)}&full=true&sort=likes`);
  const results = data.slice(0, limit);
  for (const r of results) {
    const ggufFiles = (r.siblings || []).filter(s => s.rfilename.endsWith(".gguf"));
    const quantInfo = ggufFiles.length > 0 ? ` (${ggufFiles.length} GGUF files)` : "";
    console.log(`${r.id}${quantInfo}`);
    console.log(`  ❤️ ${r.likes || 0}  ⬇️ ${r.downloads || 0}`);
    if (r.pipeline_tag) console.log(`  🏷️ ${r.pipeline_tag}`);
    if (ggufFiles.length > 0 && ggufFiles.length <= 5) {
      for (const f of ggufFiles) {
        console.log(`  📄 ${f.rfilename}`);
      }
    }
    console.log();
  }
  console.error(`Found ${results.length} results (showing top ${limit})`);
}

async function cmdInfo(input) {
  let user, repo;
  const parsed = parseHFUrl(input);
  if (parsed) {
    user = parsed.user; repo = parsed.repo;
  } else if (input.includes("/")) {
    [user, repo] = input.split("/", 2);
  } else {
    console.error("Provide owner/repo or a huggingface.co URL");
    process.exit(1);
  }

  console.error(`📦 Fetching info for ${user}/${repo}...`);
  const data = await fetchJSON(`${HF_API_BASE}/models/${user}/${repo}`);
  console.log(`Model: ${data.id}`);
  console.log(`Downloads: ${data.downloads || 0}`);
  console.log(`Likes: ${data.likes || 0}`);
  if (data.pipeline_tag) console.log(`Type: ${data.pipeline_tag}`);
  if (data.config?.model_type) console.log(`Arch: ${data.config.model_type}`);
  if (data.cardData?.language) console.log(`Language: ${data.cardData.language}`);
  console.log(`License: ${data.cardData?.license || "N/A"}`);

  const siblings = data.siblings || [];
  const ggufFiles = siblings.filter(s => s.rfilename.endsWith(".gguf"));
  const otherFiles = siblings.filter(s => !s.rfilename.endsWith(".gguf"));

  console.log(`\n📄 GGUF files (${ggufFiles.length}):`);
  for (const f of ggufFiles) {
    const size = f.size ? ` [${formatBytes(f.size)}]` : "";
    const isConfig = f.rfilename.includes("config") ? " (config)" : "";
    console.log(`  ${f.rfilename}${size}${isConfig}`);
  }

  if (otherFiles.length > 0) {
    console.log(`\n📄 Other files (${Math.min(otherFiles.length, 10)} shown):`);
    for (const f of otherFiles.slice(0, 10)) {
      const size = f.size ? ` [${formatBytes(f.size)}]` : "";
      console.log(`  ${f.rfilename}${size}`);
    }
    if (otherFiles.length > 10) console.log(`  ... and ${otherFiles.length - 10} more`);
  }
}

async function cmdDownload(input, { output = "." } = {}) {
  let user, repo, fileName;
  const parsed = parseHFUrl(input);
  if (parsed) {
    user = parsed.user;
    repo = parsed.repo;
    fileName = parsed.fileNamePreference;
  } else if (input.includes("/")) {
    const parts = input.split("/");
    if (parts.length >= 2) {
      user = parts[0];
      repo = parts[1];
      fileName = parts[2];
    }
  } else {
    console.error("Provide owner/repo/file or a huggingface.co URL");
    process.exit(1);
  }

  if (!fileName) {
    console.error(`🔍 Fetching file list for ${user}/${repo}...`);
    const data = await fetchJSON(`${HF_API_BASE}/models/${user}/${repo}`);
    const siblings = data.siblings || [];
    const ggufFiles = siblings.filter(s => s.rfilename.endsWith(".gguf") && !s.rfilename.includes("config"));

    if (ggufFiles.length === 0) {
      console.error("No GGUF files found in this repo.");
      process.exit(1);
    }

    if (ggufFiles.length === 1) {
      fileName = ggufFiles[0].rfilename;
      console.error(`📄 Found single GGUF file: ${fileName}`);
    } else {
      console.error(`\n📄 Select a GGUF file to download:`);
      for (let i = 0; i < ggufFiles.length; i++) {
        const size = ggufFiles[i].size ? ` [${formatBytes(ggufFiles[i].size)}]` : "";
        console.error(`  ${i + 1}. ${ggufFiles[i].rfilename}${size}`);
      }
      const answer = await ask(`Enter number (1-${ggufFiles.length}): `);
      const idx = parseInt(answer, 10) - 1;
      if (isNaN(idx) || idx < 0 || idx >= ggufFiles.length) {
        console.error("Invalid selection");
        process.exit(1);
      }
      fileName = ggufFiles[idx].rfilename;
    }
  }

  const url = `${HF_RESOLVE_BASE}/${user}/${repo}/resolve/main/${fileName}`;
  const dest = path.join(output, fileName);

  console.error(`\n⬇️  Downloading:`);
  console.error(`  From: ${url}`);
  console.error(`  To:   ${dest}`);

  if (fs.existsSync(dest)) {
    console.error(`⚠️  File already exists: ${dest}`);
    const answer = await ask("Overwrite? (y/N): ");
    if (answer.toLowerCase() !== "y") {
      console.error("Skipped.");
      return;
    }
  }

  fs.mkdirSync(path.dirname(dest), { recursive: true });

  const { stream, total, onData } = await fetchStream(url, (p) => {
    const pct = total > 0 ? (p.downloaded / total * 100).toFixed(1) : "?";
    const elapsed = p.total > 0 ? formatTime((Date.now() - startTime) / 1000) : "?";
    const eta = p.speed > 0 && p.total > 0 ? formatTime((p.total - p.downloaded) / p.speed) : "?";
    process.stderr.write(
      `\r  ${formatBytes(p.downloaded)} / ${formatBytes(p.total)} (${pct}%)  ${formatSpeed(p.speed)}  ETA ${eta}     `
    );
  });

  const startTime = Date.now();
  const fileStream = fs.createWriteStream(dest);

  return new Promise((resolve, reject) => {
    stream.on("data", (chunk) => {
      onData(chunk);
      fileStream.write(chunk);
    });
    stream.on("end", () => {
      fileStream.end();
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.error(`\n✅ Done in ${elapsed}s — ${formatBytes(total)}`);
      resolve();
    });
    stream.on("error", (err) => {
      fileStream.close();
      fs.unlink(dest, () => {});
      reject(err);
    });
  });
}

async function cmdResolve(filePath) {
  const fileName = path.basename(filePath);
  console.error(`🔍 Resolving "${fileName}" on HuggingFace...`);

  const clean = cleanFileName(fileName);
  console.error(`  Cleaned name: ${clean}`);

  const points = findBreakPoints(clean);
  points.push(clean.length);

  let candidates = [];
  for (let i = points.length - 1; i >= 0; i--) {
    const term = clean.substring(0, points[i]);
    if (term.length < 3) continue;
    console.error(`  Searching with term: "${term}"`);
    const repos = await fetchJSON(`${HF_API_BASE}/models?search=${encodeURIComponent(term)}&full=true&sort=likes`);
    for (const repo of repos) {
      if ((repo.siblings || []).some(s => s.rfilename.toLowerCase() === fileName.toLowerCase())) {
        const split = repo.id.split("/");
        if (split.length === 2) candidates.push(split);
      }
    }
    if (candidates.length > 0) break;
  }

  if (candidates.length === 0) {
    console.error("❌ Could not find matching repo on HuggingFace.");
    return;
  }

  const userScore = new Map(TRUSTED_USERS);
  candidates.sort((a, b) => (userScore.get(a[0]) || 0) - (userScore.get(b[0]) || 0));

  console.error(`\n🔎 Found ${candidates.length} candidate repos:`);
  for (let i = 0; i < Math.min(candidates.length, 25); i++) {
    const [u, r] = candidates[i];
    const score = userScore.get(u) || 0;
    const stars = "★".repeat(score) + "☆".repeat(3 - score);
    console.error(`  ${String(i + 1).padStart(2)}. ${u}/${r} ${stars}`);
  }

  if (candidates.length === 1) {
    console.error(`\n✅ Auto-selected: ${candidates[0][0]}/${candidates[0][1]}`);
    console.log(`${candidates[0][0]}/${candidates[0][1]}`);
    return;
  }

  const answer = await ask(`\nSelect repo (1-${Math.min(candidates.length, 25)}): `);
  const idx = parseInt(answer, 10) - 1;
  if (isNaN(idx) || idx < 0 || idx >= Math.min(candidates.length, 25)) {
    console.error("Invalid selection");
    process.exit(1);
  }
  console.log(`${candidates[idx][0]}/${candidates[idx][1]}`);
}

function ask(question) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stderr });
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

// ─── CLI ──────────────────────────────────────────────────────────────────

function help() {
  console.log(`
HF Models — HuggingFace model search & download tool
(Reconstructed from LM Studio's mechanism)

USAGE:
  hf-models search <term>              Search models on HuggingFace
  hf-models info <owner/repo>          Show model details and file list
  hf-models download <url|repo/file>   Download a GGUF model file
  hf-models resolve <file.guf>         Find which HF repo a file belongs to
`);
}

async function main() {
  const args = process.argv.slice(2);
  const cmd = args[0];

  if (!cmd || cmd === "--help" || cmd === "-h") {
    help();
    return;
  }

  switch (cmd) {
    case "search": {
      const term = args[1];
      if (!term) { console.error("Usage: hf-models search <term>"); process.exit(1); }
      await cmdSearch(term);
      break;
    }
    case "info": {
      const id = args[1];
      if (!id) { console.error("Usage: hf-models info <owner/repo>"); process.exit(1); }
      await cmdInfo(id);
      break;
    }
    case "download": {
      const id = args[1];
      if (!id) { console.error("Usage: hf-models download <owner/repo/file>"); process.exit(1); }
      const outDir = args[2] || ".";
      await cmdDownload(id, { output: outDir });
      break;
    }
    case "resolve": {
      const file = args[1];
      if (!file) { console.error("Usage: hf-models resolve <file.guf>"); process.exit(1); }
      await cmdResolve(file);
      break;
    }
    default:
      console.error(`Unknown command: ${cmd}`);
      help();
      process.exit(1);
  }
}

main().catch((err) => {
  console.error(`\n❌ Error: ${err.message}`);
  process.exit(1);
});
