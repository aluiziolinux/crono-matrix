#!/usr/bin/env python3
"""Falha se a árvore versionada contiver dados privados ou artefatos grandes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "patches" / "llama.cpp" / "crono-matrix.patch"
PATCH_SHA256 = "3a71b9d6d957377556f0c24d29ee46aa561ff54d022fb0d90fa91815a731a22e"
MODEL_SUFFIXES = {".gguf", ".safetensors", ".ckpt", ".pt", ".pth"}
FORBIDDEN_PARTS = {
    "mcp-crono-matrix", "node_modules", ".venv", "__pycache__",
    "eval_history", ".crono-agent", ".crono-native", ".crono-runtime",
}
PERSONAL_PATTERNS = (
    re.compile(r"/home/(?!user(?:/|\b)|usuario(?:/|\b)|<[^>]+>/)[A-Za-z0-9._-]+/"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)
TEXT_SUFFIXES = {
    ".py", ".js", ".mjs", ".ts", ".svelte", ".html", ".css", ".md",
    ".txt", ".toml", ".yaml", ".yml", ".json", ".sh", ".c", ".h",
    ".cpp", ".hpp", ".env", ".patch", "",
}


def tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        names = [name for name in result.stdout.split(b"\0") if name]
        if names:
            return [ROOT / os.fsdecode(name) for name in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]


def main() -> int:
    errors: list[str] = []
    files = tracked_files()
    for path in files:
        relative = path.relative_to(ROOT)
        parts = set(relative.parts)
        suffix = path.suffix.lower()
        if parts & FORBIDDEN_PARTS:
            errors.append(f"diretório proibido: {relative}")
        if "modelos" in parts or "models" in parts or suffix in MODEL_SUFFIXES:
            errors.append(f"modelo/checkpoint proibido: {relative}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"não foi possível ler {relative}: {exc}")
            continue
        if size > 10 * 1024 * 1024:
            errors.append(f"arquivo maior que 10 MiB: {relative} ({size} bytes)")
        if suffix in TEXT_SUFFIXES and size <= 2 * 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                errors.append(f"não foi possível inspecionar {relative}: {exc}")
                continue
            for pattern in PERSONAL_PATTERNS:
                if pattern.search(text):
                    errors.append(f"dado pessoal/segredo potencial em {relative}: {pattern.pattern}")

    if not PATCH.is_file():
        errors.append("patch do llama.cpp ausente")
    else:
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        if digest != PATCH_SHA256:
            errors.append(f"SHA-256 inesperado do patch: {digest}")

    if errors:
        print("RELEASE CHECK: FALHOU", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"RELEASE CHECK: OK — {len(files)} arquivos verificados; nenhum modelo/MCP incluído")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
