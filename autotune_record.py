#!/usr/bin/env python3
"""Store llama-bench JSON as a conservative Crono autotune record."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from autotune_cache import AutotuneCache, binary_identity, hardware_identity, model_identity, sha256_file
from launch_model_core import HardwareInfo, ModelMetadata, LLAMA_SERVER


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("llama-bench JSON must contain an array")
    return [row for row in payload if isinstance(row, dict)]


def _select_rows(rows: list[dict], prompt_tokens: int, generation_tokens: int) -> tuple[dict, dict]:
    prompt_rows = [
        row for row in rows
        if int(row.get("n_prompt", -1)) == prompt_tokens
        and int(row.get("n_gen", -1)) == 0
    ]
    generation_rows = [
        row for row in rows
        if int(row.get("n_prompt", -1)) == 0
        and int(row.get("n_gen", -1)) == generation_tokens
    ]
    if not prompt_rows and not generation_rows:
        raise ValueError(
            f"no benchmark rows for prompt={prompt_tokens}, generation={generation_tokens}"
        )
    return prompt_rows[0] if prompt_rows else {}, generation_rows[0] if generation_rows else {}


def _median_samples(row: dict) -> tuple[float, list[float]]:
    samples = row.get("samples_ts")
    if not isinstance(samples, list):
        samples = []
    values = []
    for value in samples:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    if not values:
        values = [float(row.get("avg_ts", 0.0) or 0.0)]
    return statistics.median(values), values


def _row_config(row: dict, args: argparse.Namespace) -> dict:
    no_kv_offload = bool(row.get("no_kv_offload", False))
    flash = "y" if row.get("flash_attn") else "n"
    device = str(row.get("devices", args.device))
    if device == "auto":
        device = "CUDA0" if args.device == "auto" else args.device
    n_cpu_moe = int(row.get("n_cpu_moe", args.n_cpu_moe))
    load_mode = str(row.get("load_mode", args.load_mode)).lower()
    if load_mode == "auto":
        load_mode = args.load_mode
    return {
        "ctx": args.ctx,
        "ngl": int(row.get("n_gpu_layers", args.ngl)),
        "parallel": args.parallel,
        "cache_k": str(row.get("type_k", args.cache_k)),
        "cache_v": str(row.get("type_v", args.cache_v)),
        "kv_unified": "y" if args.kv_unified else "n",
        "kv_offload": "n" if no_kv_offload else "y",
        "flash": flash,
        "batch": int(row.get("n_batch", args.batch)),
        "ubatch": int(row.get("n_ubatch", args.ubatch)),
        "threads": int(row.get("n_threads", args.threads)),
        "threads_batch": args.threads_batch,
        "n_cpu_moe": n_cpu_moe,
        "cpu_moe": "n",
        "split_mode": str(row.get("split_mode", args.split_mode)),
        "device": device,
        "numa": args.numa,
        "repack": "y",
        "load_mode": load_mode,
        "no_host": "y" if row.get("no_host") else "n",
        "fit": "n",
        "fit_target": int(row.get("fit_target", args.fit_target)),
        "fit_ctx": args.fit_ctx,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bench-json", required=True, type=Path)
    parser.add_argument("--server", default=LLAMA_SERVER, type=Path)
    parser.add_argument("--mode", default="interactive")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--threads-batch", type=int, default=0)
    parser.add_argument("--ngl", type=int, default=40)
    parser.add_argument("--n-cpu-moe", type=int, default=0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cache-k", default="q5_1")
    parser.add_argument("--cache-v", default="q5_1")
    parser.add_argument("--kv-unified", action="store_true")
    parser.add_argument("--split-mode", default="layer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--numa", default="none")
    parser.add_argument("--load-mode", default="mmap")
    parser.add_argument("--fit-target", type=int, default=0)
    parser.add_argument("--fit-ctx", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--sampler-seq", default="")
    parser.add_argument("--quality-notes", default="")
    parser.add_argument("--quality-validated", action="store_true")
    parser.add_argument("--quality-stable", action="store_true")
    parser.add_argument(
        "--apply", action="store_true",
        help="allow this record to affect a matching launcher workload",
    )
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    server_path = args.server.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not server_path.is_file():
        raise SystemExit(f"server not found: {server_path}")

    rows = _load_rows(args.bench_json.expanduser().resolve())
    prompt_row, generation_row = _select_rows(
        rows, args.prompt_tokens, args.generation_tokens
    )
    config_row = generation_row or prompt_row
    prompt_tps, prompt_samples = _median_samples(prompt_row) if prompt_row else (0.0, [])
    generation_tps, generation_samples = (
        _median_samples(generation_row) if generation_row else (0.0, [])
    )
    meta = ModelMetadata()
    meta.load(str(model_path))
    if not meta.meta_ok:
        raise SystemExit(f"could not read model metadata: {meta.metadata_error}")
    hw = HardwareInfo()
    hw.detect()
    model = model_identity(model_path, meta.arch, sha256_file(model_path))
    workload = {
        "mode": args.mode,
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "ctx": args.ctx,
        "batch": int(config_row.get("n_batch", args.batch)),
        "ubatch": int(config_row.get("n_ubatch", args.ubatch)),
        "parallel": args.parallel,
    }
    sampler = {
        "seed": args.seed,
        "temp": args.temp,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "sampler_seq": args.sampler_seq,
    }
    metrics = {
        "prompt_median_tps": prompt_tps,
        "prompt_samples_tps": prompt_samples,
        "generation_median_tps": generation_tps,
        "generation_samples_tps": generation_samples,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "score": generation_tps or prompt_tps,
        "build_commit": config_row.get("build_commit", ""),
    }
    quality = {
        "validated": bool(args.quality_validated),
        "stable": bool(args.quality_stable),
        "notes": args.quality_notes,
    }
    cache = AutotuneCache(args.cache) if args.cache else AutotuneCache()
    record = cache.record(
        model=model,
        hardware=hardware_identity(hw),
        runtime={"build": binary_identity(server_path)},
        workload=workload,
        sampler=sampler,
        config=_row_config(config_row, args),
        metrics=metrics,
        quality=quality,
        status="validated" if quality["validated"] and quality["stable"] else "candidate",
        apply_to_launch=args.apply,
        source="llama-bench-json",
    )
    print(json.dumps(record, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"autotune record failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
