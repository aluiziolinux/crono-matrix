#!/usr/bin/env python3
"""Valida o guard C99 com um GGUF real em um processo isolado.

Uso recomendado:
  systemd-run --user --scope --unit=crono-memory-runtime \
    .venv/bin/python scripts/validate_memory_guard_runtime.py --match Tiel
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.services import LauncherWebState


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="Tiel-Coder-35B")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    state = LauncherWebState()
    state.refresh_hardware()
    models = state.scan_models()
    selected = next(
        (row for row in models if args.match.lower() in row["name"].lower()), None
    )
    if not selected:
        raise SystemExit(f"modelo contendo {args.match!r} não encontrado")
    state.select_model(selected["id"])
    raw = state.parameter_snapshot()["values"]
    raw.update({
        "host": "127.0.0.1",
        "port": str(args.port),
        "mcp_native": "n",
        "agent_compat": "n",
        "agent_global": "n",
        "omni": "n",
        "no_mmproj_auto": "y",
    })

    print(json.dumps({
        "event": "profile",
        "model": selected["name"],
        "ctx": raw.get("ctx"),
        "cache_k": raw.get("cache_k"),
        "cache_v": raw.get("cache_v"),
        "n_cpu_moe": raw.get("n_cpu_moe"),
        "swap": state.hardware_snapshot().get("swap_plan_reason"),
    }, ensure_ascii=False), flush=True)

    started = time.monotonic()
    try:
        state.start_server(raw)
        deadline = time.monotonic() + 300
        last_report = 0.0
        while time.monotonic() < deadline:
            process = state.process_snapshot()
            now = time.monotonic()
            if now - last_report >= 5:
                print(json.dumps({
                    "event": "load",
                    "elapsed_s": round(now - started, 1),
                    "state": process["state"],
                    "guard": process["memory_guard"],
                }, ensure_ascii=False), flush=True)
                last_report = now
            if process["ready"]:
                break
            if not process["running"]:
                raise RuntimeError(process.get("error") or "llama-server encerrou")
            time.sleep(0.5)
        else:
            raise TimeoutError("llama-server não ficou pronto em 300 s")

        process = state.process_snapshot()
        model_id = state.params.get("alias") or selected["name"].removesuffix(".gguf")
        inference_started = time.monotonic()
        answer = post_json(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{
                    "role": "user",
                    "content": "Responda somente com os números primos menores que 30.",
                }],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "stream": False,
            },
        )
        elapsed = time.monotonic() - inference_started
        usage = answer.get("usage", {})
        completion = int(usage.get("completion_tokens") or 0)
        print(json.dumps({
            "event": "result",
            "load_s": round(inference_started - started, 2),
            "inference_s": round(elapsed, 2),
            "completion_tokens": completion,
            "client_tokens_s": round(completion / elapsed, 2) if elapsed else 0,
            "server_timings": answer.get("timings", {}),
            "effective_ctx": process["runtime_effective"].get("context_window"),
            "guard": state.process_snapshot()["memory_guard"],
            "answer": answer.get("choices", [{}])[0].get("message", {}).get("content", "")[:300],
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        if state.is_running():
            state.stop_server()


if __name__ == "__main__":
    raise SystemExit(main())
