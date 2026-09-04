"""Persistent, conservative cache for measured llama.cpp configurations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path


CACHE_VERSION = 1
DEFAULT_CACHE_FILE = Path(os.environ.get(
    "CRONO_AUTOTUNE_CACHE",
    Path.home() / ".cache" / "crono-launcher" / "autotune.json",
)).expanduser()

RUNTIME_CONFIG_KEYS = frozenset({
    "ctx", "ngl", "parallel", "cache_k", "cache_v", "kv_unified",
    "kv_offload", "flash", "batch", "ubatch", "threads", "threads_batch",
    "n_cpu_moe", "cpu_moe", "n_cpu_ffn", "split_mode", "device", "numa",
    "repack", "load_mode", "tensor_read_lazy", "no_host", "fit",
    "fit_target", "fit_ctx",
})

_CUDA_VERSION = None


def _cuda_toolkit_version() -> str:
    global _CUDA_VERSION
    if _CUDA_VERSION is not None:
        return _CUDA_VERSION
    candidates = [
        os.environ.get("CUDA_HOME", ""),
        os.environ.get("CUDA_PATH", ""),
        "/opt/cuda",
    ]
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.insert(0, str(Path(nvcc).resolve()))
    for candidate in candidates:
        candidate = str(candidate)
        executable = candidate if candidate.endswith("nvcc") else str(Path(candidate) / "bin" / "nvcc")
        if not executable or not Path(executable).is_file():
            continue
        try:
            result = subprocess.run(
                [executable, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(r"release\s+([0-9]+\.[0-9]+)", result.stdout or "", re.IGNORECASE)
        if match:
            _CUDA_VERSION = match.group(1)
            return _CUDA_VERSION
    _CUDA_VERSION = ""
    return _CUDA_VERSION


def _canonical(value):
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value: dict) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a model only when a measured cache record is being created."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path: str | Path, architecture: str = "", digest: str = "") -> dict:
    model_path = Path(path).expanduser().resolve()
    stat = model_path.stat()
    return {
        "path": str(model_path),
        "name": model_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": str(digest or "").lower(),
        "architecture": str(architecture or "").lower(),
    }


def hardware_identity(hw) -> dict:
    return {
        "cpu_model": str(getattr(hw, "cpu_model", "")),
        "cpu_cores": int(getattr(hw, "cpu_cores", 0) or 0),
        "cpu_threads": int(getattr(hw, "cpu_threads", 0) or 0),
        "numa_nodes": int(getattr(hw, "numa_nodes", 0) or 0),
        "gpu_model": str(getattr(hw, "gpu_model", "")),
        "gpu_vram_mb": int(getattr(hw, "gpu_vram_mb", 0) or 0),
        "gpu_driver": str(getattr(hw, "gpu_driver", "")),
        "gpu_cuda": str(getattr(hw, "gpu_cuda", "")),
        "cuda_toolkit": _cuda_toolkit_version(),
    }


def binary_identity(path: str | Path) -> dict:
    executable = Path(path).expanduser().resolve()
    result = {"path": str(executable)}
    try:
        stat = executable.stat()
        result.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    except OSError:
        return result

    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
        output = completed.stdout or ""
        result["version_output"] = output.strip().splitlines()[0][:160] if output else ""
        match = re.search(r"\(([0-9a-f]{7,40})\)", output, re.IGNORECASE)
        if match:
            result["commit"] = match.group(1).lower()
    except (OSError, subprocess.SubprocessError):
        result["version_output"] = ""
    return result


class AutotuneCache:
    """Small JSON cache; only explicit validated records can affect launch."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_FILE):
        self.path = Path(path).expanduser()
        self.lock = threading.RLock()

    def _read(self) -> dict:
        try:
            with self.path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
                return {"version": CACHE_VERSION, "records": []}
            records = payload.get("records")
            return {
                "version": CACHE_VERSION,
                "records": records if isinstance(records, list) else [],
            }
        except (OSError, ValueError, TypeError):
            return {"version": CACHE_VERSION, "records": []}

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
        os.replace(temporary, self.path)

    @staticmethod
    def _model_matches(record: dict, query: dict) -> bool:
        record_sha = str(record.get("sha256") or "").lower()
        query_sha = str(query.get("sha256") or "").lower()
        if query_sha:
            return query_sha == record_sha and query.get("architecture") == record.get("architecture")
        for key in ("architecture", "size", "mtime_ns", "path"):
            if query.get(key) != record.get(key):
                return False
        return True

    @staticmethod
    def _matches(record_value: dict, query_value: dict) -> bool:
        if not isinstance(record_value, dict) or not isinstance(query_value, dict):
            return record_value == query_value
        return all(record_value.get(key) == value for key, value in query_value.items())

    @staticmethod
    def _score(record: dict) -> float:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            return float("-inf")
        try:
            explicit = metrics.get("score")
            if explicit is not None:
                return float(explicit)
            generation = float(metrics.get("generation_tps", 0) or 0)
            prompt = float(metrics.get("prompt_tps", 0) or 0)
            return generation + prompt * 0.1
        except (TypeError, ValueError):
            return float("-inf")

    def resolve(
        self,
        model: dict,
        hardware: dict,
        runtime: dict,
        workload: dict,
        sampler: dict,
    ) -> dict | None:
        with self.lock:
            records = self._read().get("records", [])
        # Never hash a multi-gigabyte GGUF while merely selecting it in the
        # launcher. The local identity (resolved path, size and nanosecond
        # mtime) is sufficient to resolve an already validated local record.
        # SHA-256 remains part of explicit record creation/verification.
        candidates = []
        for record in records:
            if not isinstance(record, dict):
                continue
            if record.get("status") != "validated" or not record.get("apply_to_launch"):
                continue
            if not self._model_matches(record.get("model", {}), model):
                continue
            if not self._matches(record.get("hardware", {}), hardware):
                continue
            if not self._matches(record.get("runtime", {}), runtime):
                continue
            if not self._matches(record.get("workload", {}), workload):
                continue
            if not self._matches(record.get("sampler", {}), sampler):
                continue
            candidates.append(record)
        if not candidates:
            return None
        selected = max(candidates, key=self._score)
        return {
            "status": "hit",
            "record_id": selected.get("record_id", ""),
            "config": dict(selected.get("config", {})),
            "metrics": dict(selected.get("metrics", {})),
            "quality": dict(selected.get("quality", {})),
            "source": selected.get("source", "autotune"),
            "created_at": selected.get("created_at", ""),
        }

    def record(
        self,
        model: dict,
        hardware: dict,
        runtime: dict,
        workload: dict,
        sampler: dict,
        config: dict,
        metrics: dict,
        quality: dict | None = None,
        status: str = "validated",
        apply_to_launch: bool = False,
        source: str = "llama-bench",
    ) -> dict:
        unknown = set(config) - RUNTIME_CONFIG_KEYS
        if unknown:
            raise ValueError(
                "autotune config contains non-runtime keys: "
                + ", ".join(sorted(unknown))
            )
        if not model.get("sha256"):
            raise ValueError("autotune records require the full model sha256")
        if apply_to_launch and not (
            isinstance(quality, dict)
            and quality.get("validated") is True
            and quality.get("stable") is True
        ):
            raise ValueError(
                "autotune records applied to launch require "
                "quality.validated=true and quality.stable=true"
            )
        base = {
            "model": _canonical(model),
            "hardware": _canonical(hardware),
            "runtime": _canonical(runtime),
            "workload": _canonical(workload),
            "sampler": _canonical(sampler),
        }
        record = {
            "record_id": fingerprint({**base, "config": config}),
            **base,
            "config": _canonical(config),
            "metrics": _canonical(metrics),
            "quality": _canonical(quality or {}),
            "status": status,
            "apply_to_launch": bool(apply_to_launch),
            "source": source,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self.lock:
            payload = self._read()
            records = [
                item for item in payload["records"]
                if isinstance(item, dict) and item.get("record_id") != record["record_id"]
            ]
            records.append(record)
            payload["records"] = records
            self._write(payload)
        return record

    def snapshot(self) -> dict:
        with self.lock:
            payload = self._read()
        return {"path": str(self.path), "version": CACHE_VERSION,
                "records": len(payload.get("records", []))}
