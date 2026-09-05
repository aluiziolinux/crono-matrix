"""Servicos de estado para a interface web, sem dependencia de FastAPI."""

from collections import deque
from pathlib import Path
import csv
import hashlib
import io
import ipaddress
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from launch_model_core import (
    HF_API,
    HF_TRUSTED,
    LLAMA_SERVER,
    MEDIA_PATH,
    MODELS_DIR,
    HardwareInfo,
    HuggingFaceHub,
    ModelMetadata,
    OptimalParams,
    SPECULATIVE_TYPES,
    configure_gguf_py_dir,
    _server_lazy_mode_flag,
    _server_supports_flag,
    _gguf_total_size,
    _hf_base_model_name,
    _hf_fetch_json,
    _is_auxiliary_gguf,
    _requires_symmetric_kv,
    _is_secondary_shard,
)
from autotune_cache import RUNTIME_CONFIG_KEYS


INTEGER_FIELDS = {
    "ctx", "parallel", "threads", "threads_batch", "batch", "ubatch",
    "poll", "port", "cache_reuse", "top_k", "reasoning_budget",
    "mtmd_batch_max", "image_min_tokens", "image_max_tokens", "n_cpu_moe",
    "n_cpu_ffn",
    "sleep_idle", "fit_target", "fit_ctx", "cache_ram", "ctx_checkpoints",
    "checkpoint_min_step", "timeout", "log_verbosity", "agentic_max_turns",
    "agentic_max_tool_preview_lines", "spec_draft_n_max", "spec_draft_n_min",
    "spec_ngram_mod_n_min", "spec_ngram_mod_n_max", "spec_ngram_mod_n_match",
    "spec_ngram_min_hits", "repeat_last_n", "seed", "mirostat",
    "threads_http", "sse_ping_interval", "dry_penalty_last_n",
    "dry_allowed_length", "yarn_orig_ctx", "mcp_snn_threads", "mcp_snn_steps",
    "mcp_repeat_limit",
}
FLOAT_FIELDS = {
    "temp", "top_p", "repeat_penalty", "min_p", "presence_penalty",
    "slot_similarity", "spec_draft_p_min", "spec_draft_p_split",
    "frequency_penalty", "dry_multiplier", "dry_base", "top_nsigma",
    "typical_p", "xtc_probability", "xtc_threshold", "dynatemp_range",
    "dynatemp_exp", "mirostat_lr", "mirostat_ent", "adaptive_target",
    "adaptive_decay", "rope_freq_base", "rope_freq_scale", "rope_scale",
    "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_slow", "yarn_beta_fast",
}
KV_CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"}
AGENTIC_CONTEXT_MIN = 16384
# A RTX 3060 com modelos MoE grandes pode manter mais de 23 GiB de pesos
# residentes na CPU.  512 MiB deixava pouco espaço para compositor, browser e
# alocações transitórias; isto é uma reserva de *admissão* antes de carregar,
# não um limite ``memory.high`` durante a inferência.
SYSTEM_RAM_RESERVE_MB = 1024
RAM_CONTROL_TRIGGER_MB = 1536
HF_RADAR_DEFAULT_WATCHLIST = (
    "qwen,gemma,glm,deepseek,mistral,llama,nemotron,gpt-oss"
)
HF_RADAR_MAX_SEEN = 600
HF_RADAR_MIN_REFRESH_SECONDS = 45
HF_RADAR_OFFICIAL_OWNERS = {
    "qwen", "google", "zai-org", "thudm", "deepseek-ai", "mistralai",
    "meta-llama", "nvidia", "openai", "microsoft", "ibm-granite",
}
SETTINGS_FILE = Path(os.environ.get(
    "CRONO_SETTINGS_FILE", Path.home() / ".config" / "crono-launcher" / "settings.json"
)).expanduser()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NVME_SWAP_MANAGER = PROJECT_ROOT / "scripts" / "manage_nvme_swap.sh"
MEMORY_GUARD_SOURCE = PROJECT_ROOT / "native" / "crono_memory_guard.c"
MEMORY_GUARD_BINARY = PROJECT_ROOT / ".crono-native" / "crono-memory-guard"
NATIVE_MCP_DIR = PROJECT_ROOT / "mcp-crono-matrix"
NATIVE_MCP_ENTRY = NATIVE_MCP_DIR / "native_server.mjs"
NATIVE_MCP_WORKSPACE = NATIVE_MCP_DIR / "workspace"
ASAEL_GATEWAY_ENTRY = NATIVE_MCP_DIR / "gateway.mjs"
BUNDLED_MCP_AVAILABLE = NATIVE_MCP_ENTRY.is_file()
LLAMA_PLAYWRIGHT_SCRIPT = PROJECT_ROOT / "llama.cpp" / "tools" / "server" / "browser-playwright.mjs"
LLAMA_PLAYWRIGHT_MODULE = PROJECT_ROOT / "llama.cpp" / "tools" / "ui" / "node_modules" / "playwright" / "index.mjs"
AGENT_COMPAT_DIR = PROJECT_ROOT / ".crono-agent"
AGENT_ENV_FILE = AGENT_COMPAT_DIR / "agent-local.env.sh"
AGENT_METADATA_FILE = AGENT_COMPAT_DIR / "agent-local.json"
OPENCODE_GLOBAL_CONFIG = Path(os.environ.get(
    "CRONO_OPENCODE_CONFIG_FILE",
    Path.home() / ".config" / "opencode" / "opencode.jsonc",
)).expanduser()
OPENCODE_GLOBAL_STATE = AGENT_COMPAT_DIR / "opencode-global-state.json"
OPENCODE_GLOBAL_PROVIDER = "crono"

_RUNTIME_BUFFER_RE = re.compile(
    r":\s*(?P<device>.+?)\s+(?P<kind>KV|RS) buffer size\s*=\s*"
    r"(?P<size>[0-9]+(?:\.[0-9]+)?)\s+MiB",
    re.IGNORECASE,
)


def _runtime_memory_state(requested: str = "unknown") -> dict:
    return {
        "requested": requested,
        "kv": {
            "devices": {}, "total_mb": 0.0, "gpu_mb": 0.0,
            "cpu_mb": 0.0, "placement": "pending", "confirmed": None,
        },
        "rs": {
            "devices": {}, "total_mb": 0.0, "gpu_mb": 0.0,
            "cpu_mb": 0.0, "placement": "pending", "confirmed": None,
        },
    }
MEMORY_DIR = Path(os.environ.get("CRONO_MEMORY_DIR", NATIVE_MCP_DIR / "memory")).expanduser()
SNN_DIR = MEMORY_DIR / "snn"
SNN_BINARY = NATIVE_MCP_DIR / "snn_ai" / "snn_ai"
SNN_DNA = SNN_DIR / "dna.bin"
SNN_ENABLED_FILE = SNN_DIR / "enabled.json"
LOCAL_AGENT_INSTRUCTIONS = (
    "You are the language model driving a local OpenAI-compatible coding agent. "
    "The client owns the tools and skills; use the tools actually present "
    "in the request as the source of truth. Work directly in the user's workspace "
    "and keep going until the requested outcome is handled.\n\n"
    "Use the native tool set with the same discipline as a mature coding agent: "
    "read and search files before editing; prefer fast file search for discovery; "
    "use the shell for commands and tests; use the native edit or apply-patch "
    "tool for changes; use task/subagent tools for genuinely independent complex "
    "work; use the skill tool when a listed skill matches the task; and keep a "
    "todo list for multi-step work when that tool is available. Do not recreate "
    "a native tool through another tool and do not claim a tool was used when it "
    "was not available.\n\n"
    "For current or externally verifiable information, use an available web "
    "search or web fetch tool immediately when the user asks for research. If a "
    "web tool is not present, say so clearly and use an allowed local fallback "
    "only when it can provide real evidence. Never present hosted OpenAI Deep "
    "Research as available on this local model. Report sources and uncertainty "
    "when research is performed.\n\n"
    "Inspect the existing codebase first, preserve unrelated user changes, make "
    "safe scoped edits, verify risky changes, and report concrete results."
)


def _resolve_node_runtime():
    """Return a runnable Node.js binary, its version and diagnostics.

    ``shutil.which`` only proves that a file is present in PATH.  On systems
    with a partial package upgrade (or a broken custom Node installation) the
    executable can still fail before the JavaScript process starts.  The
    launcher must detect that explicitly instead of reporting a misleading
    evaluation exit code 127.

    ``CRONO_NODE_BIN`` is an intentional escape hatch for a user-managed
    Node.js installation (for example nvm, mise or a portable build).
    """
    configured = os.environ.get("CRONO_NODE_BIN", "").strip()
    requested = [configured] if configured else ["node", "nodejs"]
    candidates = []
    for item in requested:
        resolved = shutil.which(item) or item
        if resolved and resolved not in candidates:
            candidates.append(resolved)

    diagnostics = []
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            diagnostics.append(f"{candidate}: {exc}")
            continue
        output = "\n".join(
            part.strip() for part in (result.stdout or "", result.stderr or "") if part.strip()
        )
        if result.returncode == 0:
            version = (result.stdout or output or "desconhecida").strip().splitlines()[0]
            return candidate, version, ""
        diagnostics.append(
            f"{candidate}: codigo {result.returncode}"
            + (f" — {output[-500:]}" if output else "")
        )

    detail = "\n".join(diagnostics) or "nenhum executavel node/nodejs encontrado no PATH"
    return "", "", detail


def _node_unavailable_message(diagnostics: str) -> str:
    return (
        "Node.js indisponível para este recurso JavaScript. "
        "O executável foi encontrado, mas não conseguiu iniciar. "
        "Corrija a instalação do Node.js (em Arch/CachyOS, faça uma atualização "
        "completa do sistema: sudo pacman -Syu) ou defina CRONO_NODE_BIN para "
        "um Node válido; não crie symlink entre versões de libada.\n"
        f"Diagnóstico: {diagnostics}"
    )


def _default_llama_cpp_dir() -> str:
    server = Path(LLAMA_SERVER).expanduser()
    for parent in server.parents:
        if (parent / "CMakeLists.txt").is_file() and (parent / "ggml").is_dir():
            return str(parent)
    if server.parent.name in {"bin", "Release"}:
        return str(server.parent.parent)
    return str(server.parent)


class LauncherWebState:
    def __init__(
        self, models_dir: str = "", llama_cpp_dir: str = "",
        settings_file: str | Path = SETTINGS_FILE,
    ):
        self.lock = threading.RLock()
        self.settings_file = Path(settings_file).expanduser()
        settings = self._load_settings()
        self.models_dir = str(Path(
            models_dir or settings.get("models_dir") or MODELS_DIR
        ).expanduser())
        configured_llama_dir = str(
            llama_cpp_dir or settings.get("llama_cpp_dir") or _default_llama_cpp_dir()
        )
        self.llama_cpp_dir, self.llama_server, self.llama_fit_params = (
            self._resolve_llama_cpp(configured_llama_dir, require=False)
        )
        self.gguf_py_dir = configure_gguf_py_dir(self.llama_cpp_dir)
        self.mcp_native_default = str(settings.get("mcp_native", "y")).lower()
        if self.mcp_native_default not in {"y", "n"}:
            self.mcp_native_default = "y"
        if not BUNDLED_MCP_AVAILABLE:
            # A distribuição pública core-only não inclui o MCP Crono Matrix.
            # Não reutilize um "on" persistido por uma instalação completa.
            self.mcp_native_default = "n"
        agent_default = os.environ.get(
            "CRONO_AGENT_COMPAT", settings.get("agent_compat", "y")
        )
        self.agent_compat_default = str(agent_default).lower()
        if self.agent_compat_default not in {"y", "n"}:
            self.agent_compat_default = "y"
        agent_global = str(settings.get("agent_global", "n")).lower()
        self.agent_global_default = agent_global if agent_global in {"y", "n"} else "n"
        if self.agent_global_default == "y":
            # Global mode is a superset of the universal compatibility
            # profile.  Repair stale settings instead of allowing the UI to
            # advertise an active mode that _coerce_final later rejects.
            self.agent_compat_default = "y"
            self.mcp_native_default = "n"
        self.mcp_policy_default = str(settings.get("mcp_policy", "safe")).lower()
        if self.mcp_policy_default not in {"safe", "all"}:
            self.mcp_policy_default = "safe"
        self.auto_nvme_swap_default = str(
            settings.get("auto_nvme_swap", "y")
        ).lower()
        if self.auto_nvme_swap_default not in {"y", "n"}:
            self.auto_nvme_swap_default = "y"
        self.mcp_workspace_default = str(Path(
            settings.get("mcp_workspace") or NATIVE_MCP_WORKSPACE
        ).expanduser())
        try:
            self.mcp_snn_threads_default = int(settings.get("mcp_snn_threads", 2))
        except (TypeError, ValueError):
            self.mcp_snn_threads_default = 2
        if not 1 <= self.mcp_snn_threads_default <= 8:
            self.mcp_snn_threads_default = 2
        try:
            self.mcp_snn_steps_default = int(settings.get("mcp_snn_steps", 64))
        except (TypeError, ValueError):
            self.mcp_snn_steps_default = 64
        if not 1 <= self.mcp_snn_steps_default <= 256:
            self.mcp_snn_steps_default = 64
        try:
            self.mcp_repeat_limit_default = int(settings.get("mcp_repeat_limit", 2))
        except (TypeError, ValueError):
            self.mcp_repeat_limit_default = 2
        if not 2 <= self.mcp_repeat_limit_default <= 5:
            self.mcp_repeat_limit_default = 2
        self.snn_enabled_default = (
            self._load_snn_enabled(settings.get("snn_enabled", True))
            if BUNDLED_MCP_AVAILABLE else False
        )
        if BUNDLED_MCP_AVAILABLE:
            self._write_snn_enabled_file(self.snn_enabled_default, create_only=True)
        self.snn_telemetry_cache = None
        self.snn_telemetry_at = 0.0
        self.snn_probe_lock = threading.Lock()
        self.hardware = HardwareInfo()
        self.hardware_ready = False
        self.models = []
        self.models_by_id = {}
        # A origem do GGUF é mantida ao lado do arquivo, e não nas
        # configurações globais: isso permite mover/copiar uma árvore de
        # modelos sem misturar o estado de um modelo com outro. O mapa em
        # memória contém apenas o resultado da última verificação da sessão.
        self.model_update_lock = threading.RLock()
        self.model_update_thread = None
        self.model_update_results = {}
        self.model_update = {
            "state": "idle", "started_at": "", "checked_at": "",
            "total": 0, "completed": 0, "current": "", "error": "",
        }
        self.model_path = ""
        self.model_signature = None
        self.metadata_cache = {}
        self.meta = None
        self.opt = None
        self.params = {}
        # VRAM livre usada para construir ``params``. A interface desktop usa
        # este valor para não iniciar um perfil calculado enquanto outro
        # llama-server ocupava a GPU e que ficou obsoleto depois.
        self.profile_vram_free_mb = 0
        self.proc = None
        self.gateway_proc = None
        self.memory_guard_proc = None
        self.llama_scope_unit = ""
        self.agent_restore_mcp = False
        self.agent_info = {
            "enabled": False,
            "endpoint": "",
            "model": "",
            "catalog": "",
            "context_window": 0,
            "auto_compact_token_limit": 0,
            "modalities": [],
            "reasoning_enabled": False,
            "supports_reasoning_effort": False,
            "capabilities": {},
            "server_capabilities": {},
            "native_tools": [],
            "agent_env": "",
            "agent_metadata": "",
            "opencode_config": "",
            "global_enabled": False,
            "global_config": str(AGENT_ENV_FILE),
            "global_provider": "openai-compatible",
            "global_error": "",
        }
        self.process_state = "idle"
        self.process_error = ""
        self.runtime_effective = {
            "ready": False,
            "requested_context": 0,
            "context_window": 0,
            "total_slots": 0,
            "model_path": "",
            "props": {},
            "memory_buffers": _runtime_memory_state(),
        }
        self.memory_guard = {
            "enabled": True,
            "reserve_mb": SYSTEM_RAM_RESERVE_MB,
            "trigger_mb": RAM_CONTROL_TRIGGER_MB,
            "control_mode": "observacional C99 + kernel/NVMe",
            "scope_headroom_mb": 0,
            "scope_phase": "idle",
            "scope_unit": "",
            "memory_high_mb": 0,
            "available_mb": 0,
            "current_mb": 0,
            "pressure_count": 0,
            "last_action": "aguardando servidor",
            "error": "",
        }
        self.mcp_state = "disabled"
        self.mcp_tools = 0
        self.mcp_error = ""
        self.started_at = None
        self.exit_code = None
        self.last_command = ""
        self.logs = deque(maxlen=5000)
        self.log_seq = 0
        self.hf = HuggingFaceHub()
        self.hf_radar_lock = threading.RLock()
        self.hf_radar_watchlist = self._normalize_hf_watchlist(
            settings.get("hf_radar_watchlist", HF_RADAR_DEFAULT_WATCHLIST)
        )
        self.hf_radar_enabled = str(
            settings.get("hf_radar_enabled", "y")
        ).lower() not in {"0", "false", "n", "no", "off"}
        self.hf_radar_seen = self._normalize_hf_radar_records(
            settings.get("hf_radar_seen", {})
        )
        self.hf_radar_unread = self._normalize_hf_radar_records(
            settings.get("hf_radar_unread", {})
        )
        self.hf_radar_initialized = bool(settings.get("hf_radar_initialized", False))
        self.hf_radar_last_refresh = ""
        self.hf_radar_last_error = ""
        self.hf_radar_items = []
        self.hf_radar_refresh_monotonic = 0.0
        self.download = {
            "state": "idle", "downloaded": 0, "total": 0, "speed": 0,
            "filename": "", "error": "", "paths": [],
        }
        self.download_cancel = threading.Event()

    def _load_settings(self) -> dict:
        try:
            with self.settings_file.open(encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_settings(self) -> None:
        settings = self._load_settings()
        settings.update({
            "llama_cpp_dir": self.llama_cpp_dir,
            "models_dir": self.models_dir,
            "mcp_native": self.mcp_native_default,
            "agent_compat": (
                "y" if self.agent_global_default == "y"
                else self.agent_compat_default
            ),
            "agent_global": self.agent_global_default,
            "mcp_policy": self.mcp_policy_default,
            "auto_nvme_swap": self.auto_nvme_swap_default,
            "mcp_workspace": self.mcp_workspace_default,
             "mcp_snn_threads": self.mcp_snn_threads_default,
             "mcp_snn_steps": self.mcp_snn_steps_default,
             "mcp_repeat_limit": self.mcp_repeat_limit_default,
             "snn_enabled": "y" if self.snn_enabled_default else "n",
            "hf_radar_watchlist": self.hf_radar_watchlist,
            "hf_radar_enabled": "y" if self.hf_radar_enabled else "n",
            "hf_radar_seen": self.hf_radar_seen,
            "hf_radar_unread": self.hf_radar_unread,
            "hf_radar_initialized": self.hf_radar_initialized,
        })
        temporary = self.settings_file.with_name(self.settings_file.name + ".tmp")
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, self.settings_file)
        except OSError as exc:
            raise ValueError(f"Nao foi possivel salvar a configuracao: {exc}") from exc

    @staticmethod
    def _load_snn_enabled(fallback=True) -> bool:
        try:
            value = json.loads(SNN_ENABLED_FILE.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("enabled"), bool):
                return value["enabled"]
        except (OSError, ValueError, TypeError):
            pass
        if isinstance(fallback, str):
            return fallback.lower() in {"1", "true", "y", "yes", "on"}
        return bool(fallback)

    def _write_snn_enabled_file(self, enabled: bool, create_only=False) -> bool:
        SNN_DIR.mkdir(parents=True, exist_ok=True)
        if create_only and SNN_ENABLED_FILE.exists():
            return self._load_snn_enabled(enabled)
        temporary = SNN_ENABLED_FILE.with_name(SNN_ENABLED_FILE.name + ".tmp")
        temporary.write_text(
            json.dumps({"enabled": bool(enabled), "updated_at": time.time()}),
            encoding="utf-8",
        )
        os.replace(temporary, SNN_ENABLED_FILE)
        return bool(enabled)

    def snn_enabled(self) -> bool:
        with self.lock:
            return self._load_snn_enabled(self.snn_enabled_default)

    def set_snn_enabled(self, enabled: bool):
        value = self._write_snn_enabled_file(bool(enabled))
        with self.lock:
            self.snn_enabled_default = value
            settings = self._load_settings()
            settings["snn_enabled"] = value
            temporary = self.settings_file.with_name(self.settings_file.name + ".tmp")
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, self.settings_file)
            self.snn_telemetry_cache = None
            self.snn_telemetry_at = 0.0
        return self.snn_snapshot()

    @staticmethod
    def _parse_snn_output(output: str) -> dict:
        hardware = {}
        neurochemistry = {}
        section = None
        result = {
            "status": "online",
            "intent": "OBSERVATION",
            "reason": "",
            "will": "EXECUTE_PLAN",
            "confidence": 0.0,
            "neurochemistry": neurochemistry,
            "hardware": hardware,
            "raw_available": bool(output),
        }
        for line in output.splitlines():
            line = line.strip()
            if line == "NEUROCHEMICAL_MAP_START":
                section = neurochemistry
                continue
            if line == "NEUROCHEMICAL_MAP_END":
                section = None
                continue
            if line == "HARDWARE_TELEMETRY_START":
                section = hardware
                continue
            if line == "HARDWARE_TELEMETRY_END":
                section = None
                continue
            if section is not None:
                for item in line.split(","):
                    if ":" not in item:
                        continue
                    key, value = item.split(":", 1)
                    try:
                        number = float(value)
                        section[key] = int(number) if number.is_integer() else number
                    except ValueError:
                        section[key] = value
            elif line.startswith("INTENT:"):
                result["intent"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
            elif line.startswith("WILL:"):
                result["will"] = line.split(":", 1)[1].strip()
            elif line.startswith("Decision confidence:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return result

    def snn_snapshot(self) -> dict:
        enabled = self.snn_enabled()
        if not enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "intent": "OFF",
                "will": "IDLE",
                "confidence": 0.0,
                "neurochemistry": {},
                "hardware": {},
                "message": "Núcleo SNN desativado pelo operador.",
            }

        now = time.monotonic()
        with self.lock:
            if self.snn_telemetry_cache and now - self.snn_telemetry_at < 5:
                return dict(self.snn_telemetry_cache)

        if not SNN_BINARY.is_file():
            snapshot = {
                "enabled": True,
                "status": "unavailable",
                "intent": "OFFLINE",
                "will": "IDLE",
                "confidence": 0.0,
                "neurochemistry": {},
                "hardware": {},
                "message": f"Binário SNN não encontrado: {SNN_BINARY}",
            }
        elif not SNN_DNA.is_file():
            snapshot = {
                "enabled": True,
                "status": "waiting",
                "intent": "WAITING_DNA",
                "will": "IDLE",
                "confidence": 0.0,
                "neurochemistry": {},
                "hardware": {},
                "message": "DNA neural ainda não foi criado.",
            }
        else:
            with self.snn_probe_lock:
                try:
                    completed = subprocess.run(
                        [
                            str(SNN_BINARY), "0", "0.5", "0", "10",
                            str(max(16, min(self.mcp_snn_steps_default, 64))),
                            "--dna-load", str(SNN_DNA), "--sample-spikes",
                        ],
                        cwd=str(SNN_BINARY.parent),
                        capture_output=True,
                        text=True,
                        timeout=20,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": str(self.mcp_snn_threads_default),
                            "OMP_DYNAMIC": "FALSE",
                        },
                        check=False,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(completed.stderr.strip() or f"exit {completed.returncode}")
                    snapshot = {
                        "enabled": True,
                        **self._parse_snn_output(completed.stdout),
                        "message": "Telemetria lida do DNA neural persistente.",
                    }
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    snapshot = {
                        "enabled": True,
                        "status": "error",
                        "intent": "ERROR",
                        "will": "IDLE",
                        "confidence": 0.0,
                        "neurochemistry": {},
                        "hardware": {},
                        "message": f"Falha ao ler SNN: {exc}",
                    }

        with self.lock:
            self.snn_telemetry_cache = snapshot
            self.snn_telemetry_at = time.monotonic()
        return dict(snapshot)

    @staticmethod
    def _llama_build_usable(server: Path) -> tuple[bool, str]:
        """Confirm that a build can start and includes its matching fit tool.

        Merely checking executable bits is insufficient for CMake build trees:
        an executable may remain after its shared libraries were moved or
        deleted.  Selecting such a tree made the UI calculate with one build
        and then fail at launch with exit 127.  ``--version`` does not load a
        model, but it does exercise the dynamic loader and CUDA backend setup.
        """
        fit_params = server.with_name("llama-fit-params")
        for binary in (server, fit_params):
            if not binary.is_file() or not os.access(binary, os.X_OK):
                return False, f"ausente ou não executável: {binary}"
            try:
                completed = subprocess.run(
                    [str(binary), "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"{binary}: {exc}"
            if completed.returncode != 0:
                detail = " ".join(completed.stdout.strip().splitlines()[-2:])
                return False, (
                    f"{binary}: código {completed.returncode}"
                    + (f" — {detail}" if detail else "")
                )
        return True, ""

    @staticmethod
    def _resolve_llama_cpp(value: str, require: bool = True) -> tuple[str, str, str]:
        selected = Path(value).expanduser().resolve()
        if selected.is_file() or selected.name == "llama-server":
            candidates = [selected]
            display_dir = selected.parent
        else:
            candidates = [
                selected / "bin" / "llama-server",
                selected / "bin" / "Release" / "llama-server",
                selected / "build-crono" / "bin" / "llama-server",
                selected / "build-rtx3060" / "bin" / "llama-server",
                selected / "build" / "bin" / "llama-server",
                selected / "build" / "bin" / "Release" / "llama-server",
                selected / "llama-server",
            ]
            display_dir = selected
        existing = [path for path in candidates if path.is_file()]
        rejected: list[str] = []
        server = None
        for candidate in existing:
            usable, detail = LauncherWebState._llama_build_usable(candidate)
            if usable:
                server = candidate
                break
            rejected.append(detail)
        if server is None:
            server = existing[0] if existing else candidates[0]
        fit_params = server.with_name("llama-fit-params")
        if require:
            if not selected.exists():
                raise ValueError(f"Caminho do llama.cpp nao encontrado: {selected}")
            if not server.is_file():
                raise ValueError(
                    f"llama-server nao encontrado em {selected}. "
                    "Compile o llama.cpp ou informe o executavel llama-server."
                )
            if not os.access(server, os.X_OK):
                raise ValueError(f"llama-server sem permissao de execucao: {server}")
            if not fit_params.is_file() or not os.access(fit_params, os.X_OK):
                raise ValueError(f"llama-fit-params nao encontrado ou nao executavel: {fit_params}")
            usable, detail = LauncherWebState._llama_build_usable(server)
            if not usable:
                diagnostics = "; ".join(item for item in rejected if item)
                raise ValueError(
                    "Nenhuma compilação utilizável do llama.cpp foi encontrada. "
                    + (diagnostics or detail)
                )
        return str(display_dir), str(server), str(fit_params)

    def _new_optimal_params(self, meta: ModelMetadata) -> OptimalParams:
        return OptimalParams(
            self.hardware, meta,
            llama_server=self.llama_server,
            llama_fit_params=self.llama_fit_params,
        )

    def configure_paths(self, llama_cpp_dir: str, models_dir: str):
        with self.lock:
            if self.is_running():
                raise ValueError("Encerre o servidor antes de alterar os caminhos.")
        llama_dir, server, fit_params = self._resolve_llama_cpp(llama_cpp_dir)
        gguf_py_dir = configure_gguf_py_dir(llama_dir)
        models_root = Path(models_dir).expanduser().resolve()
        if not models_root.is_dir():
            raise ValueError(f"Diretorio de modelos nao encontrado: {models_root}")
        with self.lock:
            self.llama_cpp_dir = llama_dir
            self.llama_server = server
            self.llama_fit_params = fit_params
            self.gguf_py_dir = gguf_py_dir
            self.models_dir = str(models_root)
            if self.model_path and Path(self.model_path).parent != models_root \
                    and models_root not in Path(self.model_path).parents:
                self.model_path = ""
                self.model_signature = None
                self.meta = None
                self.opt = None
                self.params = {}
        self.scan_models(str(models_root))
        self._save_settings()
        return self.configuration_snapshot()

    def configuration_snapshot(self):
        with self.lock:
            return {
                "llama_cpp_dir": self.llama_cpp_dir,
                "llama_server": self.llama_server,
                "llama_fit_params": self.llama_fit_params,
                "gguf_py_dir": self.gguf_py_dir,
                "models_dir": self.models_dir,
            }

    def refresh_hardware(self):
        hw = HardwareInfo()
        hw.detect()
        with self.lock:
            self.hardware = hw
            self.hardware_ready = True
            if self.meta and not self.is_running():
                self.opt = self._new_optimal_params(self.meta)
                self.opt.vision_enabled = self.params.get("omni") == "y"
                self.opt.calculate()
                # A atualização do hardware não pode apagar alterações feitas
                # manualmente na interface. O perfil automático é aplicado
                # somente na seleção de um novo modelo; aqui atualizamos apenas
                # o objeto de cálculo usado pelos motivos/diagnóstico.
        return self.hardware_snapshot()

    def hardware_snapshot(self):
        with self.lock:
            hw = self.hardware
            return {
                "ready": self.hardware_ready,
                "cpu_model": hw.cpu_model,
                "cpu_cores": hw.cpu_cores,
                "cpu_threads": hw.cpu_threads,
                "cpu_temp": hw.cpu_temp,
                "ram_total_gb": hw.ram_total_gb,
                "ram_avail_gb": hw.ram_avail_gb,
                "ram_avail_mb": hw.ram_avail_mb,
                "swap_total_mb": hw.swap_total_mb,
                "swap_free_mb": hw.swap_free_mb,
                "swap_used_mb": hw.swap_used_mb,
                "swap_zram_total_mb": hw.swap_zram_total_mb,
                "swap_zram_used_mb": hw.swap_zram_used_mb,
                "swap_zram_priority": hw.swap_zram_priority,
                "swap_nvme_total_mb": hw.swap_nvme_total_mb,
                "swap_nvme_used_mb": hw.swap_nvme_used_mb,
                "swap_nvme_priority": hw.swap_nvme_priority,
                "swap_nvme_path": hw.swap_nvme_path,
                "swap_nvme_active": hw.swap_nvme_active,
                "swap_nvme_preferred": hw.swap_nvme_preferred,
                "gpu_detected": hw.gpu_detected,
                "gpu_model": hw.gpu_model,
                "gpu_vram_mb": hw.gpu_vram_mb,
                "gpu_vram_free_mb": hw.gpu_vram_free_mb,
                "gpu_vram_gb": hw.gpu_vram_gb,
                "gpu_vram_free_gb": hw.gpu_vram_free_gb,
                "gpu_temp": hw.gpu_temp,
                "gpu_driver": hw.gpu_driver,
                "gpu_cuda": hw.gpu_cuda,
                "storage_type": hw.storage_type,
                "disk_free_gb": hw.disk_free_gb,
                "swap_recommended_gib": (
                    self.opt.swap_recommended_gib if self.opt else 0
                ),
                "swap_plan_reason": (
                    self.opt.swap_plan_reason if self.opt else
                    "selecione um modelo para calcular o swap dinâmico"
                ),
                "auto_nvme_swap": self.auto_nvme_swap_default,
                "ram_reserve_mb": SYSTEM_RAM_RESERVE_MB,
                "memory_guard": dict(self.memory_guard),
            }

    @staticmethod
    def _mem_available_mb() -> int:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0

    @staticmethod
    def _process_memory_control_dir(pid: int) -> Path | None:
        try:
            rows = Path(f"/proc/{int(pid)}/cgroup").read_text(
                encoding="utf-8"
            ).splitlines()
            relative = next(
                row.split(":", 2)[2] for row in rows if row.startswith("0::")
            )
        except (OSError, ValueError, StopIteration, IndexError):
            return None
        if "/crono-llama-" not in relative or ".scope" not in relative:
            return None
        cgroup_root = Path("/sys/fs/cgroup").resolve()
        candidate = (cgroup_root / relative.lstrip("/")).resolve()
        try:
            candidate.relative_to(cgroup_root)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _ensure_native_memory_guard() -> Path:
        if not MEMORY_GUARD_SOURCE.is_file():
            raise OSError(f"fonte C99 não encontrada: {MEMORY_GUARD_SOURCE}")
        if (
            MEMORY_GUARD_BINARY.is_file()
            and os.access(MEMORY_GUARD_BINARY, os.X_OK)
            and MEMORY_GUARD_BINARY.stat().st_mtime_ns
                >= MEMORY_GUARD_SOURCE.stat().st_mtime_ns
        ):
            return MEMORY_GUARD_BINARY
        compiler = shutil.which("cc") or shutil.which("gcc")
        if not compiler:
            raise OSError("compilador C99 não encontrado (cc/gcc)")
        MEMORY_GUARD_BINARY.parent.mkdir(parents=True, exist_ok=True)
        temporary = MEMORY_GUARD_BINARY.with_suffix(".tmp")
        completed = subprocess.run(
            [
                compiler, "-std=c99", "-O3", "-march=native", "-flto",
                "-Wall", "-Wextra", "-Werror", "-D_FORTIFY_SOURCE=2",
                str(MEMORY_GUARD_SOURCE), "-o", str(temporary),
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, check=False,
        )
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise OSError(
                "falha ao compilar monitor C99: " + completed.stdout.strip()
            )
        temporary.chmod(0o755)
        os.replace(temporary, MEMORY_GUARD_BINARY)
        return MEMORY_GUARD_BINARY

    def _start_memory_guard(self, llama_proc) -> None:
        if not self._process_memory_control_dir(llama_proc.pid):
            with self.lock:
                self.memory_guard_proc = None
                self.memory_guard["error"] = "cgroup dedicado não confirmado"
                self.memory_guard["last_action"] = (
                    "telemetria C99 não iniciada fora de scope isolado"
                )
            self._append_log(
                "[RAM guard: scope isolado não confirmado; o launcher e o "
                "terminal não serão limitados]\n",
                "warning",
            )
            return
        try:
            binary = self._ensure_native_memory_guard()
            guard = subprocess.Popen(
                [
                    str(binary), "--pid", str(llama_proc.pid),
                    "--warning-mb", str(RAM_CONTROL_TRIGGER_MB),
                    "--interval-ms", "500",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
            with self.lock:
                self.memory_guard_proc = guard
                self.memory_guard.update({
                    "available_mb": self._mem_available_mb(),
                    "current_mb": 0,
                    "pressure_count": 0,
                    "last_action": "monitor C99 iniciando",
                    "error": "",
                })
            threading.Thread(
                target=self._read_memory_guard_output,
                args=(guard, llama_proc), daemon=True,
            ).start()
        except OSError as exc:
            with self.lock:
                self.memory_guard_proc = None
                self.memory_guard["error"] = str(exc)
                self.memory_guard["last_action"] = "guard C99 indisponível"
            self._append_log(f"[RAM guard C99 indisponível: {exc}]\n", "error")

    def _scoped_llama_command(
        self, command: list[str], available_mb: int, load_mode: str,
    ) -> tuple[list[str], str, int, int]:
        """Place llama-server in an observable scope without a memory ceiling."""
        systemd_run = shutil.which("systemd-run")
        unified_cgroup = Path("/sys/fs/cgroup/cgroup.controllers").is_file()
        _ = available_mb, load_mode
        if not systemd_run or not unified_cgroup:
            return command, "", 0, 0
        unit = f"crono-llama-{os.getpid()}-{time.monotonic_ns() & 0xFFFFFF:x}"
        wrapped = [
            systemd_run, "--user", "--scope", "--quiet",
            f"--unit={unit}",
        ]
        wrapped.extend(["--", *command])
        return wrapped, unit, 0, 0

    def _tune_scope_for_inference(self, proc) -> None:
        """Remove stale limits and publish the post-load observation phase."""
        if not getattr(self, "llama_scope_unit", ""):
            return
        cgroup_dir = self._process_memory_control_dir(proc.pid)
        if not cgroup_dir:
            return
        high_file = cgroup_dir / "memory.high"
        try:
            old_high_text = high_file.read_text(encoding="utf-8").strip()
            available_mb = self._mem_available_mb()
            if old_high_text != "max":
                high_file.write_text("max\n", encoding="utf-8")
            with self.lock:
                self.memory_guard.update({
                    "scope_phase": "inference",
                    "scope_headroom_mb": 0,
                    "memory_high_mb": 0,
                })
            self._append_log(
                f"[cgroup inferência: MemoryHigh {old_high_text} -> max; "
                f"RAM disponível {available_mb} MiB; paginação do kernel]\n",
                "success",
            )
        except (OSError, ValueError) as exc:
            self._append_log(
                f"[ajuste MemoryHigh pós-carga não aplicado: {exc}]\n",
                "warning",
            )

    @staticmethod
    def _wait_for_process_scope(proc, unit: str, timeout: float = 5.0) -> bool:
        if not unit:
            return False
        expected = f"/{unit}.scope"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                cgroup = Path(f"/proc/{proc.pid}/cgroup").read_text(encoding="utf-8")
                if expected in cgroup:
                    return True
            except OSError:
                pass
            time.sleep(0.02)
        return False

    def _read_memory_guard_output(self, guard, llama_proc) -> None:
        try:
            if guard.stdout:
                for line in guard.stdout:
                    text = line.strip()
                    if text.startswith("READY"):
                        with self.lock:
                            self.memory_guard["last_action"] = (
                                f"monitor C99 observacional ativo — aviso abaixo de "
                                f"{RAM_CONTROL_TRIGGER_MB} MiB"
                            )
                        self._append_log(f"[RAM guard C99: {text}]\n", "success")
                    elif text.startswith("STATUS"):
                        status = re.search(
                            r"available=(\d+) current=(\d+) high=(max|\d+)", text
                        )
                        if status:
                            available_mb = int(status.group(1))
                            current_mb = int(status.group(2))
                            high_text = status.group(3)
                            with self.lock:
                                self.memory_guard["available_mb"] = available_mb
                                self.memory_guard["current_mb"] = current_mb
                                self.memory_guard["memory_high_mb"] = (
                                    0 if high_text == "max" else int(high_text)
                                )
                                self.memory_guard["last_action"] = (
                                    f"saudável — {available_mb} MiB disponíveis"
                                )
                                self.memory_guard["error"] = ""
                    elif text.startswith("PRESSURE"):
                        pressure = re.search(
                            r"available=(\d+) current=(\d+) high=(max|\d+)", text
                        )
                        if pressure:
                            available_mb = int(pressure.group(1))
                            current_mb = int(pressure.group(2))
                            high_text = pressure.group(3)
                            with self.lock:
                                self.memory_guard["available_mb"] = available_mb
                                self.memory_guard["current_mb"] = current_mb
                                self.memory_guard["memory_high_mb"] = (
                                    0 if high_text == "max" else int(high_text)
                                )
                                self.memory_guard["pressure_count"] += 1
                                pressure_count = self.memory_guard["pressure_count"]
                                self.memory_guard["last_action"] = (
                                    f"pressão observada — uso={current_mb} MiB; "
                                    f"MemoryHigh={high_text}; RAM={available_mb} MiB; "
                                    "kernel/ZRAM gerenciam a paginação"
                                )
                            if pressure_count == 1 or pressure_count % 20 == 0:
                                self._append_log(
                                    f"[RAM monitor C99: {text}]\n", "warning"
                                )
                    elif text.startswith("ERROR"):
                        with self.lock:
                            self.memory_guard["error"] = text
                        self._append_log(f"[RAM guard C99: {text}]\n", "error")
        finally:
            guard.wait()
            with self.lock:
                if guard is self.memory_guard_proc:
                    self.memory_guard_proc = None
                    if llama_proc is self.proc and llama_proc.poll() is None:
                        self.memory_guard["error"] = (
                            f"monitor C99 encerrou com código {guard.returncode}"
                        )

    def _stop_memory_guard(self) -> None:
        with self.lock:
            guard = self.memory_guard_proc
            self.memory_guard_proc = None
            self.memory_guard.update({
                "current_mb": 0,
                "scope_unit": "",
                "scope_phase": "idle",
                "memory_high_mb": 0,
                "scope_headroom_mb": 0,
                "last_action": "aguardando servidor",
            })
        if guard and guard.poll() is None:
            try:
                os.killpg(os.getpgid(guard.pid), signal.SIGTERM)
                guard.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(guard.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def configure_nvme_swap(self, action: str, size_gib: int = 0):
        """Create/grow or remove only the dedicated Crono NVMe swapfile."""
        action = str(action).strip().lower()
        if action not in {"create", "remove"}:
            raise ValueError("Ação de swap inválida.")
        with self.lock:
            if self.is_running():
                raise ValueError("Encerre o llama-server antes de alterar o swap NVMe.")
            recommended = self.opt.swap_recommended_gib if self.opt else 0
        if action == "create":
            try:
                requested = int(size_gib or recommended)
            except (TypeError, ValueError):
                requested = 0
            if requested <= 0:
                raise ValueError("Selecione um modelo que necessite swap ou informe um tamanho válido.")
            requested = max(8, min(requested, 64))
            command = ["pkexec", str(NVME_SWAP_MANAGER), "create", str(requested)]
        else:
            command = ["pkexec", str(NVME_SWAP_MANAGER), "remove"]
        if not NVME_SWAP_MANAGER.is_file() or not os.access(NVME_SWAP_MANAGER, os.X_OK):
            raise ValueError(f"Gerenciador de swap não encontrado: {NVME_SWAP_MANAGER}")
        try:
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=300, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("A autenticação/alteração do swap excedeu 5 minutos.") from exc
        except OSError as exc:
            raise ValueError(f"Não foi possível executar o gerenciador de swap: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stdout.strip() or f"código {completed.returncode}"
            raise ValueError(f"Falha ao alterar swap NVMe: {detail}")
        return self.refresh_hardware()

    def ensure_dynamic_swap(self):
        """Grow and prioritize swap before a profile that depends on it."""
        with self.lock:
            if self.auto_nvme_swap_default != "y" or not self.opt:
                return self.hardware_snapshot()
            recommended = int(self.opt.swap_recommended_gib)
            current = int((self.hardware.swap_nvme_total_mb + 1023) // 1024)
            priority_unsafe = bool(
                recommended > 0
                and self.hardware.swap_nvme_active
                and not self.hardware.swap_nvme_preferred
            )
        if recommended > current or priority_unsafe:
            return self.configure_nvme_swap(
                "create", max(recommended, current, 8)
            )
        return self.hardware_snapshot()

    def set_auto_nvme_swap(self, enabled: bool):
        with self.lock:
            self.auto_nvme_swap_default = "y" if enabled else "n"
        self._save_settings()
        return self.hardware_snapshot()

    @staticmethod
    def _model_origin_path(path: str | Path) -> Path:
        return Path(f"{Path(path)}.crono-origin.json")

    @classmethod
    def _read_model_origin(cls, path: str | Path) -> dict:
        try:
            value = json.loads(cls._model_origin_path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        if not isinstance(value, dict):
            return {}
        repo_id = str(value.get("repo_id") or "").strip()
        filename = str(value.get("filename") or "").replace("\\", "/").lstrip("/")
        if "/" not in repo_id or not filename or len(repo_id) > 256 or len(filename) > 512:
            return {}
        value["repo_id"] = repo_id
        value["filename"] = filename
        value["revision"] = str(value.get("revision") or "main")
        return value

    @classmethod
    def _write_model_origin(cls, path: str | Path, origin: dict) -> None:
        """Atomically persist the remote identity of one GGUF shard."""
        target = cls._model_origin_path(path)
        payload = {
            "schema": 1,
            "source": str(origin.get("source") or "crono-huggingface"),
            "repo_id": str(origin.get("repo_id") or ""),
            "revision": str(origin.get("revision") or "main"),
            "filename": str(origin.get("filename") or Path(path).name),
            "commit": str(origin.get("commit") or ""),
            "remote_size": int(origin.get("remote_size") or 0),
            "remote_sha256": str(origin.get("remote_sha256") or "").lower(),
            "remote_last_modified": str(origin.get("remote_last_modified") or ""),
            "downloaded_size": int(origin.get("downloaded_size") or 0),
            "downloaded_sha256": str(origin.get("downloaded_sha256") or "").lower(),
            "downloaded_at": str(origin.get("downloaded_at") or ""),
            "last_checked_at": str(origin.get("last_checked_at") or ""),
        }
        if "/" not in payload["repo_id"] or not payload["filename"]:
            raise ValueError("origem remota de modelo invalida")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

    @staticmethod
    def _sha256_file(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _utc_now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _model_update_result(self, path: str | Path) -> dict:
        value = str(Path(path).resolve())
        with self.model_update_lock:
            current = self.model_update_results.get(value)
            if current:
                return dict(current)
        return {
            "state": "unchecked", "label": "NÃO VERIFICADO",
            "detail": "Origem do GGUF ainda não foi verificada.",
            "repo_id": "", "checked_at": "",
        }

    def model_update_snapshot(self) -> dict:
        with self.model_update_lock:
            return {
                **self.model_update,
                "results": {
                    key: dict(value) for key, value in self.model_update_results.items()
                },
                "running": bool(
                    self.model_update_thread and self.model_update_thread.is_alive()
                ),
            }

    def _set_model_update_result(self, path: str | Path, **values) -> None:
        key = str(Path(path).resolve())
        with self.model_update_lock:
            current = dict(self.model_update_results.get(key, {}))
            current.update(values)
            self.model_update_results[key] = current

    def _verify_one_model_update(self, path: str) -> dict:
        model_path = Path(path).resolve()
        local_size = model_path.stat().st_size
        manifest = self._read_model_origin(model_path)
        source = "manifest"
        candidates = []
        if manifest:
            repo_id = manifest["repo_id"]
            filename = manifest["filename"]
            revision = manifest.get("revision", "main")
        else:
            source = "auto-name"
            candidates = self.hf.resolve_candidates(model_path.name)
            if len(candidates) != 1:
                if not candidates:
                    return {
                        "state": "unassociated", "label": "SEM ORIGEM",
                        "detail": "Nenhum repositório GGUF foi associado ao nome exato.",
                        "repo_id": "", "checked_at": self._utc_now(),
                    }
                names = ["/".join(item) for item in candidates[:3]]
                return {
                    "state": "ambiguous", "label": "FONTES AMBÍGUAS",
                    "detail": "Mais de uma origem exata: " + ", ".join(names),
                    "repo_id": "", "candidates": names, "checked_at": self._utc_now(),
                }
            repo_id = "/".join(candidates[0])
            filename = model_path.name
            revision = "main"

        user, repo = repo_id.split("/", 1)
        info = self.hf.model_info(user, repo, revision)
        siblings = {
            str(item.get("rfilename") or "").lower(): item
            for item in info.get("siblings", [])
            if isinstance(item, dict)
        }
        item = siblings.get(filename.lower())
        checked_at = self._utc_now()
        if not item:
            return {
                "state": "missing", "label": "REMOTO AUSENTE",
                "detail": f"{filename} não existe mais em {repo_id}@{revision}.",
                "repo_id": repo_id, "checked_at": checked_at,
            }

        remote_size = int(item.get("size") or 0)
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        remote_sha = str(
            lfs.get("sha256") or item.get("sha256") or ""
        ).lower()
        old_downloaded_sha = str(manifest.get("downloaded_sha256") or "").lower()
        old_downloaded_size = int(manifest.get("downloaded_size") or 0)
        local_sha = ""
        if remote_sha and local_size == remote_size:
            local_sha = self._sha256_file(model_path)

        if remote_size and local_size != remote_size:
            result_state = "outdated"
            label = "ATUALIZAÇÃO DISPONÍVEL"
            detail = f"Tamanho local {local_size:,} B; remoto {remote_size:,} B."
        elif remote_sha and local_sha == remote_sha:
            result_state = "current"
            label = "ATUAL"
            detail = f"SHA-256 confirmado em {repo_id}."
        elif remote_sha:
            # A diferença só é chamada de atualização quando o arquivo local
            # ainda corresponde ao hash que foi baixado/associado. Assim, um
            # GGUF alterado manualmente não é apresentado como atualização.
            expected_local = old_downloaded_sha or str(manifest.get("remote_sha256") or "").lower()
            if expected_local and local_sha == expected_local and remote_sha != expected_local:
                result_state = "outdated"
                label = "ATUALIZAÇÃO DISPONÍVEL"
                detail = "O SHA-256 remoto mudou desde a origem registrada."
            else:
                result_state = "different"
                label = "ARQUIVO DIVERGENTE"
                detail = "O SHA-256 local não corresponde à origem remota registrada."
        else:
            result_state = "unverified"
            label = "NÃO VERIFICADO"
            detail = "O Hugging Face não forneceu SHA-256; apenas o tamanho foi comparado."

        origin = dict(manifest)
        origin.update({
            "source": source if not manifest else manifest.get("source", "crono-huggingface"),
            "repo_id": repo_id, "revision": revision, "filename": filename,
            "commit": str(info.get("sha") or info.get("commit") or ""),
            "remote_size": remote_size,
            "remote_sha256": remote_sha,
            "remote_last_modified": str(info.get("lastModified") or ""),
            "downloaded_size": old_downloaded_size or local_size,
            "downloaded_sha256": old_downloaded_sha or local_sha,
            "last_checked_at": checked_at,
        })
        try:
            self._write_model_origin(model_path, origin)
        except (OSError, ValueError):
            # A falha ao persistir a origem não invalida a comparação desta
            # execução; ela será mostrada como resultado, mas será reavaliada
            # na próxima sessão.
            detail += " Origem não pôde ser persistida ao lado do arquivo."
        return {
            "state": result_state, "label": label, "detail": detail,
            "repo_id": repo_id, "filename": filename, "checked_at": checked_at,
            "remote_size": remote_size, "remote_sha256": remote_sha,
        }

    def _verify_model_updates_worker(self, paths: list[str]) -> None:
        error = ""
        try:
            for index, path in enumerate(paths, 1):
                with self.model_update_lock:
                    self.model_update.update({
                        "current": Path(path).name, "completed": index - 1,
                    })
                try:
                    result = self._verify_one_model_update(path)
                except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
                    result = {
                        "state": "unavailable", "label": "NÃO VERIFICADO",
                        "detail": f"Falha ao consultar o Hugging Face: {exc}",
                        "repo_id": "", "checked_at": self._utc_now(),
                    }
                self._set_model_update_result(path, **result)
                with self.model_update_lock:
                    self.model_update["completed"] = index
        except Exception as exc:
            error = str(exc)
        finally:
            with self.model_update_lock:
                self.model_update.update({
                    "state": "error" if error else "done", "current": "",
                    "checked_at": self._utc_now(), "error": error,
                })
            try:
                self.scan_models()
            except ValueError:
                pass

    def start_model_update_check(self):
        with self.model_update_lock:
            if self.model_update_thread and self.model_update_thread.is_alive():
                raise ValueError("A verificação de atualizações já está em andamento.")
        with self.lock:
            paths = [str(item["path"]) for item in self.models]
        if not paths:
            paths = [str(item["path"]) for item in self.scan_models()]
        with self.model_update_lock:
            if self.model_update_thread and self.model_update_thread.is_alive():
                raise ValueError("A verificação de atualizações já está em andamento.")
            self.model_update = {
                "state": "running", "started_at": self._utc_now(),
                "checked_at": "", "total": len(paths), "completed": 0,
                "current": "", "error": "",
            }
            for path in paths:
                self.model_update_results[str(Path(path).resolve())] = {
                    "state": "checking", "label": "VERIFICANDO…",
                    "detail": "Consultando a origem no Hugging Face.",
                    "repo_id": "", "checked_at": "",
                }
            self.model_update_thread = threading.Thread(
                target=self._verify_model_updates_worker, args=(paths,),
                name="crono-model-update-check", daemon=True,
            )
            self.model_update_thread.start()
            return self.models_snapshot()

    def scan_models(self, directory: str = ""):
        root = Path(directory or self.models_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Diretorio de modelos nao encontrado: {root}")
        candidates = sorted(set(root.rglob("*.gguf")))
        rows = []
        for path in candidates:
            value = str(path)
            if _is_auxiliary_gguf(value) or _is_secondary_shard(value):
                continue
            try:
                size = _gguf_total_size(value)
            except (OSError, ValueError):
                continue
            model_id = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            rows.append({
                "id": model_id,
                "path": value,
                "name": path.name,
                "relative": str(path.relative_to(root)),
                "size": size,
                "size_gb": size / 1073741824,
                "selected": value == self.model_path,
                "update": self._model_update_result(value),
            })
        with self.lock:
            self.models_dir = str(root)
            self.models = rows
            self.models_by_id = {row["id"]: row for row in rows}
        return self.models_snapshot()

    def models_snapshot(self):
        with self.lock:
            return [
                dict(
                    row,
                    selected=row["path"] == self.model_path,
                    update=dict(row.get("update") or self._model_update_result(row["path"])),
                )
                for row in self.models
            ]

    def select_model(self, model_id: str):
        with self.lock:
            row = self.models_by_id.get(model_id)
        if not row:
            raise ValueError("Modelo nao encontrado. Atualize a lista.")
        path = Path(row["path"]).resolve()
        root = Path(self.models_dir).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Caminho de modelo invalido.")
        path_value = str(path)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        with self.lock:
            if (
                self.model_path == path_value
                and self.meta is not None
                and self.model_signature == signature
            ):
                return self.model_snapshot()
            cached = self.metadata_cache.get(path_value)
        if cached and cached[0] == signature:
            meta = cached[1]
        else:
            meta = ModelMetadata()
            meta.load(path_value)
        if not meta.meta_ok:
            detail = meta.metadata_error or "metadados essenciais ausentes"
            msg = f"GGUF incompativel ou incompleto: {detail}"
            error_kind = getattr(meta, "metadata_error_kind", "")
            if error_kind == "dependency":
                msg = f"Dependencia ausente do gguf-py. Execute: pip install -r requirements-web.txt"
            elif error_kind == "gguf_library":
                msg = (
                    "Biblioteca gguf-py nao encontrada no llama.cpp selecionado. "
                    "Aponte para um checkout completo ou execute "
                    "scripts/bootstrap_llama_cpp.sh."
                )
            raise ValueError(msg)
        if meta.quant_error:
            raise ValueError(meta.quant_error)
        with self.lock:
            self.metadata_cache[path_value] = (signature, meta)
            if not self.hardware_ready:
                self.refresh_hardware()
            opt = self._new_optimal_params(meta)
            opt.calculate()
            self.model_path = path_value
            self.model_signature = signature
            self.meta = meta
            self.opt = opt
            self.params = self._optimal_values(opt, meta)
            self.profile_vram_free_mb = int(self.hardware.gpu_vram_free_mb)

            # Detect MetaCognitiveHead from GGUF metadata
            try:
                import struct
                with open(str(path), 'rb') as f:
                    header = f.read(4096)
                if b'ornith.meta_head' in header:
                    self.params['metacognition'] = True
                    opt.mcp_config_reason = opt.mcp_config_reason or "MetaCognitiveHead detectada no GGUF"
            except Exception:
                self.params['metacognition'] = False
            self._apply_autotune_hit()
        return self.model_snapshot()

    def restore_optimal_profile(self):
        """Rebuild the selected model profile from the current hardware.

        Unlike ``refresh_hardware()``, this operation intentionally replaces
        the editable parameter snapshot. It is used by the explicit restore
        action and by the desktop stale-VRAM guard immediately before launch.
        """
        with self.lock:
            if self.is_running():
                raise ValueError("Encerre o servidor antes de restaurar o perfil.")
            if not self.meta or not self.model_path:
                raise ValueError("Selecione um modelo GGUF primeiro.")
            if not self.hardware_ready:
                raise ValueError("Atualize a telemetria do hardware primeiro.")
            opt = self._new_optimal_params(self.meta)
            opt.vision_enabled = self.params.get("omni") == "y"
            opt.calculate()
            self.opt = opt
            self.params = self._optimal_values(opt, self.meta)
            self.profile_vram_free_mb = int(self.hardware.gpu_vram_free_mb)
            self._apply_autotune_hit()
        return self.parameter_snapshot()

    def _apply_autotune_hit(self) -> None:
        """Apply only a validated cache entry for the active launch workload."""
        if not self.opt or not self.meta:
            return
        workload = {
            "mode": "interactive",
            "prompt_tokens": 32,
            "generation_tokens": 32,
            "ctx": self.params.get("ctx"),
            "batch": self.params.get("batch"),
            "ubatch": self.params.get("ubatch"),
            "parallel": self.params.get("parallel"),
        }
        sampler = {
            key: self.params.get(key)
            for key in ("seed", "temp", "top_k", "top_p", "min_p", "sampler_seq")
        }
        hit = self.opt.resolve_autotune(workload, sampler)
        if not hit:
            return
        config = hit.get("config", {})
        if not isinstance(config, dict):
            return
        self.opt.autotune_hit = hit
        for key in RUNTIME_CONFIG_KEYS:
            if key in config and key in self.params:
                self.params[key] = config[key]
        for key in (
            "ctx", "ngl", "cache_k", "cache_v", "batch", "ubatch",
            "threads", "threads_batch", "n_cpu_moe", "n_cpu_ffn",
            "load_mode", "tensor_read_lazy", "fit",
        ):
            if key in config and hasattr(self.opt, key):
                setattr(self.opt, key, config[key])
        metrics = hit.get("metrics", {})
        score = metrics.get("score", "?")
        self.opt.ctx_reason = f"autotune medido aplicado | score {score}"
        self.opt.ngl_reason = f"autotune medido aplicado | GPU layers {self.opt.ngl}"
        self.opt.n_cpu_moe_reason = (
            f"autotune medido aplicado | {self.opt.n_cpu_moe} camadas MoE CPU"
        )
        self.opt.cache_reason = (
            f"autotune medido aplicado | cache {self.opt.cache_k}/{self.opt.cache_v}"
        )

    def recalculate_memory(self, raw: dict):
        requested = str(raw.get("omni", "n")).lower()
        if requested not in {"y", "n"}:
            raise ValueError("omni: use y ou n")
        swa_full = str(raw.get("swa_full", self.params.get("swa_full", "n"))).lower()
        if swa_full not in {"y", "n"}:
            raise ValueError("swa_full: use y ou n")
        cache_k = str(raw.get("cache_k", self.params.get("cache_k", "f16"))).lower()
        cache_v = str(raw.get("cache_v", self.params.get("cache_v", "f16"))).lower()
        spec_type = str(raw.get("spec_type", self.params.get("spec_type", "none"))).lower()
        if spec_type not in SPECULATIVE_TYPES:
            raise ValueError(
                "spec_type inválido; use " + ", ".join(SPECULATIVE_TYPES)
            )
        if cache_k not in KV_CACHE_TYPES or cache_v not in KV_CACHE_TYPES:
            raise ValueError("tipo de cache KV inválido")
        try:
            batch = int(raw.get("batch", self.params.get("batch", 2048)))
            ubatch = int(raw.get("ubatch", self.params.get("ubatch", 512)))
        except (TypeError, ValueError):
            raise ValueError("batch e micro-batch devem ser inteiros") from None
        if batch < 1 or ubatch < 1:
            raise ValueError("batch e micro-batch devem ser maiores que zero")
        if ubatch > batch:
            raise ValueError("micro-batch não pode ser maior que batch")
        with self.lock:
            if not self.meta or not self.model_path:
                raise ValueError("Selecione um modelo GGUF primeiro.")
            previous = dict(self.params)
            current = dict(self.params)
            current.update({key: value for key, value in raw.items() if key in current})

            # The form is intentionally sent in full so that edits are not
            # lost. Keep the trigger separately: without it, changing cache K
            # also looked like an explicit edit of the old ctx value and the
            # context window was never recalculated.
            recalculation_fields = {
                "ctx", "cache_k", "cache_v", "batch", "ubatch", "omni",
                "swa_full", "spec_type", "parallel", "fit_target", "fit",
                "kv_offload", "cache_ram", "ctx_checkpoints", "fit_ctx",
            }
            changed_field = str(raw.get("recalculate_field", "")).lower()
            if changed_field not in recalculation_fields:
                changed_field = ""
                for key in recalculation_fields:
                    if key in raw and str(raw[key]).strip().lower() != str(
                        previous.get(key, "")
                    ).strip().lower():
                        changed_field = key
                        break

            # GLM-4.7-Flash uses MLA and llama.cpp rejects mixed K/V cache
            # types for it. Treat both selectors as one atomic setting.
            if _requires_symmetric_kv(self.meta) and cache_k != cache_v:
                if changed_field == "cache_v":
                    cache_k = cache_v
                else:
                    cache_v = cache_k
                current["cache_k"] = cache_k
                current["cache_v"] = cache_v

            opt = self._new_optimal_params(self.meta)
            opt.vision_enabled = requested == "y"
            try:
                opt.parallel = max(int(current.get("parallel", 1)), 1)
            except (TypeError, ValueError):
                opt.parallel = 1
            try:
                opt.fit_target = max(int(current.get("fit_target", 1024)), 0)
            except (TypeError, ValueError):
                opt.fit_target = 1024
            try:
                opt.fit_ctx = max(int(current.get("fit_ctx", 4096)), 1)
            except (TypeError, ValueError):
                opt.fit_ctx = 4096
            try:
                opt.cache_ram = max(int(current.get("cache_ram", 2048)), 0)
                opt.ctx_checkpoints = max(
                    int(current.get("ctx_checkpoints", 32)), 0
                )
            except (TypeError, ValueError):
                raise ValueError("cache RAM e context checkpoints devem ser inteiros >= 0") from None
            opt.fit = "y" if str(current.get("fit", "y")).lower() == "y" else "n"
            # Set memory-affecting values before calculate(). This makes the
            # internal fit planner use the same K/V, batch, fit margin and
            # minimum context that the command will receive.
            opt.cache_k = cache_k
            opt.cache_v = cache_v
            opt.batch = batch
            opt.ubatch = ubatch
            opt.spec_type = spec_type
            opt.swa_full = swa_full
            opt.calculate()
            opt.spec_type = spec_type
            opt.swa_full = swa_full
            opt.recalculate_memory(cache_k, cache_v, batch, ubatch)
            # Cache RAM and context checkpoints are host allocations made by
            # llama-server after loading.  They do not change the CUDA fit,
            # but must participate in the final host/swap admission plan.
            opt.cache_ram = max(int(current.get("cache_ram", 2048)), 0)
            opt.ctx_checkpoints = max(int(current.get("ctx_checkpoints", 32)), 0)
            if changed_field == "ctx":
                try:
                    opt.ctx = min(
                        max(int(current.get("ctx", opt.ctx)), 1),
                        max(int(self.meta.ctx_max), 1),
                    )
                except (TypeError, ValueError):
                    raise ValueError("contexto deve ser um inteiro válido") from None
            if changed_field == "kv_offload":
                requested_kv_offload = str(current.get("kv_offload", "y")).lower()
                if requested_kv_offload not in {"y", "n"}:
                    raise ValueError("KV offload: use GPU ou CPU")
                opt.kv_offload = requested_kv_offload
                opt.kv_offload_reason = (
                    "seleção manual — GPU gerencia o cache KV"
                    if requested_kv_offload == "y"
                    else "seleção manual — cache KV permanece na CPU"
                )
            opt._plan_host_memory()
            # Durante a edição, o perfil do modelo é apenas referência. Ele é
            # aplicado uma vez em select_model(); reaplicá-lo aqui fazia um
            # campo recém-alterado (por exemplo cache_k=f16) voltar para o
            # valor do perfil (por exemplo bf16) quando o formulário era
            # renderizado novamente.
            recommended = self._optimal_values(
                opt, self.meta, apply_profile=False
            )
            adaptive_moe = (
                self.meta.arch.lower() in {"laguna", "qwen3moe", "qwen35moe"}
                or (
                    self.meta.arch.lower() == "gemma4"
                    and self.meta.expert_count > 0
                )
            )
            for key in (
                "ctx", "ngl", "cache_k", "cache_v", "flash", "batch", "ubatch",
                "omni", "mmproj_offload", "mtmd_batch_max", "image_min_tokens",
                "image_max_tokens", "fit", "fit_target", "cpu_moe", "n_cpu_moe",
            ):
                # HTMX envia o formulário inteiro com hx-include. Se o campo
                # veio na requisição, ele é uma escolha explícita do usuário
                # e deve sobreviver ao recálculo. Só preenchemos ausentes com
                # o valor calculado automaticamente.
                if key not in raw:
                    current[key] = recommended[key]
            # K/V, batch, visão, SWA e especulação alteram a curva de
            # memória. O formulário HTMX inclui todos os campos, portanto a
            # simples presença de ctx/n_cpu_moe em ``raw`` não significa que
            # o usuário os editou nesta requisição. Quando um controle de
            # memória dispara o recálculo, substitua os resultados derivados
            # pelo novo plano e preserve as entradas escolhidas (K/V,
            # batch/ubatch, visão, SWA e spec).
            if changed_field and (
                adaptive_moe or str(current.get("fit", "y")).lower() == "y"
            ):
                derived_keys = [
                    "ctx", "ngl", "cpu_moe", "n_cpu_moe", "kv_offload", "flash",
                    "mmproj_offload", "mtmd_batch_max", "image_min_tokens", "fit_ctx",
                    "image_max_tokens",
                ]
                if changed_field == "kv_offload":
                    derived_keys.remove("kv_offload")
                if changed_field == "ctx":
                    derived_keys.remove("ctx")
                if changed_field == "fit_ctx":
                    derived_keys.remove("fit_ctx")
                for key in derived_keys:
                    current[key] = recommended[key]
            # A measured MoE placement is a concrete launch plan.  It must
            # not be lost merely because HTMX submitted the previous value of
            # every field along with the one that changed.  In particular,
            # changing K/V or batch must refresh 262K/40/N, where N is the
            # newly measured CPU-expert count.
            if adaptive_moe and int(recommended.get("n_cpu_moe", 0)) > 0:
                current["fit"] = "n"
                current["cpu_moe"] = "y"
                current["n_cpu_moe"] = recommended["n_cpu_moe"]
                if changed_field not in {"ctx", "fit_ctx"}:
                    current["ctx"] = recommended["ctx"]
                    current["fit_ctx"] = recommended["fit_ctx"]
                current["ngl"] = recommended["ngl"]
            try:
                current_ctx = int(current.get("ctx", recommended["ctx"]))
                current["ctx"] = current_ctx
                current["fit_ctx"] = min(
                    int(current.get("fit_ctx", 4096)), current_ctx
                )
            except (TypeError, ValueError):
                current["ctx"] = int(recommended["ctx"])
                current["fit_ctx"] = min(4096, current["ctx"])
            self.opt = opt
            self.params = current
            self.profile_vram_free_mb = int(self.hardware.gpu_vram_free_mb)
        return self.parameter_snapshot()

    def recalculate_for_vision(self, raw: dict):
        return self.recalculate_memory(raw)

    def model_snapshot(self):
        with self.lock:
            if not self.meta:
                return None
            mt = self.meta
            return {
                "path": self.model_path,
                "name": Path(self.model_path).name,
                "size_gb": mt.size_bytes / 1073741824,
                "arch": mt.arch,
                "quant": mt.quant,
                "params": mt.params_str,
                "ctx_max": mt.ctx_max,
                "layers": mt.layers,
                "kv_layers": mt.kv_layers,
                "recurrent_layers": mt.recurrent_layers,
                "attention_layers": mt.attention_layers,
                "moe_layers": mt.moe_layers,
                "dense_layers": mt.dense_layers,
                "layer_layout_valid": mt.layer_layout_valid,
                "swa_layers": mt.swa_layers,
                "global_layers": mt.global_layers,
                "sliding_window": mt.sliding_window,
                "key_len_swa": mt.key_len_swa,
                "val_len_swa": mt.val_len_swa,
                "full_attention_interval": mt.full_attention_interval,
                "heads": mt.heads,
                "heads_kv": mt.heads_kv,
                "expert_count": mt.expert_count,
                "expert_used_count": mt.expert_used_count,
                "state_r": mt.state_r,
                "state_s": mt.state_s,
                "embed": mt.embed,
                "mmproj": Path(mt.mmproj_file).name if mt.mmproj_file else "",
                "mmproj_valid": mt.mmproj_valid,
                "vocoder": Path(mt.vocoder_file).name if mt.vocoder_file else "",
                "has_mtp": mt.has_mtp,
                "supports_reasoning_preserve": mt.supports_reasoning_preserve,
                "profile": Path(mt.profile_file).name if mt.profile_file else "",
            }

    def _optimal_values(
        self,
        opt: OptimalParams,
        meta: ModelMetadata,
        *,
        apply_profile: bool = True,
    ):
        values = {
            "ctx": opt.ctx, "ngl": opt.ngl, "parallel": opt.parallel,
            "cache_k": opt.cache_k, "cache_v": opt.cache_v,
            "kv_unified": "y" if opt.kv_unified else "n",
            "kv_offload": opt.kv_offload, "flash": opt.flash,
            "split_mode": opt.split_mode, "device": opt.device,
            "numa": opt.numa, "repack": opt.repack,
            "load_mode": opt.load_mode,
            "tensor_read_lazy": opt.tensor_read_lazy,
            "direct_io": opt.direct_io, "no_host": opt.no_host,
            "swa_full": opt.swa_full, "cache_reuse": opt.cache_reuse,
            "threads": opt.threads, "threads_batch": opt.threads_batch,
            "batch": opt.batch, "ubatch": opt.ubatch, "poll": opt.poll,
            "host": opt.host, "port": opt.port, "mlock": opt.mlock,
            "no_mmap": "n", "temp": opt.temp, "top_k": opt.top_k,
            "top_p": opt.top_p, "repeat_penalty": opt.repeat_penalty,
            "min_p": opt.min_p, "presence_penalty": opt.presence_penalty,
            "frequency_penalty": opt.frequency_penalty,
            "repeat_last_n": opt.repeat_last_n,
            "seed": opt.seed, "ignore_eos": opt.ignore_eos,
            "sampler_seq": opt.sampler_seq,
            "dry_multiplier": opt.dry_multiplier, "dry_base": opt.dry_base,
            "dry_allowed_length": opt.dry_allowed_length,
            "dry_penalty_last_n": opt.dry_penalty_last_n,
            "top_nsigma": opt.top_nsigma, "typical_p": opt.typical_p,
            "xtc_probability": opt.xtc_probability,
            "xtc_threshold": opt.xtc_threshold,
            "dynatemp_range": opt.dynatemp_range,
            "dynatemp_exp": opt.dynatemp_exp,
            "mirostat": opt.mirostat, "mirostat_lr": opt.mirostat_lr,
            "mirostat_ent": opt.mirostat_ent,
            "adaptive_target": opt.adaptive_target,
            "adaptive_decay": opt.adaptive_decay,
            "reasoning": opt.reasoning, "reasoning_format": opt.reasoning_format,
            "reasoning_budget": opt.reasoning_budget,
            "reasoning_preserve": opt.reasoning_preserve,
            "reasoning_budget_message": opt.reasoning_budget_message,
            "chat_template_kwargs": opt.chat_template_kwargs,
            "omni": opt.omni, "mmproj_offload": opt.mmproj_offload,
            "mtmd_batch_max": opt.mtmd_batch_max, "audio": opt.audio,
            "image_min_tokens": opt.image_min_tokens,
            "image_max_tokens": opt.image_max_tokens,
            "cpu_moe": opt.cpu_moe, "n_cpu_moe": opt.n_cpu_moe,
            "n_cpu_ffn": opt.n_cpu_ffn,
            "sleep_idle": opt.sleep_idle, "jinja": opt.jinja,
            "slot_similarity": opt.slot_similarity,
            "media_path": MEDIA_PATH if os.path.isdir(MEDIA_PATH) else "",
            "tools": "all", "fit": opt.fit, "fit_target": opt.fit_target,
            "fit_ctx": min(opt.fit_ctx, opt.ctx), "cache_ram": opt.cache_ram,
            "ctx_checkpoints": opt.ctx_checkpoints,
            "checkpoint_min_step": opt.checkpoint_min_step,
            "context_shift": opt.context_shift, "warmup": opt.warmup,
            "timeout": opt.timeout, "log_verbosity": opt.log_verbosity,
            "metrics": opt.metrics, "agentic_max_turns": opt.agentic_max_turns,
            "agentic_max_tool_preview_lines": opt.agentic_max_tool_preview_lines,
            "spec_type": opt.spec_type,
            "spec_draft_n_max": opt.spec_draft_n_max,
            "spec_draft_n_min": opt.spec_draft_n_min,
            "spec_draft_p_min": opt.spec_draft_p_min,
            "spec_draft_p_split": opt.spec_draft_p_split,
            "spec_ngram_mod_n_min": opt.spec_ngram_mod_n_min,
            "spec_ngram_mod_n_max": opt.spec_ngram_mod_n_max,
            "spec_ngram_mod_n_match": opt.spec_ngram_mod_n_match,
            "spec_ngram_min_hits": opt.spec_ngram_min_hits,
            "mcp_config_file": opt.mcp_config_file,
            "mcp_config_json": opt.mcp_config_json,
            "mcp_native": self.mcp_native_default,
            "agent_compat": (
                "y" if self.agent_global_default == "y"
                else self.agent_compat_default
            ),
            "agent_global": self.agent_global_default,
            "mcp_policy": self.mcp_policy_default,
            "mcp_workspace": self.mcp_workspace_default,
            "mcp_snn_threads": self.mcp_snn_threads_default,
            "mcp_snn_steps": self.mcp_snn_steps_default,
            "mcp_repeat_limit": self.mcp_repeat_limit_default,
            "rope_scaling_type": opt.rope_scaling_type,
            "rope_scale": opt.rope_scale,
            "rope_freq_base": opt.rope_freq_base,
            "rope_freq_scale": opt.rope_freq_scale,
            "yarn_orig_ctx": opt.yarn_orig_ctx,
            "yarn_ext_factor": opt.yarn_ext_factor,
            "yarn_attn_factor": opt.yarn_attn_factor,
            "yarn_beta_slow": opt.yarn_beta_slow,
            "yarn_beta_fast": opt.yarn_beta_fast,
            "api_key": opt.api_key, "api_key_file": opt.api_key_file,
            "ssl_key_file": opt.ssl_key_file, "ssl_cert_file": opt.ssl_cert_file,
            "cors_origins": opt.cors_origins, "cors_methods": opt.cors_methods,
            "cors_headers": opt.cors_headers,
            "cors_credentials": opt.cors_credentials,
            "threads_http": opt.threads_http,
            "sse_ping_interval": opt.sse_ping_interval,
            "reuse_port": opt.reuse_port, "offline": opt.offline,
            "cont_batching": opt.cont_batching, "cache_prompt": opt.cache_prompt,
            "cache_idle_slots": opt.cache_idle_slots,
            "perf": opt.perf, "check_tensors": opt.check_tensors,
            "op_offload": opt.op_offload, "override_kv": opt.override_kv,
            "backend_sampling": opt.backend_sampling,
            "slot_save_path": opt.slot_save_path,
            "spm_infill": opt.spm_infill,
            "log_file": opt.log_file, "log_colors": opt.log_colors,
            "log_prefix": opt.log_prefix, "log_timestamps": opt.log_timestamps,
            "ui_config_file": opt.ui_config_file,
            "no_mmproj_auto": opt.no_mmproj_auto,
            "agentic": opt.agentic, "alias": opt.alias, "tags": opt.tags,
            "metacognition": False,   # auto-detected from GGUF metadata
        }
        if apply_profile:
            profile = dict(meta.profile_parameters)
            if "spec_mode" in profile and "spec_type" not in profile:
                profile["spec_type"] = "ngram-mod" if profile["spec_mode"] == "ngram" else "none"
            values.update({key: value for key, value in profile.items() if key in values})

        # An explicit measured MoE split is the effective placement plan, not
        # a suggestion for llama.cpp's generic Fit pass. Keep the form,
        # preview and desktop client coherent even if an old per-model profile
        # contains fit=y/cpu_moe=n or if a caller supplied only n_cpu_moe.
        try:
            measured_cpu_moe = int(values.get("n_cpu_moe", 0))
        except (TypeError, ValueError):
            measured_cpu_moe = 0
        if measured_cpu_moe > 0:
            values["cpu_moe"] = "y"
            values["fit"] = "n"
            values["fit_ctx"] = min(
                max(int(values.get("ctx", opt.ctx)), 1),
                max(int(meta.ctx_max), 1),
            )

        # Tool schemas are part of the prompt. The dynamically discovered MCP
        # catalog can use thousands of tokens before the user's message.
        uses_tools = (
            str(values.get("agentic", "n")).lower() == "y"
            or str(values.get("mcp_native", "n")).lower() == "y"
        )
        if uses_tools:
            try:
                context_floor = min(AGENTIC_CONTEXT_MIN, int(meta.ctx_max))
                values["ctx"] = max(int(values["ctx"]), context_floor)
                values["fit_ctx"] = min(
                    max(int(values.get("fit_ctx", 4096)), context_floor),
                    values["ctx"],
                )
            except (TypeError, ValueError):
                pass
        return values

    def parameter_snapshot(self):
        with self.lock:
            values = dict(self.params)
            opt = self.opt
        reasons = []
        command_facts = {}
        if opt:
            # ``opt`` is the planner's last calculation, while ``values`` is
            # the form state that preview_command()/start_server() actually
            # sends to llama-server.  They can intentionally differ (for
            # example, a 256K user target can force host-side KV while the
            # VRAM-only planner found a smaller context).  Never present the
            # planner explanation as if it were the command that will run.
            try:
                requested_ctx = int(values.get("ctx", opt.ctx))
            except (TypeError, ValueError):
                requested_ctx = int(opt.ctx)
            try:
                fit_ctx = int(values.get("fit_ctx", requested_ctx))
            except (TypeError, ValueError):
                fit_ctx = requested_ctx
            fit_requested = str(values.get("fit", "y")).lower() == "y"
            try:
                n_cpu_moe = int(values.get("n_cpu_moe", 0))
            except (TypeError, ValueError):
                n_cpu_moe = 0
            explicit_cpu_moe = (
                str(values.get("cpu_moe", "n")).lower() == "y"
                or n_cpu_moe > 0
            )
            # build_cmd() disables native fit when an explicit MoE placement
            # is present, so report the effective flag rather than the form
            # toggle alone.
            command_fit = "off" if fit_requested and explicit_cpu_moe else (
                "on" if fit_requested else "off"
            )
            if command_fit == "on":
                command_ngl = "auto"
            else:
                manual_ngl = values.get("ngl", opt.ngl)
                try:
                    if int(manual_ngl) >= int(self.meta.layers):
                        manual_ngl = "all"
                except (TypeError, ValueError):
                    pass
                command_ngl = manual_ngl
            command_kv = (
                "CPU (--no-kv-offload)"
                if str(values.get("kv_offload", "y")).lower() == "n"
                else "GPU (--kv-offload)"
            )
            command_load_mode = str(values.get("load_mode", opt.load_mode)).lower()
            command_facts = {
                "context": requested_ctx,
                "planner_context": int(opt.ctx),
                "fit": command_fit,
                "fit_target": values.get("fit_target", opt.fit_target),
                "fit_ctx": fit_ctx,
                "gpu_layers": command_ngl,
                "planner_gpu_layers": int(opt.ngl),
                "kv": f"{values.get('cache_k', opt.cache_k)}/{values.get('cache_v', opt.cache_v)}",
                "kv_offload": command_kv,
                "load_mode": command_load_mode,
                "flash": str(values.get("flash", opt.flash)).lower(),
                "device": str(values.get("device", opt.device) or "auto"),
                "batch": f"{values.get('batch', opt.batch)}/{values.get('ubatch', opt.ubatch)}",
                "threads": f"{values.get('threads', opt.threads)}/{values.get('threads_batch', opt.threads_batch)}",
                "vision": "on" if str(values.get("omni", "n")).lower() == "y" else "off",
                "mmproj": (
                    Path(self.meta.mmproj_file).name
                    if str(values.get("omni", "n")).lower() == "y"
                    and self.meta.mmproj_file else "none"
                ),
                "mmproj_offload": (
                    "GPU" if str(values.get("mmproj_offload", "n")).lower() == "y"
                    else "CPU"
                ),
                "reasoning": str(values.get("reasoning", opt.reasoning)).lower(),
                "sampling": (
                    f"temp={values.get('temp', opt.temp)} | "
                    f"top-k={values.get('top_k', opt.top_k)} | "
                    f"top-p={values.get('top_p', opt.top_p)} | "
                    f"min-p={values.get('min_p', opt.min_p)}"
                ),
            }
            context_reason = opt.ctx_reason
            if requested_ctx != int(opt.ctx):
                context_reason += (
                    f" | planejador VRAM: {int(opt.ctx)} tokens; "
                    f"comando/UI: {requested_ctx} tokens"
                )
            try:
                explicit_cpu_moe = int(values.get("n_cpu_moe", 0)) > 0
            except (TypeError, ValueError):
                explicit_cpu_moe = False
            if str(values.get("fit", "y")).lower() == "y" and explicit_cpu_moe:
                fit_reason = (
                    opt.fit_plan_reason
                    or "plano MoE calculado antecipadamente; o comando usa --fit off "
                       "porque --n-cpu-moe e o Fit nativo são incompatíveis"
                )
            elif str(values.get("fit", "y")).lower() == "y":
                fit_reason = (
                    f"Fit nativo ativo | piso {values.get('fit_ctx')} | "
                    f"margem {values.get('fit_target')} MiB"
                )
            else:
                fit_reason = "Fit nativo e planejamento automático desativados"
            if requested_ctx > opt.ctx and (
                str(values.get("agentic", "n")).lower() == "y"
                or str(values.get("mcp_native", "n")).lower() == "y"
            ):
                context_reason += " | piso agentic/MCP aplicado ao comando"
            gpu_reason = opt.ngl_reason
            if command_fit == "on":
                gpu_reason += (
                    f" | comando: --n-gpu-layers auto; "
                    f"estimativa estrutural atual: {int(opt.ngl)}/{self.meta.layers}"
                )
            else:
                gpu_reason += f" | comando: --n-gpu-layers {command_ngl}"
            kv_reason = opt.kv_offload_reason
            kv_reason += f" | comando efetivo: {command_kv}"
            load_reason = opt.load_mode_reason or f"padrao — {opt.load_mode}"
            if command_load_mode != str(opt.load_mode).lower():
                load_reason += f" | comando/UI: load-mode {command_load_mode}"
            else:
                load_reason += f" | comando efetivo: load-mode {command_load_mode}"
            sampling_reason = (
                f"comando efetivo: {command_facts['sampling']} | "
                f"{opt.sampling_reason}"
            )
            planned_gpu_weights_mb = math.ceil(
                opt._gpu_weight_bytes(opt.ngl) / 1048576
            )
            planned_rs_mb = math.ceil(opt._recurrent_state_bytes() / 1048576)
            planned_kv_mb = (
                int(opt.kv_device_mb)
                if str(values.get("kv_offload", "y")).lower() != "n" else 0
            )
            planned_mmproj_mb = (
                int(self.meta.mmproj_size_mb)
                if str(values.get("omni", "n")).lower() == "y"
                and str(values.get("mmproj_offload", "n")).lower() == "y"
                else 0
            )
            planned_gpu_total_mb = (
                planned_gpu_weights_mb + planned_kv_mb + planned_rs_mb
                + planned_mmproj_mb + int(opt.runtime_overhead_mb)
            )
            reasons = [
                ("Contexto", context_reason),
                ("Fit efetivo", fit_reason),
                ("GPU layers", gpu_reason),
                ("CPU MoE", opt.n_cpu_moe_reason),
                ("CPU FFN", opt.n_cpu_ffn_reason),
                ("Cache KV", opt.cache_reason),
                ("Sampling", sampling_reason),
                ("Flash attention", opt.flash_reason),
                ("KV unified", opt.kv_reason),
                ("KV offload", kv_reason),
                ("Memória GPU planejada", (
                    f"{planned_gpu_total_mb} MB = pesos {planned_gpu_weights_mb} "
                    f"+ KV {planned_kv_mb} + RS {planned_rs_mb} + MMProj "
                    f"{planned_mmproj_mb} + workspace {opt.runtime_overhead_mb}; "
                    f"reserva livre adicional {values.get('fit_target', opt.fit_target)} MB"
                )),
                ("Threads", opt.threads_reason),
                ("Threads batch", opt.threads_batch_reason),
                ("Batch", opt.batch_reason),
                ("Device", opt.device_reason),
                ("NUMA", opt.numa_reason),
                ("Repack", opt.repack_reason),
                ("SWA full", opt.swa_reason),
                ("Cache reuse", opt.cache_reuse_reason),
                ("Speculative", opt.spec_type_reason),
                ("Reasoning preserve", opt.reasoning_preserve_reason),
                ("Omni/visao", opt.omni_reason),
                ("MMProj offload", opt.mmproj_offload_reason),
                ("MTMD batch", opt.mtmd_batch_reason),
                ("Image min tokens", opt.image_min_tokens_reason),
                ("Spec draft N/P", opt.spec_draft_n_max_reason + " | " + opt.spec_draft_p_min_reason),
                ("Agente universal", "perfil OpenAI-compatible sera gerado ao iniciar" if self.params.get("agent_compat") == "y" else "desabilitado"),
                ("MCP nativo", "integrado via stdio; o llama-server controla o ciclo de vida" if self.params.get("mcp_native") == "y" else "desabilitado"),
                ("MetaCognitiveHead", "detectada no GGUF — avalia cada resposta automaticamente") if self.params.get("metacognition") else ("MetaCognitiveHead", "nao detectada neste GGUF"),
                ("Load mode", load_reason),
                ("Memória host", (
                    f"pico {opt.host_peak_mb} MB | pesos host "
                    f"{opt.host_tensor_mb} MB | cache prompt "
                    f"{opt.prompt_cache_peak_mb} MB | checkpoints "
                    f"{opt.checkpoint_peak_mb} MB | lacuna residente "
                    f"{opt.memory_shortfall_mb} MB"
                )),
                ("Swap NVMe dinâmico", opt.swap_plan_reason),
                ("Tensor read lazy", opt.tensor_read_lazy_reason),
                ("Backend sampling", opt.backend_sampling_reason),
                ("Cont batching", opt.cont_batching_reason or f"padrao — {opt.cont_batching}"),
                ("Cache prompt", opt.cache_prompt_reason or f"padrao — {opt.cache_prompt}"),
            ]
            if opt.autotune_hit:
                metrics = opt.autotune_hit.get("metrics", {})
                reasons.append(
                    (
                        "Autotune",
                        f"configuração medida aplicada: {opt.autotune_hit.get('record_id', '')[:12]} "
                        f"| score {metrics.get('score', '?')}",
                    )
                )
            else:
                reasons.append(("Autotune", "sem configuração medida validada para este workload"))
        return {
            "values": values,
            "reasons": reasons,
            "command_facts": command_facts,
            "planning_hardware": {
                "gpu_vram_free_mb": int(self.profile_vram_free_mb),
                "ram_avail_mb": int(self.hardware.ram_avail_mb),
                "swap_recommended_gib": (
                    int(opt.swap_recommended_gib) if opt else 0
                ),
            },
            "autotune": {
                "status": "hit" if self.opt and self.opt.autotune_hit else "miss",
                "record_id": (
                    self.opt.autotune_hit.get("record_id", "")
                    if self.opt and self.opt.autotune_hit else ""
                ),
                "metrics": (
                    dict(self.opt.autotune_hit.get("metrics", {}))
                    if self.opt and self.opt.autotune_hit else {}
                ),
            },
        }

    def _coerce_final(self, raw: dict):
        with self.lock:
            if not self.opt or not self.meta or not self.model_path:
                raise ValueError("Selecione um modelo GGUF primeiro.")
            raw = dict(raw)
            values = dict(self.params)
            values.update({key: value for key, value in raw.items() if key in values})
            values["model_path"] = self.model_path
            meta = self.meta
            opt = self.opt
            hardware = self.hardware
        for key in INTEGER_FIELDS:
            try:
                values[key] = int(values.get(key, 0))
            except (TypeError, ValueError):
                raise ValueError(f"{key}: informe um inteiro valido") from None
        for key in FLOAT_FIELDS:
            try:
                values[key] = float(values.get(key, 0))
            except (TypeError, ValueError):
                raise ValueError(f"{key}: informe um decimal valido") from None
        ngl = str(values.get("ngl", "auto")).lower()
        if ngl not in {"auto", "all"}:
            try:
                ngl = int(ngl)
            except ValueError:
                raise ValueError("ngl: use auto, all ou um numero inteiro") from None
        values["ngl"] = ngl
        values["kv_unified"] = str(values.get("kv_unified", "n")) == "y"
        lm = str(values.get("load_mode", "mmap")).lower()
        if lm not in {"none", "mmap", "mlock", "mmap+mlock", "dio"}:
            raise ValueError("load_mode: use none, mmap, mlock, mmap+mlock ou dio")
        values["load_mode"] = lm
        if lm in {"mlock", "mmap+mlock"}:
            projector_mb = (
                int(meta.mmproj_size_mb)
                if str(values.get("omni", "n")).lower() == "y"
                and meta.mmproj_file else 0
            )
            reserve_mb = opt._ram_safety_reserve_mb()
            locked_required_mb = int(meta.size_mb) + projector_mb + reserve_mb
            if locked_required_mb > int(hardware.ram_avail_mb):
                raise ValueError(
                    f"load_mode={lm} recusado: mlock impediria paginação de "
                    f"aproximadamente {int(meta.size_mb) + projector_mb} MiB; "
                    f"com reserva adaptativa seriam {locked_required_mb} MiB para "
                    f"{hardware.ram_avail_mb} MiB disponíveis. Use none com swap "
                    "NVMe prioritário ou mmap."
                )
        tensor_read_lazy = str(values.get("tensor_read_lazy", "auto")).lower()
        if tensor_read_lazy not in {"auto", "on", "off"}:
            raise ValueError("tensor_read_lazy: use auto, on ou off")
        if tensor_read_lazy == "on" and lm not in {"mmap", "mmap+mlock"}:
            raise ValueError("tensor_read_lazy=on exige load_mode mmap ou mmap+mlock")
        if (
            tensor_read_lazy != "auto"
            and not _server_lazy_mode_flag(self.llama_server)
        ):
            raise ValueError(
                "O llama-server selecionado nao suporta --lazy-mode "
                "nem --tensor-read-lazy"
            )
        values["tensor_read_lazy"] = tensor_read_lazy
        if not 0 <= values["n_cpu_moe"] <= meta.layers:
            raise ValueError(f"n_cpu_moe deve estar entre 0 e {meta.layers}")
        if values["n_cpu_moe"] > 0:
            # llama.cpp's --n-cpu-moe installs tensor-level overrides and is
            # mutually exclusive with the generic --fit placement pass.
            # Normalize stale/manual combinations before command generation.
            values["cpu_moe"] = "y"
            values["fit"] = "n"
        if not 0 <= values["n_cpu_ffn"] <= meta.layers:
            raise ValueError(f"n_cpu_ffn deve estar entre 0 e {meta.layers}")
        if values["n_cpu_ffn"] > 0 and meta.expert_count > 0:
            raise ValueError("n_cpu_ffn e exclusivo para modelos densos; use n_cpu_moe neste MoE")
        if (
            values["n_cpu_ffn"] > 0
            and not _server_supports_flag(self.llama_server, "--n-cpu-ffn")
        ):
            raise ValueError("O llama-server selecionado nao suporta --n-cpu-ffn")
        backend_sampling = str(values.get("backend_sampling", "auto")).lower()
        if backend_sampling not in {"auto", "y", "n"}:
            raise ValueError("backend_sampling: use auto, y ou n")
        values["backend_sampling"] = backend_sampling
        if values["ctx"] < 1 or values["ctx"] > meta.ctx_max:
            raise ValueError(f"ctx deve estar entre 1 e {meta.ctx_max}")
        if values["ubatch"] > values["batch"]:
            raise ValueError("ubatch nao pode ser maior que batch")
        if not (1 <= values["port"] <= 65535):
            raise ValueError("porta deve estar entre 1 e 65535")
        try:
            host = ipaddress.ip_address(values["host"])
        except ValueError:
            if values["host"] != "localhost":
                raise ValueError("host deve ser um IP valido ou localhost") from None
            host = ipaddress.ip_address("127.0.0.1")
        mcp_native = str(values.get("mcp_native", "n")).lower()
        if mcp_native not in {"y", "n"}:
            raise ValueError("mcp_native: use y ou n")
        values["mcp_native"] = mcp_native
        if (values.get("tools") != "none" or mcp_native == "y") and not host.is_loopback:
            raise ValueError("ferramentas agentic e MCP so podem ser usadas em loopback")
        agent_compat = str(
            values.get("agent_compat", self.agent_compat_default)
        ).lower()
        if agent_compat not in {"y", "n"}:
            raise ValueError("agent_compat: use y ou n")
        agent_global = str(values.get("agent_global", self.agent_global_default)).lower()
        if agent_global not in {"y", "n"}:
            raise ValueError("agent_global: use y ou n")
        # The global switch necessarily needs the client-neutral profile. A
        # stale form/settings value must be promoted, not rejected, otherwise
        # selecting a new model breaks an already-active global session.
        if agent_global == "y":
            agent_compat = "y"
        values["agent_compat"] = agent_compat
        values["agent_global"] = agent_global
        # No modo universal, o cliente externo mantém suas próprias ferramentas
        # e skills. Não exponha também o MCP do Crono Matrix ao llama-server:
        # além de duplicar ferramentas, isso cria dois ciclos de agente.
        if agent_global == "y":
            mcp_native = "n"
            values["mcp_native"] = "n"
        mcp_policy = str(values.get("mcp_policy", "safe")).lower()
        if mcp_policy not in {"safe", "all"}:
            raise ValueError("mcp_policy: use safe ou all")
        values["mcp_policy"] = mcp_policy
        if not 1 <= values["mcp_snn_threads"] <= 8:
            raise ValueError("mcp_snn_threads deve estar entre 1 e 8")
        if not 1 <= values["mcp_snn_steps"] <= 256:
            raise ValueError("mcp_snn_steps deve estar entre 1 e 256")
        if not 2 <= values["mcp_repeat_limit"] <= 5:
            raise ValueError("mcp_repeat_limit deve estar entre 2 e 5")
        values["mcp_native_json"] = ""
        if mcp_native == "y":
            node, _node_version, node_error = _resolve_node_runtime()
            workspace = Path(str(values.get("mcp_workspace", ""))).expanduser().resolve()
            if not node:
                raise ValueError(_node_unavailable_message(node_error))
            if not NATIVE_MCP_ENTRY.is_file():
                raise ValueError(f"Servidor MCP nativo nao encontrado: {NATIVE_MCP_ENTRY}")
            if not (NATIVE_MCP_DIR / "node_modules" / "@modelcontextprotocol" / "sdk").is_dir():
                raise ValueError(f"Dependencias MCP ausentes. Execute: npm install --prefix {NATIVE_MCP_DIR}")
            if not workspace.is_dir():
                raise ValueError(f"Workspace MCP nao encontrado: {workspace}")
            values["mcp_workspace"] = str(workspace)
            native_config = {
                "mcpServers": {
                    "crono-matrix": {
                        "command": node,
                        "args": [str(NATIVE_MCP_ENTRY)],
                        "cwd": str(NATIVE_MCP_DIR),
                        "timeout_ms": 120000,
                        "env": {
                            "MCP_STDIO": "1",
                            "CRONO_PROJECT_ROOT": str(PROJECT_ROOT),
                            "CRONO_WORKSPACE": str(workspace),
                            "CRONO_MODELS_DIR": self.models_dir,
                            "CRONO_LLAMA_PATH": self.llama_server,
                            "CRONO_LLAMA_HOST": str(values["host"]),
                            "CRONO_LLAMA_PORT": str(values["port"]),
                            "CRONO_MCP_TOOL_POLICY": mcp_policy,
                            "CRONO_SNN_THREADS": str(values["mcp_snn_threads"]),
                            "CRONO_SNN_STEPS": str(values["mcp_snn_steps"]),
                             "CRONO_SNN_TIMEOUT_MS": "15000",
                             "CRONO_MEMORY_DIR": str(SNN_DIR.parent),
                             "CRONO_SNN_ENABLED_FILE": str(SNN_ENABLED_FILE),
                             "CRONO_MCP_REPEAT_LIMIT": str(values["mcp_repeat_limit"]),
                            "CRONO_MCP_MAX_RESULT_CHARS": "32000",
                            "LOG_LEVEL": "info",
                        },
                    },
                },
            }
            values["mcp_native_json"] = json.dumps(
                native_config, ensure_ascii=True, separators=(",", ":")
            )
        adaptive_moe = (
            meta.arch.lower() in {"laguna", "qwen3moe", "qwen35moe"}
            or (meta.arch.lower() == "gemma4" and meta.expert_count > 0)
        )
        if adaptive_moe and values.get("fit") == "y" and values["ctx"] != opt.ctx:
            opt.cache_k = values["cache_k"]
            opt.cache_v = values["cache_v"]
            opt.batch = values["batch"]
            opt.ubatch = values["ubatch"]
            opt.spec_type = values["spec_type"]
            opt.swa_full = values["swa_full"]
            opt.fit_target = values["fit_target"]
            opt._update_runtime_overhead()
            if meta.arch.lower() == "laguna":
                adapt = opt._adapt_laguna
            elif meta.arch.lower() == "gemma4":
                adapt = opt._adapt_gemma4_moe
            elif meta.arch.lower() == "qwen35moe":
                adapt = opt._adapt_qwen35moe
            else:
                adapt = opt._adapt_qwen3moe
            if not adapt(values["ctx"]):
                raise ValueError(
                    "O contexto solicitado nao cabe na VRAM com o perfil MoE adaptativo. "
                    "Reduza ctx ou aumente fit-target."
                )
            values["ctx"] = opt.ctx
            values["ngl"] = opt.ngl
            values["n_cpu_moe"] = opt.n_cpu_moe
        values["fit_ctx"] = min(values["fit_ctx"], values["ctx"])
        return values

    @staticmethod
    def _local_api_root(host: str, port: int) -> str:
        """Build a loopback URL that also works when IPv6 is selected."""
        host_text = str(host)
        if host_text in {"0.0.0.0", "localhost"}:
            host_text = "127.0.0.1"
        elif host_text == "::":
            host_text = "::1"
        if ":" in host_text and not host_text.startswith("["):
            host_text = f"[{host_text}]"
        return f"http://{host_text}:{int(port)}"

    @staticmethod
    def _write_atomic_text(path: Path, content: str, mode: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)

    def _agent_model_id(self, final: dict) -> str:
        alias = str(final.get("alias", "")).strip()
        if alias:
            return alias
        return Path(str(final.get("model_path", ""))).stem

    @staticmethod
    def _agent_context_policy(context_window: int) -> tuple[int, int]:
        """Return the real runtime context and a safe client compaction point."""
        context = max(int(context_window or 1), 1)
        # Leave room for the current turn, tool schemas and the model's answer.
        # This is not a smaller server context: it is the point at which a
        # client should compact the conversation before the hard llama.cpp limit.
        if context <= 1024:
            compact = max(1, int(context * 0.8))
        else:
            compact = min(context - 512, max(1024, int(context * 0.8)))
        return context, compact

    def _agent_native_context(self) -> int:
        """Native GGUF limit, never the requested/extended runtime window.

        Zero denotes unknown: ModelMetadata's initial 4096 is not evidence.
        """
        meta = getattr(self, "meta", None)
        return max(int(getattr(meta, "ctx_max", 0) or 0), 0) if (
            meta and getattr(meta, "meta_ok", False)
        ) else 0

    def _runtime_agent_capabilities(
        self, final: dict, props: dict, chat_caps: dict,
        generation_settings: dict, reasoning_enabled: bool,
    ) -> tuple[dict, list[str], dict]:
        """Translate llama.cpp's runtime properties to client capabilities.

        ``/props`` is authoritative once the server is ready.  GGUF metadata
        is used only as a fallback for callers that generate a profile before
        a runtime response exists (for example, a preview or a unit test).
        """
        meta = getattr(self, "meta", None)
        raw_modalities = props.get("modalities")
        if isinstance(raw_modalities, dict) and any(
            key in raw_modalities for key in ("vision", "video", "audio")
        ):
            media_input = {
                "image": bool(raw_modalities.get("vision")),
                "video": bool(raw_modalities.get("video")),
                "audio": bool(raw_modalities.get("audio")),
            }
            modality_source = "runtime_props"
        else:
            # A valid projector need not contain a vision encoder. Metadata
            # describes a preview, not proof that the server loaded it. Older
            # runtimes without modality flags must not inherit the preview.
            projector_requested = bool(
                not props
                and str(final.get("omni", "n")).lower() == "y"
                and meta and meta.mmproj_file and meta.mmproj_valid
            )
            media_input = {
                "image": bool(
                    projector_requested and meta.mmproj_has_vision
                ),
                "video": False,
                "audio": bool(projector_requested and meta.mmproj_has_audio),
            }
            modality_source = "gguf_preview" if projector_requested else "unknown"

        input_modalities = ["text"] + [
            name for name in ("image", "video", "audio") if media_input[name]
        ]
        tools_requested = (
            str(final.get("agentic", "n")).lower() == "y"
            or str(final.get("tools", "none")).lower() in {"all", "readonly"}
        )
        template_caps = {
            name: bool(chat_caps.get(name, False))
            for name in (
                "supports_tools", "supports_tool_calls",
                "supports_parallel_tool_calls", "supports_system_role",
                "supports_preserve_reasoning", "supports_reasoning_effort",
                "supports_string_content", "supports_typed_content",
                "supports_object_arguments",
            )
        }
        interleaved = bool(
            reasoning_enabled
            and (
                template_caps["supports_preserve_reasoning"]
                or bool(getattr(meta, "supports_reasoning_preserve", False))
            )
        )
        capabilities = {
            "temperature": True,
            "reasoning": bool(reasoning_enabled),
            "attachment": len(input_modalities) > 1,
            "tool_call": template_caps["supports_tool_calls"],
            "input": {
                "text": True,
                "audio": media_input["audio"],
                "image": media_input["image"],
                "video": media_input["video"],
                "pdf": False,
            },
            "output": {
                "text": True,
                "audio": False,
                "image": False,
                "video": False,
                "pdf": False,
            },
            "interleaved": (
                {"field": "reasoning_content"} if interleaved else False
            ),
        }
        native_context = self._agent_native_context()
        capabilities["evidence"] = {
            "input_modalities": modality_source,
            "native_context_window": "gguf" if native_context else "unknown",
            "tool_call": "runtime_template" if "supports_tool_calls" in chat_caps else "unknown",
        }
        capabilities["model"] = {
            "architecture": str(getattr(meta, "arch", "") or ""),
            "native_context_window": native_context,
            "projector": {
                "available": bool(
                    meta and meta.mmproj_file and meta.mmproj_valid
                ),
                "vision": bool(
                    getattr(meta, "mmproj_has_vision", False)
                ),
                "audio_input": bool(
                    getattr(meta, "mmproj_has_audio", False)
                ),
                "audio_output": bool(
                    getattr(meta, "mmproj_has_gen_audio", False)
                    or getattr(meta, "vocoder_file", "")
                ),
                "tensor_count": int(
                    getattr(meta, "mmproj_tensor_count", 0) or 0
                ),
            },
        }
        capabilities["runtime"] = {
            "context_window": int(
                generation_settings.get("n_ctx")
                or final.get("ctx", 0)
                or 0
            ),
            "input": dict(media_input),
        }
        supports_reasoning_budget = bool(
            reasoning_enabled
            and (
                template_caps["supports_preserve_reasoning"]
                or template_caps["supports_reasoning_effort"]
                or "<think" in str(props.get("chat_template") or "").lower()
            )
        )
        server_capabilities = {
            "jinja": bool(props.get("chat_template")),
            "tools_requested": tools_requested,
            "template": template_caps,
            "modalities": {
                "vision": media_input["image"],
                "video": media_input["video"],
                "audio": media_input["audio"],
            },
            # llama.cpp nests the effective sampling fields in ``params``.
            # Keep both the raw object and the flattened params so clients do
            # not have to know which server version produced /props.
            "generation_params": dict(
                generation_settings.get("params")
                if isinstance(generation_settings.get("params"), dict)
                else generation_settings
            ),
            "reasoning_format": str(
                (
                    generation_settings.get("params", {}).get("reasoning_format")
                    if isinstance(generation_settings.get("params"), dict)
                    else generation_settings.get("reasoning_format")
                ) or ""
            ),
            "reasoning_budget_tokens": int(
                final.get("reasoning_budget", -1)
                if final.get("reasoning_budget") is not None else -1
            ),
            "supports_reasoning_budget": supports_reasoning_budget,
            "default_generation_settings": generation_settings,
            "chat_template": {
                "available": bool(props.get("chat_template")),
                "tool_use_available": bool(
                    props.get("chat_template_tool_use")
                    or template_caps["supports_tools"]
                    or template_caps["supports_tool_calls"]
                ),
            },
            "media_marker": str(props.get("media_marker") or ""),
            "bos_token": str(props.get("bos_token") or ""),
            "eos_token": str(props.get("eos_token") or ""),
            "model_alias": str(props.get("model_alias") or ""),
            "model_ftype": str(props.get("model_ftype") or ""),
            "build_info": str(props.get("build_info") or ""),
            "endpoint_props": bool(props.get("endpoint_props", False)),
            "endpoint_slots": bool(props.get("endpoint_slots", False)),
            "endpoint_metrics": bool(props.get("endpoint_metrics", False)),
            "cors_proxy_enabled": bool(props.get("cors_proxy_enabled", False)),
        }
        return capabilities, input_modalities, server_capabilities

    @staticmethod
    def _normalize_native_tools(payload: object) -> list[dict]:
        """Keep the complete, JSON-safe definitions returned by ``GET /tools``."""
        if not isinstance(payload, list):
            return []
        tools = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("tool") or item.get("name") or "").strip()
            if not name:
                continue
            tool = {
                "name": name,
                "display_name": str(item.get("display_name") or name),
                "type": str(item.get("type") or "server"),
                "uses_cwd": bool(item.get("uses_cwd", False)),
                "permissions": item.get("permissions", {}),
                "definition": item.get("definition", {}),
            }
            tools.append(tool)
        return tools

    def _write_agent_catalog(
        self, final: dict, model_id: str, context_window: int,
        compact_limit: int, modalities: list[str] | None = None,
        capabilities: dict | None = None, server_capabilities: dict | None = None,
        native_tools: list[dict] | None = None,
        reasoning_enabled: bool | None = None,
    ) -> tuple[Path, list[str]]:
        """Write compatibility metadata for whichever GGUF is active."""
        modalities = list(modalities or ["text"])
        capabilities = dict(capabilities or {})
        server_capabilities = dict(server_capabilities or {})
        native_tools = list(native_tools or [])
        if reasoning_enabled is None:
            reasoning_enabled = str(final.get("reasoning", "auto")).lower() != "off"
        # Keep the universal catalog aligned with the five levels exposed by
        # llama.cpp's own UI.  The exact budget is a client hint; the server
        # remains authoritative when a model/template does not implement a
        # finite effort control.
        levels = [
            {"effort": "off", "description": "Resposta direta, sem raciocínio."},
            {"effort": "low", "description": "Resposta rápida com raciocínio leve."},
            {"effort": "medium", "description": "Equilíbrio entre velocidade e profundidade."},
            {"effort": "high", "description": "Raciocínio mais profundo para tarefas complexas."},
            {"effort": "max", "description": "Raciocínio máximo, sem orçamento finito imposto pelo cliente."},
        ]
        catalog = {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "client_version": "crono-matrix-local",
            "models": [{
                "slug": model_id,
                "display_name": model_id,
                "description": (
                    "Modelo GGUF local servido diretamente pelo llama.cpp para "
                    "um agente local; metadados sincronizados com a sessão ativa."
                ),
                "default_reasoning_level": "high" if reasoning_enabled else "medium",
                "supported_reasoning_levels": levels if reasoning_enabled else [],
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 0,
                "support_verbosity": False,
                "default_verbosity": "low",
                "apply_patch_tool_type": "freeform",
                "default_reasoning_summary": "none",
                "model_messages": {
                    "instructions_template": LOCAL_AGENT_INSTRUCTIONS,
                },
                # This is the hard runtime context, not a second arbitrary
                # 10K client ceiling. Clients should use auto_compact below
                # for proactive compaction while retaining the full window.
                "truncation_policy": {"mode": "tokens", "limit": context_window},
                "context_window": context_window,
                "max_context_window": context_window,
                "native_context_window": self._agent_native_context(),
                "max_output_tokens": context_window,
                "auto_compact_token_limit": compact_limit,
                "effective_context_window_percent": 95,
                "experimental_supported_tools": [
                    tool["name"] for tool in native_tools if tool.get("name")
                ],
                "input_modalities": modalities,
                "supports_image_detail_original": False,
                "supports_search_tool": False,
                "use_responses_lite": False,
                "node_repl_auto_review_required": False,
                "node_repl_disabled": True,
                "tool_mode": "code_mode_only",
                "capabilities": capabilities,
                "server_capabilities": server_capabilities,
                "native_tools": native_tools,
            }],
        }
        catalog_path = AGENT_COMPAT_DIR / "model-catalog.json"
        self._write_atomic_text(
            catalog_path,
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        )
        return catalog_path, modalities

    def _write_universal_agent_profile(
        self, endpoint: str, model_id: str, context_window: int,
        compact_limit: int, modalities: list[str], reasoning_enabled: bool,
        api_key: str, capabilities: dict | None = None,
        server_capabilities: dict | None = None,
        native_tools: list[dict] | None = None,
        props: dict | None = None,
    ) -> tuple[Path, Path]:
        """Publish connection data usable by any OpenAI-compatible client."""
        api_root = endpoint.rstrip("/") + "/v1"
        capabilities = dict(capabilities or {})
        server_capabilities = dict(server_capabilities or {})
        native_tools = list(native_tools or [])
        props = dict(props or {})
        endpoints = {
            "health": endpoint.rstrip("/") + "/health",
            "props": endpoint.rstrip("/") + "/props",
            "tools": endpoint.rstrip("/") + "/tools",
            "models": api_root + "/models",
            "chat_completions": api_root + "/chat/completions",
            "responses": api_root + "/responses",
            "audio_transcriptions": api_root + "/audio/transcriptions",
            "embeddings": api_root + "/embeddings",
        }
        runtime = {}
        if props:
            # Do not copy complete chat templates into every profile: they can
            # be very large. These are the runtime facts needed by clients and
            # diagnostics, including the exact model actually loaded.
            runtime = {
                "model_alias": str(props.get("model_alias") or ""),
                "model_ftype": str(props.get("model_ftype") or ""),
                "model_path": str(props.get("model_path") or ""),
                "build_info": str(props.get("build_info") or ""),
                "modalities": props.get("modalities", {}),
                "media_marker": str(props.get("media_marker") or ""),
                "default_generation_settings": props.get(
                    "default_generation_settings", {}
                ),
                "chat_template_available": bool(props.get("chat_template")),
                "chat_template_tool_use_available": bool(
                    props.get("chat_template_tool_use")
                    or dict(props.get("chat_template_caps") or {}).get(
                        "supports_tools"
                    )
                    or dict(props.get("chat_template_caps") or {}).get(
                        "supports_tool_calls"
                    )
                ),
                "chat_template_caps": props.get("chat_template_caps", {}),
                "endpoint_slots": bool(props.get("endpoint_slots", False)),
                "endpoint_props": bool(props.get("endpoint_props", False)),
                "endpoint_metrics": bool(props.get("endpoint_metrics", False)),
                "cors_proxy_enabled": bool(props.get("cors_proxy_enabled", False)),
            }
        env_lines = [
            "# Gerado pelo Crono Matrix Launcher; não edite durante a execução.",
            "# Use: source .crono-agent/agent-local.env.sh",
            f"export OPENAI_BASE_URL={shlex.quote(api_root)}",
            f"export OPENAI_API_BASE={shlex.quote(api_root)}",
            f"export OPENAI_API_KEY={shlex.quote(api_key)}",
            f"export OPENAI_MODEL={shlex.quote(model_id)}",
            f"export MODEL={shlex.quote(model_id)}",
            f"export CRONO_MODEL_ID={shlex.quote(model_id)}",
            f"export CRONO_API_ROOT={shlex.quote(endpoint.rstrip('/'))}",
            f"export CRONO_CHAT_COMPLETIONS_URL={shlex.quote(endpoints['chat_completions'])}",
            f"export CRONO_RESPONSES_URL={shlex.quote(endpoints['responses'])}",
            f"export CRONO_MODELS_URL={shlex.quote(endpoints['models'])}",
            f"export CRONO_PROPS_URL={shlex.quote(endpoints['props'])}",
            f"export CRONO_TOOLS_URL={shlex.quote(endpoints['tools'])}",
            f"export CRONO_CONTEXT_WINDOW={shlex.quote(str(context_window))}",
            f"export CRONO_NATIVE_CONTEXT_WINDOW={shlex.quote(str(self._agent_native_context()))}",
            f"export CRONO_MAX_OUTPUT_TOKENS={shlex.quote(str(context_window))}",
            f"export CRONO_AUTO_COMPACT_TOKEN_LIMIT={shlex.quote(str(compact_limit))}",
            f"export CRONO_INPUT_MODALITIES={shlex.quote(','.join(modalities))}",
            f"export CRONO_REASONING_ENABLED={shlex.quote('1' if reasoning_enabled else '0')}",
            f"export CRONO_REASONING_EFFORTS={shlex.quote('off,low,medium,high,max' if reasoning_enabled else '')}",
            f"export CRONO_REASONING_FORMAT={shlex.quote('auto' if reasoning_enabled else 'none')}",
            "",
        ]
        metadata = {
            "schema": "crono-matrix.agent.v1",
            "api": "openai-compatible",
            "base_url": api_root,
            "chat_completions_url": api_root + "/chat/completions",
            "responses_url": api_root + "/responses",
            "models_url": api_root + "/models",
            "props_url": endpoint.rstrip("/") + "/props",
            "tools_url": endpoint.rstrip("/") + "/tools",
            "health_url": endpoint.rstrip("/") + "/health",
            "api_key_configured": bool(api_key and api_key != "local"),
            "model": model_id,
            "context_window": context_window,
            "max_context_window": context_window,
            "native_context_window": self._agent_native_context(),
            "max_output_tokens": context_window,
            "auto_compact_token_limit": compact_limit,
            "reasoning_enabled": reasoning_enabled,
            "input_modalities": modalities,
            "capabilities": capabilities,
            "server_capabilities": server_capabilities,
            "native_tools": native_tools,
            "endpoints": endpoints,
            "runtime": runtime,
            "reasoning": {
                "enabled": reasoning_enabled,
                "format": server_capabilities.get("reasoning_format", ""),
                "request_format": "auto" if reasoning_enabled else "none",
                "supported_efforts": (
                    ["off", "low", "medium", "high", "max"]
                    if reasoning_enabled else []
                ),
                "default_effort": "server" if reasoning_enabled else "off",
                "budget_field": "thinking_budget_tokens" if reasoning_enabled else "",
                "supports_effort": bool(
                    server_capabilities.get("supports_reasoning_budget", False)
                    or server_capabilities.get("template", {}).get(
                        "supports_reasoning_effort", False
                    )
                ),
                "supports_budget": bool(
                    server_capabilities.get("supports_reasoning_budget", False)
                ),
            },
            "client_compaction": {
                "recommended_at": compact_limit,
                "hard_limit": context_window,
                "server_managed": False,
            },
            "server_managed_compaction": False,
            "notes": (
                "A compactação é responsabilidade do cliente. O valor indicado "
                "é o ponto recomendado para preservar espaço para ferramentas e saída."
            ),
        }
        self._write_atomic_text(
            AGENT_ENV_FILE, "\n".join(env_lines), mode=0o600,
        )
        self._write_atomic_text(
            AGENT_METADATA_FILE,
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )
        return AGENT_ENV_FILE, AGENT_METADATA_FILE

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_jsonc_object(content: str) -> dict:
        """Parse JSONC without changing comment-like text inside strings."""
        output = []
        index = 0
        in_string = False
        escaped = False
        while index < len(content):
            char = content[index]
            following = content[index + 1] if index + 1 < len(content) else ""
            if in_string:
                output.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                output.append(char)
                index += 1
                continue
            if char == "/" and following == "/":
                index += 2
                while index < len(content) and content[index] not in "\r\n":
                    index += 1
                continue
            if char == "/" and following == "*":
                index += 2
                while index + 1 < len(content) and content[index:index + 2] != "*/":
                    index += 1
                index = min(index + 2, len(content))
                continue
            output.append(char)
            index += 1

        # JSONC accepts trailing commas. Remove only commas outside strings
        # whose next non-whitespace character closes an array/object.
        source = "".join(output)
        cleaned = []
        index = 0
        in_string = False
        escaped = False
        while index < len(source):
            char = source[index]
            if in_string:
                cleaned.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                cleaned.append(char)
                index += 1
                continue
            if char == ",":
                probe = index + 1
                while probe < len(source) and source[probe].isspace():
                    probe += 1
                if probe < len(source) and source[probe] in "}]":
                    index += 1
                    continue
            cleaned.append(char)
            index += 1
        value = json.loads("".join(cleaned) or "{}")
        if not isinstance(value, dict):
            raise ValueError("A configuração do OpenCode deve ser um objeto JSON/JSONC.")
        return value

    @staticmethod
    def _opencode_state() -> dict:
        try:
            value = json.loads(OPENCODE_GLOBAL_STATE.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _opencode_managed_provider(self) -> dict:
        model_id = str(self.agent_info["model"])
        modalities = list(self.agent_info.get("modalities") or ["text"])
        context_window = max(int(self.agent_info.get("context_window") or 1), 1)
        # llama.cpp treats n_predict=-1 as "until the context is full". Do
        # not impose the old 64K/quarter-window client ceiling here; the
        # server and the client's current prompt still enforce the real limit.
        output_limit = context_window
        endpoint = str(self.agent_info["endpoint"]).rstrip("/")
        reasoning_enabled = bool(self.agent_info.get("reasoning_enabled", False))
        capabilities = self.agent_info.get("capabilities")
        has_runtime_capabilities = isinstance(capabilities, dict)
        capabilities = dict(capabilities or {})
        model = {
            "name": model_id,
            "modalities": {
                "input": modalities,
                "output": ["text"],
            },
            "limit": {
                "context": context_window,
                "output": output_limit,
            },
            "temperature": bool(capabilities.get("temperature", True)),
            "attachment": bool(
                capabilities.get("attachment", len(modalities) > 1)
            ),
            "tool_call": bool(capabilities.get("tool_call", False)),
        }
        if reasoning_enabled:
            # llama.cpp defaults to raw inline thinking for some templates
            # (reasoning_format=none).  OpenCode needs the parsed field to
            # render a Thought part, including when its Default variant is
            # selected and no per-variant override is present.
            model["options"] = {"reasoning_format": "auto"}
        if reasoning_enabled:
            # OpenCode otherwise treats a custom OpenAI-compatible model as a
            # plain text model and silently merges ``reasoning_content`` into
            # neither the Thought UI nor its reasoning token accounting.
            model["reasoning"] = True
            interleaved = capabilities.get("interleaved")
            if isinstance(interleaved, dict):
                model["interleaved"] = interleaved
            elif not has_runtime_capabilities:
                # Preserve compatibility with profiles generated before
                # runtime /props capabilities were introduced.
                model["interleaved"] = {"field": "reasoning_content"}
            # Mirror tools/ui/src/lib/constants/reasoning-effort.constants.ts
            # from this exact local llama.cpp checkout.  Token budgets are
            # enforced by llama-server's reasoning sampler independently of a
            # template's supports_reasoning_effort capability.  The effort
            # string is kept as well for templates that do understand it.
            def budget_variant(effort: str, tokens: int) -> dict:
                return {
                    "reasoningEffort": effort,
                    "reasoning_format": "auto",
                    "chat_template_kwargs": {"enable_thinking": True},
                    "thinking_budget_tokens": tokens,
                    "reasoning_budget_tokens": tokens,
                    "reasoning_control": True,
                }

            model["variants"] = {
                "off": {
                    "reasoningEffort": "none",
                    "reasoning_format": "auto",
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                "low": budget_variant("low", 512),
                "medium": budget_variant("medium", 2048),
                "high": budget_variant("high", 8192),
                # The llama.cpp UI represents Max by omitting the finite token
                # budget while keeping thinking and runtime control enabled.
                "max": {
                    "reasoning_format": "auto",
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_control": True,
                },
            }
        return {
            "name": "Crono Matrix - llama.cpp",
            "npm": "@ai-sdk/openai-compatible",
            "options": {
                "baseURL": endpoint,
                "apiKey": "local",
            },
            "models": {
                model_id: model,
            },
        }

    @staticmethod
    def _is_legacy_crono_provider(provider: object) -> bool:
        """Return whether *provider* is an old local Crono adapter.

        The global adapter predates the current state file and used the same
        ``crono`` provider name with a simpler OpenAI-compatible definition.
        Such a provider is safe to migrate because it is demonstrably local
        and belongs to Crono.  This check is intentionally narrow: an
        arbitrary provider using the same name, a remote endpoint, or a
        malformed entry must still be treated as an external edit.
        """
        if not isinstance(provider, dict):
            return False

        name = str(provider.get("name", "")).lower()
        npm = str(provider.get("npm", "")).lower()
        if "crono" not in name or npm != "@ai-sdk/openai-compatible":
            return False

        options = provider.get("options", {})
        if not isinstance(options, dict):
            return False
        base_url = str(options.get("baseURL", "")).rstrip("/").lower()
        local_prefixes = (
            "http://127.0.0.1:",
            "http://localhost:",
            "http://[::1]:",
        )
        if not base_url.startswith(local_prefixes):
            return False

        models = provider.get("models", {})
        return isinstance(models, dict) and bool(models)

    def _activate_opencode_global(self) -> Path:
        """Synchronize the active model into OpenCode's global provider."""
        config_path = OPENCODE_GLOBAL_CONFIG
        original_exists = config_path.is_file()
        current_text = config_path.read_text(encoding="utf-8") if original_exists else "{}\n"
        current = self._parse_jsonc_object(current_text)
        previous_state = self._opencode_state()
        providers = current.setdefault("provider", {})
        if not isinstance(providers, dict):
            raise ValueError("provider em opencode.jsonc deve ser um objeto.")

        reconciled_legacy_provider = False
        if previous_state.get("active"):
            previous_managed = previous_state.get("managed_provider")
            existing = providers.get(OPENCODE_GLOBAL_PROVIDER)
            if existing != previous_managed:
                if not self._is_legacy_crono_provider(existing):
                    raise ValueError(
                        f"{config_path}: o provider '{OPENCODE_GLOBAL_PROVIDER}' foi alterado "
                        "fora do Crono Matrix; não sobrescrevi essas mudanças."
                    )
                # A stale state file plus the old local Crono schema is a
                # migration, not an external edit.  Use the file as it is
                # now as the new rollback point so user changes made since
                # the old activation are preserved on deactivation.
                reconciled_legacy_provider = True
                backup_text = current_text
                original_exists = True
            else:
                backup_text = str(previous_state.get("backup_content", "{}\n"))
                original_exists = bool(previous_state.get("original_exists", False))
        else:
            backup_text = current_text

        managed_provider = self._opencode_managed_provider()
        managed_model = f"{OPENCODE_GLOBAL_PROVIDER}/{self.agent_info['model']}"
        providers[OPENCODE_GLOBAL_PROVIDER] = managed_provider
        current["model"] = managed_model
        rewritten = json.dumps(current, ensure_ascii=False, indent=2) + "\n"
        self._write_atomic_text(config_path, rewritten, mode=0o600)
        state = {
            "active": True,
            "config_path": str(config_path),
            "original_exists": original_exists,
            "backup_content": backup_text,
            "active_sha256": self._sha256_text(rewritten),
            "managed_provider": managed_provider,
            "managed_model": managed_model,
            "reconciled_legacy_provider": reconciled_legacy_provider,
        }
        self._write_atomic_text(
            OPENCODE_GLOBAL_STATE,
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        )
        return config_path

    def _deactivate_opencode_global(self) -> None:
        state = self._opencode_state()
        if not state.get("active"):
            return
        config_path = Path(str(state.get("config_path") or OPENCODE_GLOBAL_CONFIG)).expanduser()
        current_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else "{}\n"
        if self._sha256_text(current_text) == str(state.get("active_sha256", "")):
            if state.get("original_exists"):
                self._write_atomic_text(config_path, str(state.get("backup_content", "{}\n")))
            elif config_path.exists():
                config_path.unlink()
        else:
            current = self._parse_jsonc_object(current_text)
            backup = self._parse_jsonc_object(str(state.get("backup_content", "{}\n")))
            providers = current.get("provider")
            if not isinstance(providers, dict):
                providers = {}
                current["provider"] = providers
            if providers.get(OPENCODE_GLOBAL_PROVIDER) != state.get("managed_provider"):
                raise ValueError(
                    f"{config_path}: o provider gerenciado foi alterado externamente; "
                    "não restaurei para evitar perda de dados."
                )
            backup_providers = backup.get("provider", {})
            if isinstance(backup_providers, dict) and OPENCODE_GLOBAL_PROVIDER in backup_providers:
                providers[OPENCODE_GLOBAL_PROVIDER] = backup_providers[OPENCODE_GLOBAL_PROVIDER]
            else:
                providers.pop(OPENCODE_GLOBAL_PROVIDER, None)
            if current.get("model") == state.get("managed_model"):
                if "model" in backup:
                    current["model"] = backup["model"]
                else:
                    current.pop("model", None)
            self._write_atomic_text(
                config_path, json.dumps(current, ensure_ascii=False, indent=2) + "\n", mode=0o600,
            )
        state["active"] = False
        state["deactivated_at"] = time.time()
        self._write_atomic_text(
            OPENCODE_GLOBAL_STATE,
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        )

    def _activate_agent_global(self, final: dict) -> dict:
        """Enable direct-server mode and synchronize supported local clients."""
        if not self.agent_info.get("enabled"):
            raise ValueError("Ative a compatibilidade universal ao iniciar o servidor primeiro.")
        opencode_config = self._activate_opencode_global()
        with self.lock:
            self.agent_info["global_enabled"] = True
            self.agent_info["global_config"] = str(AGENT_ENV_FILE)
            self.agent_info["global_provider"] = "openai-compatible"
            self.agent_info["opencode_config"] = str(opencode_config)
            self.agent_info["global_error"] = ""
        return dict(self.agent_info)

    def _deactivate_agent_global(self) -> dict:
        """Disable universal mode and restore managed client adapters."""
        self._deactivate_opencode_global()
        with self.lock:
            self.agent_info["global_enabled"] = False
            self.agent_info["global_config"] = str(AGENT_ENV_FILE)
            self.agent_info["global_provider"] = "openai-compatible"
            self.agent_info["opencode_config"] = ""
            self.agent_info["global_error"] = ""
        return dict(self.agent_info)

    def set_agent_global(self, enabled: bool) -> dict:
        with self.lock:
            self.agent_global_default = "y" if enabled else "n"
            if enabled:
                # Keep the persisted/editable profile consistent with the
                # global toggle before taking the launch snapshot.
                self.agent_compat_default = "y"
                self.params["agent_compat"] = "y"
            running = self.is_running()
            final = dict(self.params)
            has_crono_mcp = str(final.get("mcp_native", "n")).lower() == "y"
            if enabled and not running:
                # The control is also useful before startup: remember the
                # desired mode so the next model session starts directly with
                # the client-neutral API.
                self.mcp_native_default = "n"
                self.params["mcp_native"] = "n"
                self.params["agent_global"] = "y"
                self.agent_restore_mcp = False
                self._save_settings()
                return self.process_snapshot()
            if enabled and has_crono_mcp:
                # A gateway already running cannot be converted in place: the
                # public port and the llama endpoint change together. Restart
                # through the same launcher path so activation is atomic.
                final["mcp_native"] = "n"
                self.agent_restore_mcp = True
                self.params["mcp_native"] = "n"
                self.mcp_native_default = "n"
                self._save_settings()
            elif enabled:
                self.agent_restore_mcp = False
        if enabled:
            if has_crono_mcp:
                self.stop_server()
                final["agent_global"] = "y"
                try:
                    return self.start_server(final)
                except Exception:
                    with self.lock:
                        self.agent_global_default = "n"
                        self.params["agent_global"] = "n"
                        self._save_settings()
                    raise
            try:
                result = self._activate_agent_global(final)
            except Exception:
                self.agent_global_default = "n"
                raise
        else:
            restore_mcp = self.agent_restore_mcp and running
            if restore_mcp:
                # Return to the exact gateway topology that existed before the
                # universal mode was enabled.
                final["mcp_native"] = "y"
                final["agent_global"] = "n"
                self.stop_server()
                self.agent_restore_mcp = False
                return self.start_server(final)
            result = self._deactivate_agent_global()
            self.agent_restore_mcp = False
        self._save_settings()
        return result

    @staticmethod
    def _agent_api_key(final: dict) -> str:
        direct = str(final.get("api_key", "") or "").strip()
        if direct:
            return direct
        key_file = str(final.get("api_key_file", "") or "").strip()
        if key_file:
            try:
                for line in Path(key_file).expanduser().read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        return line.strip()
            except OSError:
                pass
        return "local"

    def _write_agent_compatibility(
        self, final: dict, upstream_port: int, effective_context: int | None = None,
    ) -> dict:
        """Generate client-neutral metadata for the active llama-server."""
        if str(final.get("agent_compat", "n")).lower() != "y":
            self.agent_info = {
                "enabled": False,
                "endpoint": "",
                "model": "",
                "catalog": "",
                "context_window": 0,
                "auto_compact_token_limit": 0,
                "modalities": [],
                "reasoning_enabled": False,
                "supports_reasoning_effort": False,
                "capabilities": {},
                "server_capabilities": {},
                "native_tools": [],
                "agent_env": "",
                "agent_metadata": "",
                "opencode_config": "",
                "global_enabled": False,
                "global_config": str(AGENT_ENV_FILE),
                "global_provider": "openai-compatible",
                "global_error": "",
            }
            return self.agent_info

        agent_endpoint = self._local_api_root(
            final["host"], final["port"] if str(final.get("mcp_native")) == "y" else upstream_port
        )
        runtime_endpoint = self._local_api_root(final["host"], upstream_port)
        # The alias reported by /props is the identifier accepted by the
        # running server. It wins over a stale UI filename/alias after a
        # reload, while the configured value remains a fallback before READY.
        props = dict(self.runtime_effective.get("props") or {})
        model_id = str(props.get("model_alias") or self._agent_model_id(final)).strip()
        if not model_id:
            raise ValueError("Nao foi possivel determinar o model id para o agente local.")

        generation_settings = dict(props.get("default_generation_settings") or {})
        runtime_context = generation_settings.get("n_ctx") or effective_context
        context_window, compact_limit = self._agent_context_policy(
            int(runtime_context or final.get("ctx") or self._agent_native_context() or 1)
        )
        requested_reasoning = str(final.get("reasoning", "auto")).lower()
        reasoning_enabled = requested_reasoning != "off"
        chat_caps = dict(props.get("chat_template_caps") or {})
        runtime_params = generation_settings.get("params")
        if not isinstance(runtime_params, dict):
            runtime_params = generation_settings
        runtime_reasoning_format = str(
            runtime_params.get("reasoning_format") or ""
        ).lower()
        # ``reasoning_format=none`` is not a reliable capability flag.  In
        # llama.cpp it means that no special parser/extractor was selected for
        # the response; Qwen-style templates can still generate ``<think>``
        # and preserve it as ``reasoning_content``.  For automatic mode, use
        # the template capabilities and source as the evidence.  An explicit
        # UI ``off`` remains authoritative, while explicit ``on`` preserves
        # the user's request for templates that implement their own format.
        template_source = str(props.get("chat_template") or "").lower()
        template_has_think = "<think" in template_source
        template_supports_reasoning = bool(
            chat_caps.get("supports_preserve_reasoning")
            or chat_caps.get("supports_reasoning_effort")
            or template_has_think
            or getattr(self.meta, "supports_reasoning_preserve", False)
        )
        if requested_reasoning == "auto":
            reasoning_enabled = template_supports_reasoning
        capabilities, modalities, server_capabilities = (
            self._runtime_agent_capabilities(
                final, props, chat_caps, generation_settings, reasoning_enabled
            )
        )
        capabilities["runtime"]["context_window"] = context_window
        capabilities["evidence"]["context_window"] = (
            "runtime" if runtime_context else "configured_preview"
        )
        server_capabilities["reasoning_detection"] = {
            "requested": requested_reasoning,
            "runtime_format": runtime_reasoning_format,
            "template_supports_reasoning": template_supports_reasoning,
            "template_has_think_tags": template_has_think,
            "source": (
                "explicit_off" if requested_reasoning == "off"
                else "explicit_on" if requested_reasoning == "on"
                else "chat_template"
            ),
        }

        native_tools = []
        tools_probe = {
            "attempted": bool(props),
            "available": False,
            "error": "",
        }
        if props:
            try:
                tools_payload = self._fetch_runtime_payload(
                    runtime_endpoint.rstrip("/") + "/tools",
                    self._agent_api_key(final),
                )
                if not isinstance(tools_payload, list):
                    raise ValueError("/tools não retornou uma lista de ferramentas")
                native_tools = self._normalize_native_tools(tools_payload)
                tools_probe["available"] = True
            except Exception as exc:
                tools_probe["error"] = str(exc)
        server_capabilities["native_tools_endpoint"] = (
            runtime_endpoint.rstrip("/") + "/tools"
        )
        server_capabilities["native_tools"] = tools_probe
        server_capabilities["native_tool_count"] = len(native_tools)
        catalog_path, modalities = self._write_agent_catalog(
            final, model_id, context_window, compact_limit,
            modalities=modalities,
            capabilities=capabilities,
            server_capabilities=server_capabilities,
            native_tools=native_tools,
            reasoning_enabled=reasoning_enabled,
        )
        supports_reasoning_effort = bool(
            reasoning_enabled
            and (
                chat_caps.get("supports_reasoning_effort", False)
                or server_capabilities.get("supports_reasoning_budget", False)
            )
        )
        agent_env, agent_metadata = self._write_universal_agent_profile(
            agent_endpoint, model_id, context_window, compact_limit, modalities,
            reasoning_enabled, self._agent_api_key(final),
            capabilities=capabilities,
            server_capabilities=server_capabilities,
            native_tools=native_tools,
            props=props,
        )
        self.agent_info = {
            "enabled": True,
            "endpoint": agent_endpoint + "/v1",
            "model": model_id,
            "catalog": str(catalog_path),
            "context_window": context_window,
            "auto_compact_token_limit": compact_limit,
            "modalities": modalities,
            "reasoning_enabled": reasoning_enabled,
            "supports_reasoning_effort": supports_reasoning_effort,
            "capabilities": capabilities,
            "server_capabilities": server_capabilities,
            "native_tools": native_tools,
            "agent_env": str(agent_env),
            "agent_metadata": str(agent_metadata),
            "opencode_config": "",
            "global_enabled": False,
            "global_config": str(AGENT_ENV_FILE),
            "global_provider": "openai-compatible",
            "global_error": "",
        }
        return dict(self.agent_info)

    @staticmethod
    def _display_command(cmd: list[str]) -> str:
        hidden_after = {"--api-key", "--mcp-config", "--mcp-servers-json"}
        redacted = []
        hide_next = False
        for arg in cmd:
            if hide_next:
                redacted.append("<redacted>")
                hide_next = False
                continue
            redacted.append(str(arg))
            hide_next = str(arg) in hidden_after
        return shlex.join(redacted)

    def preview_command(self, raw: dict):
        final = self._coerce_final(raw)
        runtime = dict(final)
        gateway_enabled = final.get("mcp_native") == "y"
        if gateway_enabled:
            if final["port"] >= 65535:
                raise ValueError("A integracao Asael precisa da porta seguinte para o llama-server")
            runtime["port"] = final["port"] + 1
            try:
                native_config = json.loads(runtime["mcp_native_json"])
                native_env = native_config["mcpServers"]["crono-matrix"]["env"]
                native_env["CRONO_LLAMA_PORT"] = str(runtime["port"])
                runtime["mcp_native_json"] = json.dumps(
                    native_config, ensure_ascii=True, separators=(",", ":")
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Configuracao MCP nativa invalida: {exc}") from exc
        with self.lock:
            self.params.update({key: final[key] for key in self.params if key in final})
            cmd = self.opt.build_cmd(runtime)
        display = self._display_command(cmd)
        if gateway_enabled:
            display = (
                f"# API pública definida na interface: {final['host']}:{final['port']}\n"
                f"# llama-server interno usado pelo gateway: {runtime['host']}:{runtime['port']}\n"
                + display
            )
            gateway_cmd = [
                "env", f"CRONO_GATEWAY_HOST={final['host']}",
                f"CRONO_GATEWAY_PORT={final['port']}",
                f"CRONO_LLAMA_TARGET=http://{runtime['host']}:{runtime['port']}",
                native_config["mcpServers"]["crono-matrix"]["command"],
                str(ASAEL_GATEWAY_ENTRY),
            ]
            display += "\n" + shlex.join(gateway_cmd)
        if final.get("agent_compat") == "y":
            upstream_port = runtime["port"]
            display += (
                "\n# Agente local universal: apos iniciar, use "
                f"source {AGENT_ENV_FILE} "
                "ou configure qualquer cliente OpenAI-compatible com a API publicada."
            )
        return final, cmd, display

    def is_running(self):
        return bool(self.proc and self.proc.poll() is None)

    @staticmethod
    def _assert_port_available(host: str, port: int):
        bind_host = "127.0.0.1" if host == "localhost" else host
        family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError as exc:
                raise ValueError(
                    f"A porta {host}:{port} ja esta ocupada por outro processo. "
                    "Encerre o servidor anterior antes de iniciar outro."
                ) from exc

    def _validate_resident_memory_route(self, final: dict) -> None:
        """Reject resident modes that would recreate the ZRAM/OOM loop."""
        load_mode = str(final.get("load_mode", "mmap")).lower()
        if load_mode not in {"none", "dio", "mlock", "mmap+mlock"}:
            return
        with self.lock:
            if not self.opt or not self.meta:
                return
            opt = self.opt
            meta = self.meta
            hw = self.hardware

        reserve_mb = opt._ram_safety_reserve_mb()
        projector_mb = (
            int(meta.mmproj_size_mb)
            if str(final.get("omni", "n")).lower() == "y"
            and meta.mmproj_file else 0
        )
        fixed_host_mb = 1024 + int(opt.runtime_overhead_mb)
        try:
            n_cpu_moe = max(int(final.get("n_cpu_moe", opt.n_cpu_moe)), 0)
        except (TypeError, ValueError):
            n_cpu_moe = max(int(opt.n_cpu_moe), 0)
        expert_bytes = getattr(meta, "expert_weight_bytes_by_layer", [])
        if n_cpu_moe > 0 and expert_bytes:
            host_tensor_mb = math.ceil(
                sum(expert_bytes[:min(n_cpu_moe, len(expert_bytes))]) / 1048576
            )
        else:
            host_tensor_mb = max(int(opt.host_tensor_mb), 0)

        kv_host_mb = 0
        if str(final.get("kv_offload", "y")).lower() == "n":
            try:
                old_cache_v = opt.cache_v
                opt.cache_v = str(final.get("cache_v", opt.cache_v))
                kv_host_mb = math.ceil(
                    opt._cache_bytes_for_context(
                        int(final.get("ctx", opt.ctx)),
                        str(final.get("cache_k", opt.cache_k)),
                    ) / 1048576
                )
            finally:
                opt.cache_v = old_cache_v

        try:
            prompt_cache_mb = max(int(final.get("cache_ram", opt.cache_ram)), 0)
            checkpoint_count = max(
                int(final.get("ctx_checkpoints", opt.ctx_checkpoints)), 0
            )
        except (TypeError, ValueError):
            prompt_cache_mb = max(int(opt.prompt_cache_peak_mb), 0)
            checkpoint_count = max(int(opt.ctx_checkpoints), 0)
        checkpoint_mb = max(int(opt.checkpoint_snapshot_mb), 0) * checkpoint_count
        active_mb = (
            host_tensor_mb + projector_mb + kv_host_mb
            + fixed_host_mb + reserve_mb
        )
        resident_mb = (
            max(int(meta.size_mb), host_tensor_mb) + projector_mb + kv_host_mb
            + prompt_cache_mb + checkpoint_mb + fixed_host_mb + reserve_mb
        )

        if load_mode in {"mlock", "mmap+mlock"}:
            locked_mb = int(meta.size_mb) + projector_mb + reserve_mb
            if locked_mb > int(hw.ram_avail_mb):
                raise ValueError(
                    f"load_mode={load_mode} bloqueado: {locked_mb} MiB ficariam "
                    "sem possibilidade de paginação. Use mmap ou none com swap "
                    "NVMe prioritário."
                )
            return

        shortfall_mb = max(resident_mb - int(hw.ram_avail_mb), 0)
        if shortfall_mb <= 0:
            return
        nvme_free_mb = max(
            int(hw.swap_nvme_total_mb) - int(hw.swap_nvme_used_mb), 0
        )
        if active_mb > int(hw.ram_avail_mb):
            raise ValueError(
                f"load_mode={load_mode} recusado: conjunto ativo estimado "
                f"{active_mb} MiB excede os {hw.ram_avail_mb} MiB disponíveis; "
                "paginar experts/KV ativos destruiria o desempenho. Use mmap ou "
                "reduza estado host/contexto."
            )
        if not hw.swap_nvme_preferred:
            raise ValueError(
                f"load_mode={load_mode} precisa paginar cerca de {shortfall_mb} MiB, "
                f"mas o swap NVMe tem prioridade {hw.swap_nvme_priority} e a ZRAM "
                f"{hw.swap_zram_priority}. Reaplique o swap NVMe na telemetria "
                "antes de iniciar."
            )
        if nvme_free_mb < shortfall_mb + 1024:
            raise ValueError(
                f"load_mode={load_mode} precisa de {shortfall_mb} MiB de paginação "
                f"mais margem, mas o NVMe possui {nvme_free_mb} MiB livres. "
                "Aumente o swap dinâmico ou use mmap."
            )

    @staticmethod
    def _port_is_listening(host: str, port: int) -> bool:
        """Detect a listener owned outside this LauncherWebState instance.

        The launcher cannot infer the PID of an unrelated listener safely from
        a TCP socket alone. A connect probe is enough to prevent the UI from
        claiming IDLE while another llama-server is using the configured API
        port, without ever terminating a process it does not own.
        """
        connect_host = "127.0.0.1" if host in {"", "localhost", "0.0.0.0"} else host
        family = socket.AF_INET6 if ":" in connect_host else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.15)
                return sock.connect_ex((connect_host, int(port))) == 0
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _fetch_runtime_payload(url: str, api_key: str, timeout: float = 2.0) -> object:
        headers = {"Accept": "application/json"}
        if api_key and api_key != "local":
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _fetch_runtime_json(url: str, api_key: str, timeout: float = 2.0) -> dict:
        payload = LauncherWebState._fetch_runtime_payload(url, api_key, timeout)
        if not isinstance(payload, dict):
            raise ValueError(f"Resposta JSON invalida em {url}")
        return payload

    def _wait_for_server_ready(self, proc, final: dict, upstream_port: int) -> None:
        base_url = self._local_api_root(final["host"], upstream_port)
        api_key = self._agent_api_key(final)
        try:
            configured_timeout = int(final.get("timeout", 600) or 600)
        except (TypeError, ValueError):
            configured_timeout = 600
        deadline = time.monotonic() + max(60, min(configured_timeout, 900))
        last_error = "llama-server ainda nao respondeu"

        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            try:
                self._fetch_runtime_json(base_url + "/health", api_key)
                props = self._fetch_runtime_json(base_url + "/props", api_key)
                generation = props.get("default_generation_settings", {})
                effective_ctx = int(generation.get("n_ctx") or final.get("ctx") or 1)
                total_slots = int(props.get("total_slots") or final.get("parallel") or 1)
                # Apply the inference ceiling before publishing READY. This
                # prevents clients from racing the LOADING -> INFERENCE state
                # transition and makes UI telemetry reflect the real cgroup.
                self._tune_scope_for_inference(proc)
                with self.lock:
                    if proc is not self.proc or proc.poll() is not None:
                        return
                    memory_buffers = self.runtime_effective.get(
                        "memory_buffers", _runtime_memory_state()
                    )
                    self.runtime_effective = {
                        "ready": True,
                        "requested_context": int(final.get("ctx") or 0),
                        "context_window": effective_ctx,
                        "total_slots": total_slots,
                        "model_path": str(props.get("model_path") or final.get("model_path") or ""),
                        "props": props,
                        "memory_buffers": memory_buffers,
                    }
                    self.params["effective_ctx"] = effective_ctx
                    self.process_state = "running"
                    self.process_error = ""
                self._append_log(
                    f"[runtime confirmado: contexto={effective_ctx}, slots={total_slots}]\n",
                    "success",
                )
                self._log_runtime_memory_confirmation()
                try:
                    self._write_agent_compatibility(final, upstream_port, effective_ctx)
                    if final.get("agent_global") == "y":
                        self._activate_agent_global(final)
                        self._append_log(
                            "[modo universal ativo com contexto efetivo confirmado]\n",
                            "success",
                        )
                except Exception as exc:
                    with self.lock:
                        self.agent_info["global_error"] = str(exc)
                    self._append_log(f"[perfil universal nao publicado: {exc}]\n", "error")
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.25)

        with self.lock:
            if proc is not self.proc or proc.poll() is not None:
                return
            self.process_state = "error"
            self.process_error = f"llama-server nao ficou pronto: {last_error}"
        self._append_log(f"[{self.process_error}]\n", "error")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    def start_server(self, raw: dict):
        with self.lock:
            if self.is_running():
                raise ValueError("O servidor ja esta em execucao.")
        if not Path(self.llama_server).is_file():
            raise ValueError(f"llama-server nao encontrado: {self.llama_server}")
        build_usable, build_error = self._llama_build_usable(
            Path(self.llama_server)
        )
        if not build_usable:
            raise ValueError(
                "A compilação selecionada do llama.cpp não consegue iniciar: "
                + build_error
            )
        try:
            self._ensure_native_memory_guard()
        except OSError as exc:
            raise ValueError(
                f"Guard de memória C99 obrigatório indisponível: {exc}"
            ) from exc
        final, cmd, display = self.preview_command(raw)
        # The first preview applies all UI memory choices to ``self.opt``.
        # Grow/reprioritize the dedicated NVMe swap when this exact profile
        # depends on page-out, then rebuild and validate against the refreshed
        # kernel swap table.
        self.ensure_dynamic_swap()
        final, cmd, display = self.preview_command(raw)
        self._validate_resident_memory_route(final)
        available_before_mb = self._mem_available_mb()
        if available_before_mb and available_before_mb < SYSTEM_RAM_RESERVE_MB:
            raise ValueError(
                f"RAM disponível antes da carga: {available_before_mb} MiB; "
                f"o mínimo de segurança para iniciar é {SYSTEM_RAM_RESERVE_MB} MiB. "
                "Aguarde a paginação ou encerre outra carga antes de iniciar."
            )
        self._assert_port_available(final["host"], final["port"])
        gateway_enabled = final.get("mcp_native") == "y"
        upstream_port = final["port"] + 1 if gateway_enabled else final["port"]
        gateway_node = ""
        if gateway_enabled:
            self._assert_port_available(final["host"], upstream_port)
            if not ASAEL_GATEWAY_ENTRY.is_file():
                raise ValueError(f"Gateway Asael nao encontrado: {ASAEL_GATEWAY_ENTRY}")
            gateway_node, _gateway_version, gateway_error = _resolve_node_runtime()
            if not gateway_node:
                raise ValueError(_node_unavailable_message(gateway_error))
        if str(final.get("agent_compat", "n")).lower() != "y":
            self._write_agent_compatibility(final, upstream_port)
        else:
            with self.lock:
                self.agent_info.update({
                    "enabled": False,
                    "endpoint": "",
                    "model": "",
                    "context_window": 0,
                    "auto_compact_token_limit": 0,
                    "modalities": [],
                    "reasoning_enabled": False,
                    "supports_reasoning_effort": False,
                    "capabilities": {},
                    "server_capabilities": {},
                    "native_tools": [],
                    "agent_env": "",
                    "agent_metadata": "",
                    "catalog": "",
                    "global_enabled": False,
                    "global_error": "aguardando /health e /props",
                })
        with self.lock:
            self.mcp_native_default = final["mcp_native"]
            self.agent_compat_default = final["agent_compat"]
            self.agent_global_default = final["agent_global"]
            self.mcp_policy_default = final["mcp_policy"]
            self.mcp_workspace_default = str(final.get("mcp_workspace", self.mcp_workspace_default))
            self.mcp_snn_threads_default = final["mcp_snn_threads"]
            self.mcp_snn_steps_default = final["mcp_snn_steps"]
            self.mcp_repeat_limit_default = final["mcp_repeat_limit"]
        self._save_settings()
        with self.lock:
            self.process_state = "starting"
            self.process_error = ""
            self.runtime_effective = {
                "ready": False,
                "requested_context": int(final.get("ctx") or 0),
                "context_window": 0,
                "total_slots": 0,
                "model_path": str(final.get("model_path") or ""),
                "props": {},
                "memory_buffers": _runtime_memory_state(
                    "cpu" if str(final.get("kv_offload", "y")).lower() == "n"
                    else "gpu"
                ),
            }
            self.exit_code = None
            self.last_command = display
            self.mcp_state = "starting" if final.get("mcp_native") == "y" else "disabled"
            self.mcp_tools = 0
            self.mcp_error = ""
            self._append_log(f"$ {display}\n", "command")
            try:
                server_env = os.environ.copy()
                node_bin, _node_version, _node_error = _resolve_node_runtime()
                server_env.setdefault("LLAMA_PLAYWRIGHT_NODE", node_bin or "node")
                server_env.setdefault("LLAMA_PLAYWRIGHT_SCRIPT", str(LLAMA_PLAYWRIGHT_SCRIPT))
                server_env.setdefault("LLAMA_PLAYWRIGHT_MODULE", str(LLAMA_PLAYWRIGHT_MODULE))
                (
                    launch_cmd, scope_unit, memory_high_mb, scope_headroom_mb,
                ) = self._scoped_llama_command(
                    cmd, available_before_mb, final.get("load_mode", "mmap")
                )
                self.proc = subprocess.Popen(
                    launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=True, env=server_env,
                )
                self.llama_scope_unit = scope_unit
                self.memory_guard.update({
                    "scope_unit": f"{scope_unit}.scope" if scope_unit else "",
                    "memory_high_mb": memory_high_mb,
                    "scope_headroom_mb": scope_headroom_mb,
                    "scope_phase": "loading" if scope_unit else "fallback",
                })
            except Exception as exc:
                self.process_state = "error"
                self.process_error = str(exc)
                raise
            if gateway_enabled:
                gateway_env = os.environ.copy()
                gateway_env.update({
                    "CRONO_GATEWAY_HOST": str(final["host"]),
                    "CRONO_GATEWAY_PORT": str(final["port"]),
                    "CRONO_LLAMA_HOST": str(final["host"]),
                    "CRONO_LLAMA_PORT": str(upstream_port),
                    "CRONO_LLAMA_TARGET": f"http://{final['host']}:{upstream_port}",
                    "CRONO_LLAMA_PATH": self.llama_server,
                    "CRONO_MODELS_DIR": self.models_dir,
                    "CRONO_PROJECT_ROOT": str(PROJECT_ROOT),
                    "CRONO_WORKSPACE": str(final.get("mcp_workspace", NATIVE_MCP_WORKSPACE)),
                    "CRONO_SNN_THREADS": str(final["mcp_snn_threads"]),
                    "CRONO_SNN_STEPS": str(final["mcp_snn_steps"]),
                     "CRONO_SNN_TIMEOUT_MS": "15000",
                     "CRONO_MEMORY_DIR": str(SNN_DIR.parent),
                     "CRONO_SNN_ENABLED_FILE": str(SNN_ENABLED_FILE),
                     "LOG_LEVEL": "info",
                })
                try:
                    self.gateway_proc = subprocess.Popen(
                        [gateway_node, str(ASAEL_GATEWAY_ENTRY)],
                        cwd=str(NATIVE_MCP_DIR), env=gateway_env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", bufsize=1,
                        start_new_session=True,
                    )
                except Exception:
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    self.proc = None
                    raise
            self.started_at = time.time()
            proc = self.proc
            scope_unit = self.llama_scope_unit
        if scope_unit:
            if self._wait_for_process_scope(proc, scope_unit):
                self._append_log(
                    f"[cgroup dedicado: {scope_unit}.scope; "
                    f"MemoryHigh={'max' if not self.memory_guard['memory_high_mb'] else str(self.memory_guard['memory_high_mb']) + ' MiB'}; "
                    f"headroom de carga={scope_headroom_mb} MiB; "
                    f"load-mode={final.get('load_mode', 'mmap')}]\n",
                    "success",
                )
            else:
                self._append_log(
                    f"[cgroup dedicado não confirmado: {scope_unit}.scope]\n",
                    "warning",
                )
        self._start_memory_guard(proc)
        threading.Thread(target=self._read_output, args=(proc,), daemon=True).start()
        if self.gateway_proc:
            threading.Thread(
                target=self._read_gateway_output, args=(self.gateway_proc,), daemon=True
            ).start()
        threading.Thread(
            target=self._wait_for_server_ready,
            args=(proc, dict(final), upstream_port),
            daemon=True,
        ).start()
        return self.process_snapshot()

    def _read_gateway_output(self, proc):
        try:
            if proc.stdout:
                for line in proc.stdout:
                    level = "success" if "CRONO_ASAEL_GATEWAY_READY" in line else (
                        "error" if any(value in line.lower() for value in ("error", "failed", "fatal")) else ""
                    )
                    self._append_log(f"[asael] {line}", level)
        except (OSError, ValueError) as exc:
            self._append_log(f"[asael] Erro ao ler saida: {exc}\n", "error")
        rc = proc.wait()
        with self.lock:
            if proc is not self.gateway_proc:
                return
            self.gateway_proc = None
            llama_proc = self.proc
            expected_stop = self.process_state == "stopping"
            if not expected_stop and llama_proc and llama_proc.poll() is None:
                self.process_state = "error"
                self.process_error = f"Gateway Asael encerrado com codigo {rc}"
                try:
                    os.killpg(os.getpgid(llama_proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def _append_log(self, line: str, level: str = ""):
        with self.lock:
            self.log_seq += 1
            self.logs.append({"seq": self.log_seq, "line": line, "level": level})

    def _capture_runtime_buffer(self, line: str) -> None:
        """Record the buffers allocated by llama.cpp, not planner guesses."""
        match = _RUNTIME_BUFFER_RE.search(line)
        if not match:
            return
        device = match.group("device").strip()
        kind = match.group("kind").lower()
        size_mb = float(match.group("size"))
        is_cpu = device.upper().startswith("CPU")
        with self.lock:
            memory = self.runtime_effective.setdefault(
                "memory_buffers", _runtime_memory_state()
            )
            bucket = memory.setdefault(kind, _runtime_memory_state()[kind])
            devices = bucket.setdefault("devices", {})
            devices[device] = round(float(devices.get(device, 0.0)) + size_mb, 2)
            bucket["total_mb"] = round(float(bucket.get("total_mb", 0.0)) + size_mb, 2)
            target = "cpu_mb" if is_cpu else "gpu_mb"
            bucket[target] = round(float(bucket.get(target, 0.0)) + size_mb, 2)
            gpu_mb = float(bucket.get("gpu_mb", 0.0))
            cpu_mb = float(bucket.get("cpu_mb", 0.0))
            bucket["placement"] = (
                "hybrid" if gpu_mb > 0 and cpu_mb > 0
                else "gpu" if gpu_mb > 0
                else "cpu" if cpu_mb > 0
                else "pending"
            )
            requested = str(memory.get("requested", "unknown"))
            bucket["confirmed"] = (
                gpu_mb > 0 if requested == "gpu"
                else cpu_mb > 0 if requested == "cpu"
                else None
            )

    def _log_runtime_memory_confirmation(self) -> None:
        with self.lock:
            memory = self.runtime_effective.get("memory_buffers", {})
            requested = str(memory.get("requested", "unknown"))
            snapshots = {
                kind: dict(memory.get(kind, {})) for kind in ("kv", "rs")
            }
        observed = []
        mismatch = False
        for label, kind in (("KV", "kv"), ("RS", "rs")):
            bucket = snapshots[kind]
            if float(bucket.get("total_mb", 0.0)) <= 0:
                continue
            devices = ", ".join(
                f"{name} {size:.2f} MiB"
                for name, size in bucket.get("devices", {}).items()
            )
            observed.append(f"{label}: {devices}")
            if bucket.get("confirmed") is False:
                mismatch = True
        if not observed:
            self._append_log(
                "[colocação KV/RS não verificada: o llama.cpp não informou os buffers]\n",
                "warning",
            )
            return
        level = "error" if mismatch else "success"
        status = "DIVERGENTE" if mismatch else "CONFIRMADA"
        self._append_log(
            f"[memória de contexto {status}: solicitado={requested}; "
            + "; ".join(observed) + "]\n",
            level,
        )

    def _read_output(self, proc):
        try:
            if proc.stdout:
                for line in proc.stdout:
                    self._capture_runtime_buffer(line)
                    lower = line.lower()
                    mcp_ready = re.search(r"MCP warmup: 'crono-matrix' discovered (\d+) tools", line)
                    if mcp_ready:
                        with self.lock:
                            self.mcp_state = "ready"
                            self.mcp_tools = int(mcp_ready.group(1))
                            self.mcp_error = ""
                    elif "mcp starting failed" in lower or (
                        "mcp" in lower and ("failed to spawn" in lower or "failed to start" in lower)
                    ):
                        with self.lock:
                            self.mcp_state = "error"
                            self.mcp_error = line.strip()
                    if any(value in lower for value in ("error", "failed", "fatal", "errno")):
                        level = "error"
                    elif "warn" in lower:
                        level = "warning"
                    elif any(value in lower for value in ("listening", "model loaded", "server is ready")):
                        level = "success"
                    else:
                        level = ""
                    self._append_log(line, level)
        except (OSError, ValueError) as exc:
            self._append_log(f"Erro ao ler saida: {exc}\n", "error")
        rc = proc.wait()
        with self.lock:
            if proc is self.proc:
                gateway_proc = self.gateway_proc
                self.gateway_proc = None
                self.proc = None
                self.exit_code = rc
                self.process_state = "idle" if rc == 0 else "error"
                self.runtime_effective["ready"] = False
                if rc:
                    self.process_error = f"Processo encerrado com codigo {rc}"
                if self.mcp_state not in {"disabled", "error"}:
                    self.mcp_state = "stopped"
            else:
                gateway_proc = None
        if gateway_proc and gateway_proc.poll() is None:
            try:
                os.killpg(os.getpgid(gateway_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._stop_memory_guard()
        if self.agent_info.get("global_enabled"):
            try:
                self._deactivate_agent_global()
            except Exception as exc:
                with self.lock:
                    self.agent_info["global_error"] = str(exc)
                self._append_log(f"[modo universal continua ativo: {exc}]\n", "error")
        self._append_log(f"[processo encerrado: {rc}]\n", "warning" if rc else "")

    def stop_server(self):
        with self.lock:
            proc = self.proc
            gateway_proc = self.gateway_proc
            if not proc or proc.poll() is not None:
                self.proc = None
                self.gateway_proc = None
                self.process_state = "idle"
                self.runtime_effective["ready"] = False
                if gateway_proc and gateway_proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(gateway_proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if self.agent_info.get("global_enabled"):
                    try:
                        self._deactivate_agent_global()
                    except Exception as exc:
                        with self.lock:
                            self.agent_info["global_error"] = str(exc)
                return self.process_snapshot()
            self.process_state = "stopping"
            self.runtime_effective["ready"] = False
        self._stop_memory_guard()
        if gateway_proc and gateway_proc.poll() is None:
            try:
                os.killpg(os.getpgid(gateway_proc.pid), signal.SIGTERM)
                gateway_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(gateway_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.agent_info.get("global_enabled"):
            try:
                self._deactivate_agent_global()
            except Exception as exc:
                with self.lock:
                    self.agent_info["global_error"] = str(exc)
        return self.process_snapshot()

    def process_snapshot(self):
        with self.lock:
            running = self.is_running()
            try:
                public_port = int(self.params.get("port", 8080))
            except (TypeError, ValueError):
                public_port = 8080
            params = dict(self.params)
            host = params.get("host", "127.0.0.1")
            external_conflict = (
                not running
                and self.process_state in {"idle", "error"}
                and self._port_is_listening(host, public_port)
            )
            ready = running and self.process_state == "running" and bool(
                self.runtime_effective.get("ready")
            )
            gateway_enabled = str(params.get("mcp_native", "n")).lower() == "y"
            state = "external" if external_conflict else self.process_state
            error = self.process_error
            if external_conflict and not error:
                error = (
                    f"A porta {host}:{public_port} está ocupada por um processo externo. "
                    "Verifique o llama-server/VRAM antes de iniciar outro."
                )
            return {
                "state": state,
                "running": running,
                "ready": ready,
                "pid": self.proc.pid if running else None,
                "model": Path(self.model_path).name if self.model_path else "",
                "host": host,
                "port": public_port,
                "public_port": public_port,
                "upstream_port": (
                    public_port + 1 if gateway_enabled else public_port
                ),
                "gateway_enabled": gateway_enabled,
                "started_at": self.started_at,
                "exit_code": self.exit_code,
                "last_command": getattr(self, "last_command", ""),
                "error": error,
                "external_conflict": external_conflict,
                "mcp_state": self.mcp_state,
                "mcp_tools": self.mcp_tools,
                "mcp_error": self.mcp_error,
                "runtime_effective": dict(self.runtime_effective),
                "memory_guard": dict(self.memory_guard),
                "agent_global_default": self.agent_global_default == "y",
                "agent_compat": dict(self.agent_info),
            }

    def evaluation_context(self):
        parameter_keys = (
            "ctx", "effective_ctx", "ngl", "parallel", "cache_k", "cache_v", "kv_unified",
            "kv_offload", "flash", "batch", "ubatch", "threads",
            "threads_batch", "device", "split_mode", "numa", "mlock",
            "no_mmap", "load_mode", "tensor_read_lazy", "cache_reuse",
            "slot_similarity", "n_cpu_moe", "n_cpu_ffn",
            "temp", "top_k", "top_p", "min_p", "repeat_penalty", "seed",
            "reasoning", "reasoning_budget", "spec_type", "spec_draft_n_max",
            "fit", "fit_target", "fit_ctx", "omni", "mmproj_offload",
            "no_mmproj_auto", "image_min_tokens", "image_max_tokens",
        )
        with self.lock:
            parameters = {key: self.params.get(key) for key in parameter_keys}
            model = self.model_snapshot()
            process = self.process_snapshot()
        vision_enabled = bool(
            model
            and model.get("mmproj_valid")
            and model.get("mmproj")
            and parameters.get("omni") == "y"
        )
        return {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "parameters": parameters,
            "process": process,
            "hardware": self.hardware_snapshot(),
            "llama_server": self.llama_server,
            "capabilities": {
                "vision": {
                    "enabled": vision_enabled,
                    "mmproj": model.get("mmproj", "") if model else "",
                    "reason": (
                        "mmproj valido e multimodal habilitado"
                        if vision_enabled
                        else "modelo sem mmproj valido ou multimodal desabilitado"
                    ),
                },
            },
        }

    def logs_after(self, sequence: int):
        with self.lock:
            return [dict(item) for item in self.logs if item["seq"] > sequence]

    def search_hf(self, term: str):
        direct = self.hf.search_by_url_or_id(term.strip())
        if direct:
            info = self.hf.model_info(direct["user"], direct["repo"], direct.get("revision", "main"))
            return [info]
        return self.hf.search(term.strip(), limit=30)

    # ── Radar Hugging Face ──────────────────────────────────────────────────
    # O radar não tenta adivinhar se um modelo "cabe" na máquina sem ter o GGUF
    # e os metadados reais. Ele apenas prioriza lançamentos relevantes e deixa a
    # seleção continuar pelo fluxo normal, que lê o GGUF antes de recomendar
    # parâmetros ou iniciar download.
    @staticmethod
    def _normalize_hf_watchlist(value) -> str:
        raw = str(value or "")
        entries = []
        for item in re.split(r"[,;\n]+", raw.lower()):
            token = re.sub(r"\s+", "", item.strip())
            if 1 <= len(token) <= 64 and token not in entries:
                entries.append(token)
        if not entries:
            entries = HF_RADAR_DEFAULT_WATCHLIST.split(",")
        return ",".join(entries[:16])

    @staticmethod
    def _normalize_hf_radar_records(value) -> dict:
        if not isinstance(value, dict):
            return {}
        records = {}
        for repo_id, raw in value.items():
            repo_id = str(repo_id or "").strip()
            if "/" not in repo_id or len(repo_id) > 256:
                continue
            if isinstance(raw, dict):
                item = {
                    key: raw.get(key)
                    for key in (
                        "id", "last_modified", "created_at", "downloads", "likes",
                        "pipeline_tag", "tags", "family", "official", "capabilities",
                        "trusted", "score", "relevance", "event",
                    )
                }
                item["id"] = repo_id
            else:
                item = {"id": repo_id, "last_modified": str(raw or "")}
            item["last_modified"] = str(item.get("last_modified") or "")
            item["tags"] = [str(tag)[:80] for tag in item.get("tags", [])[:16]] \
                if isinstance(item.get("tags"), list) else []
            item["capabilities"] = [str(tag)[:32] for tag in item.get("capabilities", [])[:8]] \
                if isinstance(item.get("capabilities"), list) else []
            records[repo_id] = item
        return records

    @staticmethod
    def _hf_radar_is_model_like(text: str) -> bool:
        return any(token in text for token in (
            "text-generation", "conversational", "transformers", "llama",
            "qwen", "gemma", "glm", "mistral", "deepseek", "nemotron",
            "gpt-oss", "mixture-of-experts", "language-model",
        ))

    def _hf_radar_classify(self, source: dict, watch_terms: list[str]) -> dict | None:
        if not isinstance(source, dict):
            return None
        repo_id = str(source.get("id") or "").strip()
        if "/" not in repo_id or len(repo_id) > 256:
            return None
        raw_tags = source.get("tags") if isinstance(source.get("tags"), list) else []
        tags = [str(tag).lower() for tag in raw_tags if isinstance(tag, str)]
        haystack = " ".join((repo_id.lower(), str(source.get("pipeline_tag") or "").lower(), *tags))
        family = next((term for term in watch_terms if term in haystack), "")
        if not family and not self._hf_radar_is_model_like(haystack):
            return None

        owner = repo_id.split("/", 1)[0].lower()
        gguf = "gguf" in haystack or "llama.cpp" in haystack
        moe = bool(re.search(r"\bmoe\b|mixture-of-experts|a\d+b", haystack))
        vision = any(token in haystack for token in (
            "vision-language", "vision", "multimodal", "image-to-text", "vlm",
        ))
        code = any(token in haystack for token in ("coder", "coding", "codegen", "code-"))
        agent = "agent" in haystack or "tool-use" in haystack
        reasoning = "reasoning" in haystack or "chain-of-thought" in haystack or "\bcot\b" in haystack
        capabilities = [
            label for label, enabled in (
                ("GGUF", gguf), ("MoE", moe), ("VISÃO", vision),
                ("CÓDIGO", code), ("AGENTE", agent), ("REASONING", reasoning),
            ) if enabled
        ]
        score = 0
        if owner in HF_RADAR_OFFICIAL_OWNERS:
            score += 100
        score += HF_TRUSTED.get(owner, 0) * 18
        score += 36 if gguf else 0
        score += 22 if moe else 0
        score += 14 if code else 0
        score += 12 if agent else 0
        score += 8 if vision else 0
        score += min(15, int(source.get("likes") or 0) // 20)
        score += min(12, int(source.get("downloads") or 0) // 10_000)
        relevance = "ALTA" if score >= 100 else "MÉDIA" if score >= 45 else "RADAR"
        return {
            "id": repo_id,
            "last_modified": str(source.get("lastModified") or source.get("createdAt") or ""),
            "created_at": str(source.get("createdAt") or ""),
            "downloads": int(source.get("downloads") or 0),
            "likes": int(source.get("likes") or 0),
            "pipeline_tag": str(source.get("pipeline_tag") or "modelo"),
            "tags": tags[:16],
            "family": family or "geral",
            "official": owner in HF_RADAR_OFFICIAL_OWNERS,
            "trusted": owner in HF_TRUSTED,
            "capabilities": capabilities,
            "score": score,
            "relevance": relevance,
            "sources": [],
        }

    @staticmethod
    def _hf_radar_trim(records: dict, maximum=HF_RADAR_MAX_SEEN) -> dict:
        ordered = sorted(
            records.items(),
            key=lambda item: str(item[1].get("last_modified") or ""),
            reverse=True,
        )
        return dict(ordered[:maximum])

    def hf_radar_snapshot(self) -> dict:
        with self.hf_radar_lock:
            return {
                "enabled": self.hf_radar_enabled,
                "watchlist": self.hf_radar_watchlist,
                "initialized": self.hf_radar_initialized,
                "last_refresh": self.hf_radar_last_refresh,
                "error": self.hf_radar_last_error,
                "refreshing": bool(getattr(self, "hf_radar_refreshing", False)),
                "unread_count": len(self.hf_radar_unread),
                "items": [dict(item) for item in self.hf_radar_items],
            }

    def refresh_hf_radar(self, force=False) -> dict:
        now_monotonic = time.monotonic()
        with self.hf_radar_lock:
            if not self.hf_radar_enabled:
                return self.hf_radar_snapshot()
            if getattr(self, "hf_radar_refreshing", False):
                return self.hf_radar_snapshot()
            if (
                not force
                and self.hf_radar_items
                and now_monotonic - self.hf_radar_refresh_monotonic < HF_RADAR_MIN_REFRESH_SECONDS
            ):
                return self.hf_radar_snapshot()
            self.hf_radar_refreshing = True
            watch_terms = self.hf_radar_watchlist.split(",")

        raw_items = []
        errors = []
        try:
            for term in watch_terms:
                try:
                    for item in self.hf.latest_models(
                        search=term, pipeline_tag="text-generation", limit=18,
                    ):
                        raw_items.append((f"família:{term}", item))
                except Exception as exc:
                    errors.append(f"{term}: {exc}")
            try:
                for item in self.hf.latest_models(filter_tag="gguf", limit=60):
                    raw_items.append(("GGUF", item))
            except Exception as exc:
                errors.append(f"GGUF: {exc}")

            if not raw_items:
                raise ValueError("Radar não recebeu modelos da API do Hugging Face.")
            merged = {}
            for source_name, raw in raw_items:
                item = self._hf_radar_classify(raw, watch_terms)
                if not item:
                    continue
                current = merged.get(item["id"])
                if current is None:
                    item["sources"] = [source_name]
                    merged[item["id"]] = item
                else:
                    current["sources"] = sorted(set(current["sources"] + [source_name]))
                    current["score"] = max(current["score"], item["score"])
                    current["capabilities"] = sorted(set(current["capabilities"] + item["capabilities"]))
                    current["official"] = current["official"] or item["official"]

            with self.hf_radar_lock:
                first_scan = not self.hf_radar_initialized
                seen = dict(self.hf_radar_seen)
                unread = dict(self.hf_radar_unread)
                for repo_id, item in merged.items():
                    previous = seen.get(repo_id, {})
                    previous_modified = str(previous.get("last_modified") or "")
                    current_modified = item["last_modified"]
                    event = ""
                    if not first_scan:
                        if not previous:
                            event = "NOVO"
                        elif current_modified and current_modified > previous_modified:
                            event = "ATUALIZADO"
                    if event:
                        alert = dict(item)
                        alert["event"] = event
                        unread[repo_id] = alert
                    seen[repo_id] = {"id": repo_id, "last_modified": current_modified}

                for repo_id, item in merged.items():
                    if repo_id in unread:
                        item["unread"] = True
                        item["event"] = unread[repo_id].get("event", "NOVO")
                    else:
                        item["unread"] = False
                        item["event"] = ""
                # Mantém um aviso mesmo quando ele sai da primeira página da API.
                for repo_id, item in unread.items():
                    if repo_id not in merged:
                        carried = dict(item)
                        carried["unread"] = True
                        merged[repo_id] = carried

                self.hf_radar_seen = self._hf_radar_trim(seen)
                self.hf_radar_unread = self._hf_radar_trim(unread, 80)
                self.hf_radar_initialized = True
                self.hf_radar_last_refresh = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self.hf_radar_last_error = "; ".join(errors[:3])
                self.hf_radar_items = sorted(
                    merged.values(),
                    key=lambda item: (
                        not bool(item.get("unread")),
                        not bool(item.get("official")),
                        not bool(item.get("trusted")),
                        -int(item.get("score") or 0),
                        str(item.get("last_modified") or ""),
                    ),
                )[:48]
                self.hf_radar_refresh_monotonic = time.monotonic()
                self._save_settings()
                return self.hf_radar_snapshot()
        except Exception as exc:
            with self.hf_radar_lock:
                self.hf_radar_last_error = str(exc)
                return self.hf_radar_snapshot()
        finally:
            with self.hf_radar_lock:
                self.hf_radar_refreshing = False

    def set_hf_radar_preferences(self, watchlist: str, enabled: bool) -> dict:
        with self.hf_radar_lock:
            self.hf_radar_watchlist = self._normalize_hf_watchlist(watchlist)
            self.hf_radar_enabled = bool(enabled)
            self.hf_radar_refresh_monotonic = 0.0
            self._save_settings()
            return self.hf_radar_snapshot()

    def mark_hf_radar_read(self) -> dict:
        with self.hf_radar_lock:
            self.hf_radar_unread = {}
            for item in self.hf_radar_items:
                item["unread"] = False
                item["event"] = ""
            self._save_settings()
            return self.hf_radar_snapshot()

    def _resolve_equivalent(self, user: str, repo: str):
        base_repo = _hf_base_model_name(repo)
        term = f"{base_repo} GGUF"
        url = f"{HF_API}/models?search={urllib.parse.quote(term)}&full=true&sort=likes&limit=100"
        data = _hf_fetch_json(url)
        owner_tokens = re.findall(r"[a-z0-9]+", user.lower())
        model_tokens = re.findall(r"[a-z0-9]+", base_repo.lower())
        overlap = 0
        for size in range(1, min(len(owner_tokens), len(model_tokens)) + 1):
            if owner_tokens[-size:] == model_tokens[:size]:
                overlap = size
        exact = {"".join(model_tokens), "".join(owner_tokens + model_tokens[overlap:])}
        canonical = f"{user}/{base_repo}".lower()
        candidates = []
        for item in data:
            repo_id = item.get("id", "")
            if "/" not in repo_id:
                continue
            files = [s for s in item.get("siblings", []) if s.get("rfilename", "").lower().endswith(".gguf")]
            if not files:
                continue
            candidate_user, candidate_name = repo_id.split("/", 1)
            candidate_norm = re.sub(r"[^a-z0-9]", "", _hf_base_model_name(candidate_name).lower())
            if candidate_norm not in exact:
                continue
            quantized_bases = {
                tag[len("base_model:quantized:"):].lower()
                for tag in item.get("tags", []) if isinstance(tag, str)
                and tag.lower().startswith("base_model:quantized:")
            }
            if quantized_bases and canonical not in quantized_bases:
                continue
            official = 100 if candidate_user.lower() == user.lower() else 0
            score = official + HF_TRUSTED.get(candidate_user, 0) * 10 + len(files) / 10
            candidates.append((score, candidate_user, candidate_name))
        candidates.sort(reverse=True)
        return (candidates[0][1], candidates[0][2]) if candidates else None

    def hf_details(self, repo_id: str):
        if "/" not in repo_id:
            raise ValueError("Repositorio invalido")
        original = repo_id
        user, repo = repo_id.split("/", 1)
        info = self.hf.model_info(user, repo)
        files = [s for s in info.get("siblings", []) if s.get("rfilename", "").lower().endswith(".gguf")]
        if not files:
            resolved = self._resolve_equivalent(user, repo)
            if not resolved:
                return {"repo_id": repo_id, "original": original, "files": [], "meta": {}, "info": info}
            user, repo = resolved
            repo_id = f"{user}/{repo}"
            info = self.hf.model_info(user, repo)
            files = [s for s in info.get("siblings", []) if s.get("rfilename", "").lower().endswith(".gguf")]
        try:
            url = (f"{HF_API}/models/{urllib.parse.quote(user, safe='')}/"
                   f"{urllib.parse.quote(repo, safe='')}?expand%5B%5D=gguf")
            metadata = _hf_fetch_json(url).get("gguf", {}) or {}
        except Exception:
            metadata = {}
        display_files = []
        seen = set()
        file_map = {item["rfilename"]: item for item in files}
        for item in files:
            name = item["rfilename"]
            match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", name, re.I)
            if match:
                key = match.group(1)
                if key in seen:
                    continue
                seen.add(key)
                count = int(match.group(3))
                parts = [f"{key}-{index:05d}-of-{count:05d}.gguf" for index in range(1, count + 1)]
                size = sum(int(file_map.get(part, {}).get("size", 0)) for part in parts)
                key_path = Path(key)
                label = key_path.parent.name if key_path.parent != Path(".") else key_path.name
                display_files.append({"name": parts[0], "label": label, "size": size, "parts": count})
            else:
                display_files.append({"name": name, "label": Path(name).stem, "size": int(item.get("size", 0)), "parts": 1})
        return {
            "repo_id": repo_id, "original": original, "files": display_files,
            "meta": metadata, "info": info,
        }

    def start_download(self, repo_id: str, filename: str):
        with self.lock:
            if self.download["state"] == "running":
                raise ValueError("Ja existe um download em andamento.")
            self.download = {
                "state": "running", "downloaded": 0, "total": 0, "speed": 0,
                "filename": filename, "error": "", "paths": [],
            }
            self.download_cancel.clear()
        thread = threading.Thread(
            target=self._download_worker, args=(repo_id, filename), daemon=True
        )
        thread.start()
        return self.download_snapshot()

    def _download_worker(self, repo_id: str, filename: str):
        client = HuggingFaceHub()
        try:
            user, repo = repo_id.split("/", 1)
            info = client.model_info(user, repo)
            all_files = {item["rfilename"]: item for item in info.get("siblings", [])}
            match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", filename, re.I)
            if match:
                count = int(match.group(3))
                names = [f"{match.group(1)}-{index:05d}-of-{count:05d}.gguf" for index in range(1, count + 1)]
            else:
                names = [filename]
            missing = [name for name in names if name not in all_files]
            if missing:
                raise ValueError("Repositorio multipartes incompleto")
            total_all = sum(int(all_files[name].get("size", 0)) for name in names)
            completed = 0
            paths = []
            for name in names:
                item = all_files[name]
                def progress(downloaded, _total, speed, base=completed):
                    with self.lock:
                        self.download.update({
                            "downloaded": base + downloaded,
                            "total": total_all,
                            "speed": speed,
                        })
                path = client.download(
                    user, repo, name, self.models_dir, on_progress=progress,
                    expected_size=int(item.get("size", 0)),
                    expected_sha256=(item.get("lfs") or {}).get("sha256", ""),
                    cancel_event=self.download_cancel,
                )
                remote_sha = str(
                    (item.get("lfs") or {}).get("sha256")
                    or item.get("sha256") or ""
                ).lower()
                self._write_model_origin(path, {
                    "source": "crono-huggingface",
                    "repo_id": repo_id,
                    "revision": "main",
                    "filename": name,
                    "commit": str(info.get("sha") or info.get("commit") or ""),
                    "remote_size": int(item.get("size", 0) or 0),
                    "remote_sha256": remote_sha,
                    "remote_last_modified": str(info.get("lastModified") or ""),
                    "downloaded_size": Path(path).stat().st_size,
                    "downloaded_sha256": remote_sha or self._sha256_file(path),
                    "downloaded_at": self._utc_now(),
                    "last_checked_at": self._utc_now(),
                })
                completed += int(item.get("size", 0))
                paths.append(path)
            with self.lock:
                self.download.update({
                    "state": "done", "downloaded": total_all, "total": total_all,
                    "paths": paths,
                })
            self.scan_models()
        except Exception as exc:
            with self.lock:
                self.download.update({
                    "state": "cancelled" if self.download_cancel.is_set() else "error",
                    "error": str(exc),
                })

    def cancel_download(self):
        self.download_cancel.set()
        self.hf.cancel_downloads()
        with self.lock:
            if self.download["state"] == "running":
                self.download["state"] = "cancelling"
        return self.download_snapshot()

    def download_snapshot(self):
        with self.lock:
            return dict(self.download)


class EvalRunner:
    def __init__(self):
        import threading
        self.lock = threading.RLock()
        self.proc = None
        self.state = "idle"
        self.error = ""
        self.logs = deque(maxlen=5000)
        self.log_seq = 0
        self.axes_filter = ""
        self.effective_axes_filter = ""
        self.repeats = 1
        self.seed = 0
        self.scale = "auto"
        self.mode = "auto"
        self.reasoning_effort = "default"
        self.reasoning_budget = "auto"
        self.sampling = "server"
        self.temperature = 0.6
        self.top_k = 20
        self.top_p = 0.95
        self.min_p = 0.05
        self.repeat_penalty = 1.0
        self.max_tokens = 16384
        self.timeout = 300
        self.xctx_scale = 1.0
        self.os_filter = "all"
        self.judge_url = ""
        self.judge_model = ""
        self.api_url = "http://127.0.0.1:8080/v1/chat/completions"
        self.progress = {
            "current": 0, "total": 0, "passed": 0, "failed": 0,
            "skipped": 0, "axis": "", "test": "", "axes": {},
        }
        self.stop_requested = False
        self.results_file = ""
        self.dashboard_data = None
        self.dashboard_timestamp = ""
        self.history_load_attempted = False
        project_root = Path(__file__).resolve().parent.parent
        self.runs_dir = project_root / "eval_history" / "web_runs"
        self.dashboard_output_dir = project_root / "eval_dashboard" / "data"
        self.runtime_context = {}

    def snapshot(self):
        with self.lock:
            progress = dict(self.progress)
            progress["axes"] = {
                axis: dict(values)
                for axis, values in self.progress.get("axes", {}).items()
            }
            return {
                "state": self.state,
                "error": self.error,
                "progress": progress,
                "results_file": self.results_file,
                "dashboard_timestamp": self.dashboard_timestamp,
                "runtime": json.loads(json.dumps(self.runtime_context, default=str)),
                "config": {
                    "axes_filter": self.axes_filter,
                    "effective_axes_filter": self.effective_axes_filter,
                    "repeats": self.repeats,
                    "seed": self.seed,
                    "scale": self.scale,
                    "mode": self.mode,
                    "reasoning_effort": self.reasoning_effort,
                    "reasoning_budget": self.reasoning_budget,
                    "sampling": self.sampling,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "min_p": self.min_p,
                    "repeat_penalty": self.repeat_penalty,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout,
                    "xctx_scale": self.xctx_scale,
                    "os_filter": self.os_filter,
                    "judge_url": self.judge_url,
                    "judge_model": self.judge_model,
                    "api_url": self.api_url,
                },
            }

    def logs_after(self, seq):
        with self.lock:
            return [item for item in self.logs if item["seq"] > seq]

    def _emit(self, level, line):
        with self.lock:
            self.log_seq += 1
            self.logs.append({"seq": self.log_seq, "level": level, "line": line})

    def _parse_progress(self, line):
        axis_match = re.search(r"EIXO\s+(\d+)", line)
        if axis_match:
            with self.lock:
                self.progress["axis"] = axis_match.group(1)
                self.progress["test"] = ""
            return
        test_result = re.search(r"\[(\w+)\].*?(✓|✗|⚠)\s+(\d+\.?\d*)", line)
        if test_result:
            with self.lock:
                test_id = test_result.group(1)
                axis = re.match(r"\d+", test_id)
                axis = axis.group(0) if axis else self.progress.get("axis", "")
                score = float(test_result.group(3))
                self.progress["test"] = test_id
                symbol = test_result.group(2)
                axis_data = self.progress["axes"].setdefault(axis, {
                    "completed": 0, "passed": 0, "failed": 0,
                    "skipped": 0, "score_sum": 0.0, "mean_score": 0.0,
                    "latest_test": "",
                })
                axis_data["completed"] += 1
                axis_data["score_sum"] += score
                axis_data["mean_score"] = axis_data["score_sum"] / axis_data["completed"]
                axis_data["latest_test"] = test_id
                if "✓" in symbol or "PASS" in line:
                    self.progress["passed"] = self.progress.get("passed", 0) + 1
                    axis_data["passed"] += 1
                elif "✗" in symbol or "FAIL" in line:
                    self.progress["failed"] = self.progress.get("failed", 0) + 1
                    axis_data["failed"] += 1
                else:
                    self.progress["skipped"] = self.progress.get("skipped", 0) + 1
                    axis_data["skipped"] += 1
                self.progress["current"] = (
                    self.progress.get("passed", 0)
                    + self.progress.get("failed", 0)
                    + self.progress.get("skipped", 0)
                )
            return
        test_err = re.search(r"\[(\w+)\].*?(⚠|ERR|SKIP)", line)
        if test_err:
            with self.lock:
                test_id = test_err.group(1)
                axis_match = re.match(r"\d+", test_id)
                axis = axis_match.group(0) if axis_match else self.progress.get("axis", "")
                self.progress["test"] = test_id
                self.progress["skipped"] = self.progress.get("skipped", 0) + 1
                axis_data = self.progress["axes"].setdefault(axis, {
                    "completed": 0, "passed": 0, "failed": 0,
                    "skipped": 0, "score_sum": 0.0, "mean_score": 0.0,
                    "latest_test": "",
                })
                axis_data["completed"] += 1
                axis_data["skipped"] += 1
                axis_data["latest_test"] = test_id
                axis_data["mean_score"] = axis_data["score_sum"] / axis_data["completed"]
                self.progress["current"] = (
                    self.progress.get("passed", 0)
                    + self.progress.get("failed", 0)
                    + self.progress.get("skipped", 0)
                )

    @staticmethod
    def _expected_test_count(suite_path: str, axes_filter: str):
        try:
            source = Path(suite_path).read_text(encoding="utf-8")
            ids = re.findall(r"runTest\s*\(\s*\{\s*id:\s*['\"](\d+[a-z0-9_-]*)['\"]", source, re.I)
        except OSError:
            return 118
        if not axes_filter.strip():
            return len(ids) or 118
        selected = {int(value.strip()) for value in axes_filter.split(",") if value.strip()}
        return sum(1 for test_id in ids if int(re.match(r"\d+", test_id).group(0)) in selected)

    @staticmethod
    def _fetch_server_properties(api_url: str):
        parsed = urllib.parse.urlsplit(api_url)
        if not parsed.scheme or not parsed.netloc:
            return {"error": "URL da API invalida"}
        props_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/props", "", ""))
        try:
            with urllib.request.urlopen(props_url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return {"url": props_url, "reported": payload}
        except Exception as exc:
            return {"url": props_url, "error": str(exc)}

    @staticmethod
    def _normalize_run_config(
        repeats=1, seed=0, scale="auto", mode="auto",
        reasoning_effort="default", reasoning_budget="auto", sampling="server",
        temperature=0.6, top_k=20, top_p=0.95, min_p=0.05,
        repeat_penalty=1.0, max_tokens=16384, timeout=300, xctx_scale=1.0,
        os_filter="all", judge_url="", judge_model="",
    ):
        def integer(name, value, minimum, maximum):
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                raise ValueError(f"{name} deve ser um inteiro entre {minimum} e {maximum}.") from None
            if not minimum <= parsed <= maximum:
                raise ValueError(f"{name} deve estar entre {minimum} e {maximum}.")
            return parsed

        def decimal(name, value, minimum, maximum):
            try:
                parsed = float(str(value).strip())
            except (TypeError, ValueError):
                raise ValueError(f"{name} deve ser um número entre {minimum} e {maximum}.") from None
            if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
                raise ValueError(f"{name} deve estar entre {minimum} e {maximum}.")
            return parsed

        parsed_scale = str(scale or "auto").lower()
        if parsed_scale not in {"auto", "small", "medium", "large", "xlarge"}:
            raise ValueError("Escala inválida.")
        parsed_mode = str(mode or "auto").lower()
        if parsed_mode not in {"auto", "think", "nothink"}:
            raise ValueError("Modo de token de raciocínio inválido.")
        parsed_effort = str(reasoning_effort or "default").lower()
        if parsed_effort not in {"default", "off", "low", "medium", "high", "max"}:
            raise ValueError("Esforço de raciocínio inválido.")
        if parsed_effort != "default" and parsed_mode != "auto":
            raise ValueError(
                "Use apenas um controle de raciocínio: com esforço diferente "
                "de padrão, deixe o token Qwen em Auto."
            )
        parsed_budget = str(reasoning_budget or "auto").strip().lower()
        if parsed_budget != "auto":
            integer("Budget de raciocínio", parsed_budget, -1, 262144)
        parsed_sampling = str(sampling or "server").lower()
        if parsed_sampling not in {"server", "fixed"}:
            raise ValueError("Perfil de sampling inválido.")
        parsed_os = str(os_filter or "all").lower()
        if parsed_os not in {"all", "linux", "win"}:
            raise ValueError("Filtro de SO inválido.")

        return {
            "repeats": integer("Repetições", repeats, 1, 100),
            "seed": integer("Seed", seed, -2147483648, 2147483647),
            "scale": parsed_scale,
            "mode": parsed_mode,
            "reasoning_effort": parsed_effort,
            "reasoning_budget": parsed_budget,
            "sampling": parsed_sampling,
            "temperature": decimal("Temperatura", temperature, 0, 5),
            "top_k": integer("Top K", top_k, 0, 100000),
            "top_p": decimal("Top P", top_p, 0, 1),
            "min_p": decimal("Min P", min_p, 0, 1),
            "repeat_penalty": decimal("Repeat penalty", repeat_penalty, 0, 10),
            "max_tokens": integer("Saída máxima", max_tokens, 1, 262144),
            "timeout": integer("Timeout", timeout, 1, 86400),
            "xctx_scale": decimal("Escala de contexto longo", xctx_scale, 0.05, 10),
            "os_filter": parsed_os,
            "judge_url": str(judge_url or "").strip(),
            "judge_model": str(judge_model or "").strip(),
        }

    def start_run(
        self, axes_filter="", repeats=1, seed=0, scale="auto", mode="auto",
        api_url="", runtime_context=None, reasoning_effort="default",
        reasoning_budget="auto", sampling="server", temperature=0.6, top_k=20,
        top_p=0.95, min_p=0.05, repeat_penalty=1.0, max_tokens=16384,
        timeout=300, xctx_scale=1.0, os_filter="all", judge_url="", judge_model="",
    ):
        with self.lock:
            if self.state in ("running", "stopping"):
                raise ValueError("Uma avaliação já está em execução.")
        run_config = self._normalize_run_config(
            repeats=repeats, seed=seed, scale=scale, mode=mode,
            reasoning_effort=reasoning_effort, reasoning_budget=reasoning_budget,
            sampling=sampling, temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, repeat_penalty=repeat_penalty, max_tokens=max_tokens,
            timeout=timeout, xctx_scale=xctx_scale, os_filter=os_filter,
            judge_url=judge_url, judge_model=judge_model,
        )
        suite_path = str(Path(__file__).resolve().parent.parent / "alpha_eval_suite_v3.3.mjs")
        try:
            selected_axes = [int(value.strip()) for value in axes_filter.split(",") if value.strip()]
        except ValueError:
            raise ValueError("Eixos devem ser numeros separados por virgula, por exemplo: 1,2,17") from None
        if any(axis < 1 or axis > 23 for axis in selected_axes):
            raise ValueError("Eixos devem estar entre 1 e 23.")
        requested_axes = selected_axes or list(range(1, 24))
        runtime_context = dict(runtime_context or {})
        vision = runtime_context.get("capabilities", {}).get("vision", {})
        skipped_axes = []
        if not vision.get("enabled") and 3 in requested_axes:
            requested_axes.remove(3)
            skipped_axes.append({"axis": 3, "reason": vision.get("reason") or "visao indisponivel"})
        effective_axes_filter = ",".join(str(axis) for axis in requested_axes)
        effective_api_url = api_url or "http://127.0.0.1:8080/v1/chat/completions"
        benchmark_context = {
            "axes": axes_filter or "all",
            "effective_axes": effective_axes_filter or "none",
            "skipped_axes": skipped_axes,
            **run_config,
            "api_url": effective_api_url,
        }
        expected_tests = (
            self._expected_test_count(suite_path, effective_axes_filter)
            if requested_axes else 0
        )
        if expected_tests == 0:
            with self.lock:
                self.state = "skipped"
                self.error = "Eixo 3 pulado: modelo sem visao multimodal ativa."
                self.logs.clear()
                self.stop_requested = False
                self.progress = {
                    "current": 0, "total": 0, "passed": 0, "failed": 0,
                    "skipped": 0, "axis": "", "test": "", "axes": {},
                }
                self.runtime_context = runtime_context
                self.axes_filter = axes_filter
                self.effective_axes_filter = ""
                self.repeats = run_config["repeats"]
                self.seed = run_config["seed"]
                self.scale = run_config["scale"]
                self.mode = run_config["mode"]
                self.reasoning_effort = run_config["reasoning_effort"]
                self.reasoning_budget = run_config["reasoning_budget"]
                self.sampling = run_config["sampling"]
                self.temperature = run_config["temperature"]
                self.top_k = run_config["top_k"]
                self.top_p = run_config["top_p"]
                self.min_p = run_config["min_p"]
                self.repeat_penalty = run_config["repeat_penalty"]
                self.max_tokens = run_config["max_tokens"]
                self.timeout = run_config["timeout"]
                self.xctx_scale = run_config["xctx_scale"]
                self.os_filter = run_config["os_filter"]
                self.judge_url = run_config["judge_url"]
                self.judge_model = run_config["judge_model"]
                self.api_url = effective_api_url
                self.runtime_context["benchmark"] = benchmark_context
                self.runtime_context["server_properties"] = self._fetch_server_properties(
                    effective_api_url
                )
            self._emit("warn", self.error)
            return

        with self.lock:
            self.state = "running"
            self.error = ""
            self.logs.clear()
            self.progress = {
                "current": 0, "total": expected_tests, "passed": 0, "failed": 0,
                "skipped": 0, "axis": "", "test": "", "axes": {},
            }
            self.stop_requested = False
            self.axes_filter = axes_filter
            self.effective_axes_filter = effective_axes_filter
            self.repeats = run_config["repeats"]
            self.seed = run_config["seed"]
            self.scale = run_config["scale"]
            self.mode = run_config["mode"]
            self.reasoning_effort = run_config["reasoning_effort"]
            self.reasoning_budget = run_config["reasoning_budget"]
            self.sampling = run_config["sampling"]
            self.temperature = run_config["temperature"]
            self.top_k = run_config["top_k"]
            self.top_p = run_config["top_p"]
            self.min_p = run_config["min_p"]
            self.repeat_penalty = run_config["repeat_penalty"]
            self.max_tokens = run_config["max_tokens"]
            self.timeout = run_config["timeout"]
            self.xctx_scale = run_config["xctx_scale"]
            self.os_filter = run_config["os_filter"]
            self.judge_url = run_config["judge_url"]
            self.judge_model = run_config["judge_model"]
            self.api_url = effective_api_url
            self.results_file = ""
            self.runtime_context = runtime_context
            self.runtime_context["benchmark"] = benchmark_context
            self.runtime_context["server_properties"] = self._fetch_server_properties(self.api_url)

        node_bin, node_version, node_error = _resolve_node_runtime()
        if not node_bin:
            message = _node_unavailable_message(node_error)
            with self.lock:
                self.state = "error"
                self.error = message
                self.runtime_context["evaluation_runtime"] = {
                    "status": "unavailable",
                    "node": "",
                    "version": "",
                    "error": node_error,
                }
            self._emit("error", message)
            return
        with self.lock:
            self.runtime_context["evaluation_runtime"] = {
                "status": "ready",
                "node": node_bin,
                "version": node_version,
            }

        if not os.path.isfile(suite_path):
            with self.lock:
                self.state = "error"
                self.error = f"Suite não encontrada: {suite_path}"
            return

        args = [
            node_bin, suite_path,
            "--url", self.api_url,
            "--scale", self.scale,
            "--mode", self.mode,
            "--repeats", str(self.repeats),
            "--seed", str(self.seed),
            "--reasoning-effort", self.reasoning_effort,
            "--reasoning-budget", self.reasoning_budget,
            "--sampling", self.sampling,
            "--temperature", str(self.temperature),
            "--top-k", str(self.top_k),
            "--top-p", str(self.top_p),
            "--min-p", str(self.min_p),
            "--repeat-penalty", str(self.repeat_penalty),
            "--max-tokens", str(self.max_tokens),
            "--timeout", str(self.timeout),
            "--xctx-scale", str(self.xctx_scale),
            "--os", self.os_filter,
        ]
        if self.judge_url:
            args.extend(["--judge-url", self.judge_url])
        if self.judge_model:
            args.extend(["--judge-model", self.judge_model])
        if effective_axes_filter:
            args.append(f"--axis={effective_axes_filter}")

        thread = threading.Thread(target=self._run_worker, args=(args,), daemon=True)
        thread.start()

    def _run_worker(self, args):
        import re as _re_mod
        _ansi_re = _re_mod.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
        try:
            with self.lock:
                if self.stop_requested:
                    self.state = "stopped"
                    return
            self._emit("info", f"Iniciando avaliacao: {shlex.join(args)}")
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parent.parent),
                start_new_session=True,
            )
            with self.lock:
                self.proc = proc
                stop_now = self.stop_requested
            if stop_now:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            for raw in iter(proc.stdout.readline, ""):
                line = _ansi_re.sub("", raw.rstrip("\n"))
                self._parse_progress(line)
                if "PASS" in line or "✓" in line:
                    self._emit("pass", line)
                elif "FAIL" in line or "✗" in line:
                    self._emit("fail", line)
                elif "SKIP" in line or "SKIP" in line or "ERR" in line:
                    self._emit("warn", line)
                elif "ERROR" in line.upper() or "Erro" in line:
                    self._emit("error", line)
                else:
                    self._emit("info", line)
            proc.wait()
            exit_code = proc.returncode
            with self.lock:
                if self.proc is proc:
                    self.proc = None
            if self.stop_requested:
                self._emit("warn", "Avaliacao interrompida pelo usuario.")
            elif exit_code == 0:
                self._emit("info", "Avaliacao concluida com sucesso.")
                self._emit("info", "Gerando dados do dashboard...")
                self._generate_dashboard_data()
            else:
                self._emit("error", f"Suite encerrou com codigo {exit_code}.")
            with self.lock:
                if self.stop_requested:
                    self.state = "stopped"
                    self.error = ""
                else:
                    self.state = "done" if exit_code == 0 else "error"
                    self.error = "" if exit_code == 0 else f"Suite encerrou com codigo {exit_code}"
        except Exception as exc:
            with self.lock:
                self.state = "error"
                self.error = str(exc)
            self._emit("error", f"Erro interno: {exc}")
        finally:
            self.proc = None

    def _generate_dashboard_data(self):
        try:
            project_root = Path(__file__).resolve().parent.parent
            gen_script = str(project_root / "eval_dashboard" / "tools" / "generate-dashboard-data.mjs")
            result_file = str(project_root / "eval_results.json")
            runs_dir = self.runs_dir
            if not os.path.isfile(gen_script):
                self._emit("warn", f"Gerador de dashboard não encontrado: {gen_script}")
                return
            if not os.path.isfile(result_file):
                self._emit("warn", "Resultado final da avaliacao nao foi criado.")
                return
            try:
                with open(result_file, "r", encoding="utf-8") as fh:
                    result = json.load(fh)
                runtime = json.loads(json.dumps(self.runtime_context, default=str))
                hardware = runtime.get("hardware", {})
                stable_hardware = {
                    key: hardware.get(key)
                    for key in (
                        "cpu_model", "cpu_cores", "cpu_threads", "ram_total_gb",
                        "gpu_detected", "gpu_model", "gpu_vram_gb", "gpu_driver",
                        "gpu_cuda", "storage_type",
                    )
                }
                fingerprint_source = {
                    "model": runtime.get("model"),
                    "parameters": runtime.get("parameters"),
                    "hardware": stable_hardware,
                    "llama_server": runtime.get("llama_server"),
                    "benchmark": {
                        key: runtime.get("benchmark", {}).get(key)
                        for key in (
                            "axes", "scale", "reasoning_mode", "reasoning_effort",
                            "reasoning_budget", "sampling", "temperature", "top_k",
                            "top_p", "min_p", "repeat_penalty", "max_tokens",
                            "timeout", "xctx_scale", "os_filter", "judge_url",
                            "judge_model",
                        )
                    },
                }
                encoded = json.dumps(fingerprint_source, sort_keys=True, ensure_ascii=True)
                fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
                parameters = runtime.get("parameters", {})
                cache_k = str(parameters.get("cache_k") or "?").upper()
                cache_v = str(parameters.get("cache_v") or "?").upper()
                ctx = parameters.get("ctx", "?")
                ngl = parameters.get("ngl", "?")
                flash = parameters.get("flash", "?")
                variant_label = f"KV {cache_k}/{cache_v} · ctx {ctx} · GPU {ngl} · FA {flash} · {fingerprint[:6]}"
                base_model = result.get("model") or runtime.get("model", {}).get("name") or "modelo"
                runtime["variant"] = {
                    "fingerprint": fingerprint,
                    "label": variant_label,
                    "display_name": f"{base_model} · {variant_label}",
                }
                captured_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                result["timestamp"] = captured_at
                result["run_id"] = f"web-{captured_at}-{fingerprint}"
                result["runtime_context"] = runtime
                temporary = f"{result_file}.{os.getpid()}.tmp"
                with open(temporary, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, ensure_ascii=False, indent=2)
                os.replace(temporary, result_file)
                runs_dir.mkdir(parents=True, exist_ok=True)
                safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(base_model)).strip("_")
                stamp = captured_at.replace("-", "").replace(":", "")
                archive = runs_dir / f"checkpoint_{stamp}_{safe_model}_{fingerprint}.json"
                archive_temp = archive.with_suffix(".tmp")
                with open(archive_temp, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, ensure_ascii=False, indent=2)
                os.replace(archive_temp, archive)
            except (OSError, ValueError) as exc:
                self._emit("warn", f"Falha ao anexar contexto reproduzivel: {exc}")
                return
            node_bin, _node_version, node_error = _resolve_node_runtime()
            if not node_bin:
                self._emit("warn", _node_unavailable_message(node_error))
                return
            proc = subprocess.run(
                [node_bin, gen_script, f"--input-root={runs_dir}", f"--output={self.dashboard_output_dir}"],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
                timeout=30,
            )
            output = proc.stdout.strip()
            if output:
                for line in output.split("\n"):
                    self._emit("info", line)
            if proc.returncode != 0:
                self._emit("warn", f"Gerador falhou: {proc.stderr.strip()}")
                return
            data_file = str(self.dashboard_output_dir / "dashboard-data.json")
            if os.path.isfile(data_file):
                with open(data_file, "r", encoding="utf-8") as fh:
                    self.dashboard_data = json.load(fh)
                    self.dashboard_timestamp = self.dashboard_data.get("generatedAt", "")
                    self.results_file = str(archive)
                    self.history_load_attempted = True
                self._emit("info", "✅ Dashboard data gerado.")
        except FileNotFoundError:
            self._emit("warn", "Node.js não encontrado — não foi possível gerar dashboard data.")
        except Exception as exc:
            self._emit("warn", f"Erro ao gerar dashboard: {exc}")

    def stop_run(self):
        with self.lock:
            if self.state != "running":
                return
            proc = self.proc
            self.stop_requested = True
            self.state = "stopping"
            self.error = ""
        if not proc or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()

        def force_stop():
            time.sleep(3)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()

        threading.Thread(target=force_stop, daemon=True).start()

    def get_dashboard_data(self):
        with self.lock:
            if self.dashboard_data:
                return dict(self.dashboard_data), self.dashboard_timestamp
            if self.history_load_attempted:
                return None, ""
            self.history_load_attempted = True

        project_root = Path(__file__).resolve().parent.parent
        runs_dir = self.runs_dir
        gen_script = project_root / "eval_dashboard" / "tools" / "generate-dashboard-data.mjs"
        data_file = self.dashboard_output_dir / "dashboard-data.json"
        if not gen_script.is_file() or not any(runs_dir.glob("checkpoint_*.json")):
            return None, ""
        try:
            node_bin, _node_version, _node_error = _resolve_node_runtime()
            if not node_bin:
                return None, ""
            generated = subprocess.run(
                [node_bin, str(gen_script), f"--input-root={runs_dir}", f"--output={self.dashboard_output_dir}"],
                capture_output=True, text=True, cwd=str(project_root), timeout=30,
            )
            if generated.returncode != 0 or not data_file.is_file():
                return None, ""
            with open(data_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not data.get("checkpoints"):
                return None, ""
            with self.lock:
                self.dashboard_data = data
                self.dashboard_timestamp = data.get("generatedAt", "")
                return dict(data), self.dashboard_timestamp
        except (OSError, ValueError, subprocess.SubprocessError):
            return None, ""

    @staticmethod
    def _csv_content(headers, rows):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list)) else "" if value is None else value
                for value in row
            ])
        return stream.getvalue().encode("utf-8-sig")

    def academic_export(self):
        with self.lock:
            if not self.dashboard_data or not self.dashboard_data.get("checkpoints"):
                raise ValueError("Nenhuma avaliacao concluida disponivel para exportacao.")
            data = json.loads(json.dumps(self.dashboard_data, ensure_ascii=False, default=str))

        checkpoints = data.get("checkpoints", [])
        run_rows = []
        axis_rows = []
        test_rows = []
        for checkpoint in checkpoints:
            summary = checkpoint.get("summary", {})
            runtime = checkpoint.get("runtimeContext") or {}
            parameters = runtime.get("parameters") or {}
            benchmark = runtime.get("benchmark") or {}
            run_rows.append([
                checkpoint.get("runId"), checkpoint.get("timestamp"),
                checkpoint.get("baseModel"), checkpoint.get("model"),
                checkpoint.get("configFingerprint"), checkpoint.get("variantLabel"),
                summary.get("meanScore"), summary.get("passRate"),
                summary.get("coverage"), summary.get("passed"), summary.get("failed"),
                summary.get("skipped"), summary.get("medianLatencyMs"),
                summary.get("p90LatencyMs"), summary.get("promptTokens"),
                summary.get("completionTokens"), summary.get("candidateWallMs"),
                summary.get("completionTokensPerSecond"),
                parameters.get("cache_k"), parameters.get("cache_v"),
                parameters.get("ctx"), parameters.get("ngl"), parameters.get("flash"),
                parameters.get("parallel"), parameters.get("batch"), parameters.get("ubatch"),
                benchmark.get("axes"), benchmark.get("effective_axes"),
                benchmark.get("skipped_axes"), benchmark.get("repeats"), benchmark.get("seed"),
            ])
            for axis in summary.get("axes", []):
                axis_rows.append([
                    checkpoint.get("runId"), checkpoint.get("configFingerprint"),
                    checkpoint.get("baseModel"), checkpoint.get("variantLabel"),
                    axis.get("axisNumber"), axis.get("label"), axis.get("available"),
                    axis.get("evaluated"), axis.get("passed"), axis.get("meanScore"),
                    axis.get("passRate"),
                ])
            for test in checkpoint.get("tests", []):
                candidate = (test.get("metrics") or {}).get("candidate") or {}
                test_rows.append([
                    checkpoint.get("runId"), checkpoint.get("configFingerprint"),
                    checkpoint.get("baseModel"), checkpoint.get("variantLabel"),
                    test.get("id"), test.get("axisNumber"), test.get("axisLabel"),
                    test.get("name"), test.get("difficulty"), test.get("status"),
                    test.get("score"), test.get("latencyMs"), test.get("failureType"),
                    candidate.get("calls"), candidate.get("retries"),
                    candidate.get("prompt_tokens"), candidate.get("completion_tokens"),
                    candidate.get("wall_ms"), test.get("details"), test.get("response"),
                ])

        files = {
            "data/dashboard-data.json": json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            "tables/runs.csv": self._csv_content([
                "run_id", "timestamp", "base_model", "variant", "config_fingerprint",
                "variant_label", "mean_score", "pass_rate", "coverage", "passed", "failed",
                "skipped", "median_latency_ms", "p90_latency_ms", "prompt_tokens",
                "completion_tokens", "candidate_wall_ms", "completion_tokens_per_second",
                "kv_cache_k", "kv_cache_v", "context", "gpu_layers", "flash_attention",
                "parallel", "batch", "ubatch", "requested_axes", "effective_axes",
                "skipped_axes", "repeats", "seed",
            ], run_rows),
            "tables/axes.csv": self._csv_content([
                "run_id", "config_fingerprint", "base_model", "variant_label", "axis_number",
                "axis_label", "available", "evaluated", "passed", "mean_score", "pass_rate",
            ], axis_rows),
            "tables/tests.csv": self._csv_content([
                "run_id", "config_fingerprint", "base_model", "variant_label", "test_id",
                "axis_number", "axis_label", "test_name", "difficulty", "status", "score",
                "latency_ms", "failure_type", "candidate_calls", "candidate_retries",
                "prompt_tokens", "completion_tokens", "candidate_wall_ms", "details", "response",
            ], test_rows),
        }

        runs_dir = self.runs_dir
        for path in sorted(runs_dir.glob("checkpoint_*.json")):
            try:
                files[f"raw_runs/{path.name}"] = path.read_bytes()
            except OSError:
                continue

        readme = """# Alpha Eval - Pacote academico

Este pacote foi gerado localmente pelo Crono Matrix Launcher.

## Conteudo
- data/dashboard-data.json: dataset normalizado completo.
- tables/runs.csv: uma linha por configuracao/execucao.
- tables/axes.csv: metricas agregadas por eixo.
- tables/tests.csv: evidencias e metricas por teste.
- raw_runs/: resultados JSON originais, incluindo runtime_context.
- manifest.json: hashes SHA-256 para verificacao de integridade.

## Interpretacao
- Score varia de 0 a 10; aprovacao exige o threshold definido pela suite.
- completion_tokens_per_second = tokens de resposta / tempo agregado das chamadas candidatas.
- Latencia inclui a execucao completa do teste e pode conter retries, ferramentas e juiz LLM.
- Comparacoes devem usar testes comuns e observar cobertura, seed, hardware e configuracao de inferencia.
- Eixos pulados por incapacidade, como visao sem MMProj, nao equivalem a reprovação.

## Reprodutibilidade
Cada run inclui modelo, fingerprint, KV cache, contexto, GPU layers, sampling, hardware,
propriedades do servidor, eixos, repeticoes e seed.
"""
        files["README.md"] = readme.encode("utf-8")
        manifest = {
            "format": "alpha-eval-academic-bundle-v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runs": len(checkpoints),
            "files": [
                {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in sorted(files.items())
            ],
        }
        files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return f"alpha-eval-academico-{stamp}.zip", output.getvalue()

    def delete_run(self, checkpoint_id: str):
        with self.lock:
            if self.state in ("running", "stopping"):
                raise ValueError("Pare o benchmark antes de excluir uma execução.")
            data = self.dashboard_data
        if not data:
            data, _ = self.get_dashboard_data()
        checkpoint = next(
            (item for item in (data or {}).get("checkpoints", []) if item.get("id") == checkpoint_id),
            None,
        )
        if not checkpoint:
            raise ValueError("Execução selecionada não foi encontrada.")

        runs_dir = self.runs_dir.resolve()
        deleted = []
        for source in checkpoint.get("sourceRefs", []):
            path = Path(source).resolve()
            if path.parent != runs_dir or not path.name.startswith("checkpoint_") or path.suffix != ".json":
                continue
            try:
                path.unlink()
                deleted.append(path.name)
            except FileNotFoundError:
                continue
        if not deleted:
            raise ValueError("O arquivo desta execução não pertence ao histórico web ou já foi removido.")

        with self.lock:
            self.dashboard_data = None
            self.dashboard_timestamp = ""
            self.history_load_attempted = False
        refreshed, _ = self.get_dashboard_data()
        return {
            "deleted": deleted,
            "remaining": len((refreshed or {}).get("checkpoints", [])),
        }
