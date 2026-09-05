#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#   Crono llama-server core  v5.0
#   Hardware, GGUF, parâmetros e cliente Hugging Face.
#   NÃO modifica o script original.
#
#   Sem dependência de interface gráfica.
#
#   CPU: Xeon E5-2682 v4 (Broadwell) | GPU: RTX 3060 12GB
# ============================================================

import subprocess
import threading
import math
_hf_translator = None
try:
    from deep_translator import GoogleTranslator
    _hf_translator = GoogleTranslator(source="en", target="pt")
    _hf_translate_ok = True
except ImportError:
    try:
        from googletrans import Translator
        _hf_translator = Translator()
        _hf_translate_ok = True
    except ImportError:
        _hf_translate_ok = False
_hf_translate_lock = threading.Lock()
import time as _time
import os
import sys
import re
import glob
import json
import hashlib
import shlex
import struct
import urllib.parse
import urllib.request
from pathlib import Path

from autotune_cache import (
    AutotuneCache,
    binary_identity,
    hardware_identity,
    model_identity,
    sha256_file,
)

# ════════════════════════════════════════════════════════════
#   CONFIGURAÇÕES — idênticas ao launch_model.sh
# ════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_LLAMA_CPP = _PROJECT_ROOT / "llama.cpp"
_LOCAL_LLAMA_BUILD = _LOCAL_LLAMA_CPP / "build-crono"
MODELS_DIR = os.environ.get(
    "CRONO_MODELS_DIR", str(_PROJECT_ROOT / "modelos")
)
LLAMA_SERVER = os.environ.get(
    "CRONO_LLAMA_SERVER",
    str(_LOCAL_LLAMA_BUILD / "bin" / "llama-server"),
)
LLAMA_FIT_PARAMS = os.environ.get(
    "CRONO_LLAMA_FIT_PARAMS", str(Path(LLAMA_SERVER).with_name("llama-fit-params"))
)
_LOCAL_GGUF_PY_DIR = _LOCAL_LLAMA_CPP / "gguf-py"


def _valid_gguf_py_dir(path: Path) -> bool:
    return (path / "gguf" / "__init__.py").is_file()


def resolve_gguf_py_dir(llama_cpp_dir: str | Path = "") -> str:
    """Localiza o ``gguf-py`` pertencente ao checkout selecionado.

    O produto não inclui uma cópia do llama.cpp. Por isso o diretório escolhido
    nas interfaces é a fonte primária, inclusive quando o usuário aponta para
    o projeto pai, para um build ou diretamente para ``llama-server``.
    ``CRONO_GGUF_PY_DIR`` continua disponível como override explícito.
    """
    candidates: list[Path] = []
    configured = os.environ.get("CRONO_GGUF_PY_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    if llama_cpp_dir:
        selected = Path(llama_cpp_dir).expanduser()
        if selected.is_file() or selected.name == "llama-server":
            selected = selected.parent
        lineage = [selected, *list(selected.parents)[:5]]
        for base in lineage:
            candidates.append(base / "gguf-py")
            candidates.append(base / "llama.cpp" / "gguf-py")

    candidates.extend((_LOCAL_GGUF_PY_DIR, _PROJECT_ROOT.parent / "llama.cpp" / "gguf-py"))
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if _valid_gguf_py_dir(resolved):
            return key
    return ""


GGUF_PY_DIR = resolve_gguf_py_dir(_LOCAL_LLAMA_CPP)
if GGUF_PY_DIR and GGUF_PY_DIR not in sys.path:
    sys.path.insert(0, GGUF_PY_DIR)
MEDIA_PATH = os.environ.get(
    "CRONO_MEDIA_PATH",
    str(Path(__file__).resolve().parent / ".crono-runtime" / "uploads"),
)
NVME_SWAP_FILE = Path.home() / ".local" / "share" / "crono-matrix" / "swapfile"

# Maior tipo ggml suportado pelo build local (GGML_TYPE_COUNT = 43 -> [0, 42])
_MAX_SUPPORTED_GGUF_TYPE = 42
# Ponto de desempenho medido na RTX 3060 12 GB com Qwen3.6-35B-A3B-Q6_K,
# contexto 262144, KV BF16/Q8_0 e batch 2048/512. Em três gerações por
# configuração, 33 camadas MoE na CPU usaram 11118 MiB e lideraram com
# 25,17 t/s; 34 e 35 camadas ficaram em 19,38 e 20,35 t/s. A margem de
# 256 MiB seleciona 33 somente quando a VRAM atualmente livre comporta esse
# perfil; se o desktop/visão ocupar mais VRAM, o estimador recua para 34+.
_QWEN35MOE_RTX3060_RESERVE_MB = 256
_LAGUNA_RTX3060_RESERVE_MB = 256
_GEMMA4_RTX3060_RESERVE_MB = 256
_AUTOTUNE_CACHE = AutotuneCache()
_SERVER_FLAG_CACHE = {}

# Valores aceitos pelo ``--spec-type`` do llama.cpp integrado. Mantenha esta
# lista alinhada ao binario local; aliases genericos como ``draft`` e ``ngram``
# nao sao aceitos pelo parser atual.
SPECULATIVE_TYPES = (
    "none",
    "draft-simple",
    "draft-eagle3",
    "draft-mtp",
    "draft-dflash",
    "draft-dspark",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "ngram-cache",
)

_GGUF_READER_PATCHED = False
_METADATA_GGUF_READER_CLASS = None


def configure_gguf_py_dir(llama_cpp_dir: str | Path = "") -> str:
    """Ativa o leitor GGUF do checkout em uso e invalida caches antigos."""
    global GGUF_PY_DIR, _GGUF_READER_PATCHED, _METADATA_GGUF_READER_CLASS
    resolved = resolve_gguf_py_dir(llama_cpp_dir)
    if resolved == GGUF_PY_DIR:
        if resolved and resolved not in sys.path:
            sys.path.insert(0, resolved)
        return resolved

    previous = GGUF_PY_DIR
    GGUF_PY_DIR = resolved
    if previous and previous in sys.path:
        sys.path.remove(previous)
    if resolved and resolved not in sys.path:
        sys.path.insert(0, resolved)
    for module_name in tuple(sys.modules):
        if module_name == "gguf" or module_name.startswith("gguf."):
            del sys.modules[module_name]
    _GGUF_READER_PATCHED = False
    _METADATA_GGUF_READER_CLASS = None
    return resolved


def _server_supports_flag(server_path: str, flag: str) -> bool:
    """Query the selected binary instead of assuming a llama.cpp CLI version."""
    try:
        resolved = str(Path(server_path).expanduser().resolve())
        stat = os.stat(resolved)
        identity = (resolved, stat.st_mtime_ns, stat.st_size)
        flags = _SERVER_FLAG_CACHE.get(identity)
        if flags is None:
            completed = subprocess.run(
                [resolved, "--help"], capture_output=True, text=True,
                timeout=10, check=False,
            )
            help_text = completed.stdout + "\n" + completed.stderr
            flags = frozenset(re.findall(r"(?<!\w)--[a-z0-9][a-z0-9-]*", help_text))
            _SERVER_FLAG_CACHE.clear()
            _SERVER_FLAG_CACHE[identity] = flags
        return flag in flags
    except (OSError, subprocess.SubprocessError):
        return False


def _server_lazy_mode_flag(server_path: str) -> str:
    """Return the lazy tensor flag supported by the selected llama-server.

    Upstream renamed ``--tensor-read-lazy`` to ``--lazy-mode`` in b10689.
    Supporting both names keeps saved launcher profiles usable with builds on
    either side of that change.
    """
    if _server_supports_flag(server_path, "--lazy-mode"):
        return "--lazy-mode"
    if _server_supports_flag(server_path, "--tensor-read-lazy"):
        return "--tensor-read-lazy"
    return ""


def _patch_gguf_reader_base() -> None:
    """Permite ao gguf-py ler GGUFs com tipos de quantização mais novos que o build
    (ex.: tipo 50 / MXFP6) sem travar com 'not a valid GGMLQuantizationType'.
    """
    global _GGUF_READER_PATCHED
    if _GGUF_READER_PATCHED:
        return
    try:
        import importlib.util
        if os.path.isdir(GGUF_PY_DIR) and GGUF_PY_DIR not in sys.path:
            sys.path.insert(0, GGUF_PY_DIR)
        from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

        def _missing(value):
            value = int(value)
            member = int.__new__(GGMLQuantizationType, value)
            member._name_ = f"GGML_TYPE_{value}"
            GGMLQuantizationType._value2member_map_[value] = member
            return member

        GGMLQuantizationType._missing_ = _missing
        for v in range(0, 256):
            if v not in GGML_QUANT_SIZES:
                GGML_QUANT_SIZES[v] = (64, 16)
        _GGUF_READER_PATCHED = True
    except Exception:
        pass


def _metadata_gguf_reader(path: str):
    """Read GGUF metadata without materializing tokenizer arrays or tensors.

    ``gguf.GGUFReader`` is a general-purpose inspection API. It creates one
    NumPy object for every tokenizer token/merge and maps a data view for
    every tensor. On a modern 35B GGUF that costs hundreds of MiB of RSS even
    though the launcher only needs scalar metadata and tensor headers. This
    subclass retains upstream parsing and validation while skipping those
    large payloads. No model weight is read into RAM.
    """
    global _METADATA_GGUF_READER_CLASS
    _patch_gguf_reader_base()
    if _METADATA_GGUF_READER_CLASS is None:
        import numpy as np
        from gguf import GGUFReader  # type: ignore
        from gguf.constants import (  # type: ignore
            GGML_QUANT_SIZES,
            GGMLQuantizationType,
            GGUFValueType,
        )
        from gguf.gguf_reader import ReaderField, ReaderTensor  # type: ignore

        class MetadataGGUFReader(GGUFReader):
            _SKIP_FIELDS = frozenset({
                "tokenizer.ggml.tokens",
                "tokenizer.ggml.scores",
                "tokenizer.ggml.token_type",
                "tokenizer.ggml.merges",
            })

            def _skip_field_value(self, orig_offs: int, raw_type: int) -> int:
                """Return encoded value size without allocating its contents."""
                gtype = GGUFValueType(int(raw_type))
                if gtype == GGUFValueType.STRING:
                    length = int(self._get(orig_offs, np.uint64)[0])
                    return 8 + length
                nptype = self.gguf_scalar_to_np.get(gtype)
                if nptype is not None:
                    return int(np.dtype(nptype).itemsize)
                if gtype != GGUFValueType.ARRAY:
                    raise ValueError(f"Unknown/unhandled GGUF field type {gtype}")

                subtype = GGUFValueType(int(self._get(orig_offs, np.uint32)[0]))
                count = int(self._get(orig_offs + 4, np.uint64)[0])
                offs = orig_offs + 12
                scalar_type = self.gguf_scalar_to_np.get(subtype)
                if scalar_type is not None:
                    return 12 + count * int(np.dtype(scalar_type).itemsize)
                if subtype == GGUFValueType.STRING:
                    # Token/merge arrays are variable-width. Walk only their
                    # length prefixes; do not retain millions of NumPy views.
                    swapped = self.byte_order == "S"
                    endian = ">" if (sys.byteorder == "little") == swapped else "<"
                    unpack_u64 = struct.Struct(endian + "Q").unpack_from
                    for _ in range(count):
                        length = int(unpack_u64(self.data, offs)[0])
                        offs += 8 + length
                    return offs - orig_offs
                raise ValueError(f"Unknown/unhandled GGUF array type {subtype}")

            def _build_fields(self, offs: int, count: int) -> int:
                for _ in range(int(count)):
                    orig_offs = offs
                    key_len, key_data = self._get_str(offs)
                    offs += int(key_len.nbytes + key_data.nbytes)
                    raw_type = self._get(offs, np.uint32)
                    offs += int(raw_type.nbytes)
                    key = str(bytes(key_data), encoding="utf-8")
                    if key in self._SKIP_FIELDS:
                        offs += self._skip_field_value(offs, int(raw_type[0]))
                        continue

                    parts = [key_len, key_data, raw_type]
                    idxs_offs = len(parts)
                    size, field_parts, field_idxs, field_types = (
                        self._get_field_parts(offs, raw_type[0])
                    )
                    parts += field_parts
                    self._push_field(ReaderField(
                        orig_offs, key, parts,
                        [index + idxs_offs for index in field_idxs],
                        field_types,
                    ), skip_sum=True)
                    offs += size
                return offs

            def _build_tensors(self, start_offs: int, fields: list) -> None:
                tensors = []
                names = set()
                for field in fields:
                    _name_len, name_data, _n_dims, dims, raw_dtype, offset = field.parts
                    name = str(bytes(name_data), encoding="utf-8")
                    if name in names:
                        raise ValueError(f"Found duplicated tensor with name {name}")
                    names.add(name)
                    tensor_type = GGMLQuantizationType(int(raw_dtype[0]))
                    elements = 1
                    for dimension in dims.tolist():
                        elements *= int(dimension)
                    block_size, type_size = GGML_QUANT_SIZES[tensor_type]
                    n_bytes = elements * type_size // block_size
                    tensors.append(ReaderTensor(
                        name=name,
                        tensor_type=tensor_type,
                        shape=dims,
                        n_elements=elements,
                        n_bytes=n_bytes,
                        data_offset=int(start_offs + offset[0]),
                        data=np.empty(0, dtype=np.uint8),
                        field=field,
                    ))
                self.tensors = tensors

        _METADATA_GGUF_READER_CLASS = MetadataGGUFReader
    return _METADATA_GGUF_READER_CLASS(path)


_FIT_CURVE_CACHE = {}
_FIT_CURVE_LOCK = threading.Lock()
_FIT_PLAN_CACHE = {}
_FIT_PLAN_LOCK = threading.Lock()
_QWEN_MOE_ESTIMATE_CACHE = {}
_QWEN_MOE_ESTIMATE_LOCK = threading.Lock()


def _is_glm47_flash(meta) -> bool:
    """Identify GLM-4.7-Flash despite its GGUF DEEPSEEK2 architecture tag."""
    identity = re.sub(
        r"[^a-z0-9]", "",
        f"{getattr(meta, 'general_basename', '')} {getattr(meta, 'path', '')}".lower(),
    )
    return getattr(meta, "arch", "").lower() == "deepseek2" and "glm47flash" in identity


def _requires_symmetric_kv(meta) -> bool:
    """Models whose MLA implementation rejects different K/V cache types."""
    return _is_glm47_flash(meta)


def _is_gemma4_moe(meta) -> bool:
    """Return true only for the MoE member of the Gemma 4 family."""
    return (
        getattr(meta, "arch", "").lower() == "gemma4"
        and int(getattr(meta, "expert_count", 0) or 0) > 0
    )


def _is_auxiliary_gguf(path: str) -> bool:
    stem = Path(path).stem.lower()
    return any(marker in stem for marker in (
        "mmproj", "vocoder", "imatrix", "importance-matrix",
    ))


def _is_secondary_shard(path: str) -> bool:
    match = re.search(r"-(\d{5})-of-\d{5}\.gguf$", path, re.I)
    return bool(match and int(match.group(1)) > 1)


def _gguf_parts(path: str) -> list:
    match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path, re.I)
    if not match:
        return [path]
    total = int(match.group(3))
    parts = [f"{match.group(1)}-{index:05d}-of-{total:05d}.gguf"
             for index in range(1, total + 1)]
    missing = [part for part in parts if not os.path.isfile(part)]
    if missing:
        raise ValueError(f"GGUF multipartes incompleto: faltam {len(missing)} parte(s)")
    return parts


def _gguf_total_size(path: str) -> int:
    return sum(os.path.getsize(part) for part in _gguf_parts(path))


def _find_companion(
    model_path: str,
    kind: str,
    model_basename: str = "",
    projection_dim: int = 0,
) -> str:
    # Models may be kept in their own subdirectory while the companion
    # projector/vocoder remains in the shared models directory. Search the
    # model directory first and then its immediate parent; never walk the
    # whole tree, which could associate a companion from another model.
    model_dir = Path(model_path).resolve().parent
    search_dirs = [model_dir]
    if model_dir.parent != model_dir:
        search_dirs.append(model_dir.parent)
    candidates = []
    for directory in search_dirs:
        for path in glob.glob(str(directory / "*.gguf")):
            if kind in Path(path).stem.lower() and path not in candidates:
                candidates.append(path)
    if not candidates:
        return ""

    # Metadata is the primary identity.  A model directory can contain
    # several projectors whose filenames are abbreviated or vendor-specific;
    # the GGUF basename and projection dimension are more reliable than a
    # filename convention.  Keep this optional so discovery still works when
    # a minimal installation has no gguf-py available.
    if model_basename or projection_dim:
        try:
            _patch_gguf_reader_base()
            def field_value(field):
                if field is None:
                    return None
                try:
                    return field.contents()
                except Exception:
                    pass
                try:
                    return field.parts[-1].tobytes().decode(
                        "utf-8", errors="replace"
                    ).strip("\x00")
                except Exception:
                    return None

            def field(fields, key):
                if key in fields:
                    return fields[key]
                suffix = "." + key.lower()
                return next(
                    (
                        value for name, value in fields.items()
                        if name.lower().endswith(suffix)
                    ),
                    None,
                )

            normalize = lambda value: re.sub(
                r"[^a-z0-9]", "", str(value).lower()
            )
            wanted = normalize(model_basename)
            metadata_matches = []
            for candidate in candidates:
                try:
                    fields = _metadata_gguf_reader(candidate).fields
                    basename = field_value(
                        fields.get("general.basename")
                        or fields.get("general.name")
                    )
                    has_vision = field_value(
                        fields.get("clip.has_vision_encoder")
                    )
                    has_audio = field_value(
                        fields.get("clip.has_audio_encoder")
                    )
                    has_gen_audio = field_value(
                        fields.get("clip.has_gen_audio_encoder")
                    )
                    projections = [
                        field_value(fields.get("clip.vision.projection_dim")),
                        field_value(fields.get("clip.audio.projection_dim")),
                        field_value(fields.get("clip.gen.audio.projection_dim")),
                    ]
                    try:
                        has_vision = int(has_vision) == 1
                    except (TypeError, ValueError):
                        has_vision = False
                    try:
                        has_audio = int(has_audio) == 1
                    except (TypeError, ValueError):
                        has_audio = False
                    try:
                        has_gen_audio = int(has_gen_audio) == 1
                    except (TypeError, ValueError):
                        has_gen_audio = False
                    projection_values = set()
                    for projection in projections:
                        try:
                            projection_values.add(int(projection))
                        except (TypeError, ValueError):
                            continue
                    basename_matches = bool(
                        wanted and basename and normalize(basename) == wanted
                    )
                    dimension_matches = bool(
                        projection_dim and int(projection_dim) in projection_values
                    )
                    has_input_encoder = has_vision or has_audio
                    if kind == "mmproj" and has_input_encoder and (
                        basename_matches or dimension_matches
                    ):
                        # Prefer a projector that exposes more of an Omni
                        # model (vision + audio) over a smaller vision-only
                        # companion with an equally compatible name.  File
                        # size is only a final tie-breaker; metadata remains
                        # authoritative.
                        modality_count = sum((
                            has_vision, has_audio, has_gen_audio,
                        ))
                        directory_rank = (
                            0
                            if Path(candidate).resolve().parent == model_dir
                            else 1
                        )
                        metadata_matches.append((
                            directory_rank,
                            -modality_count,
                            0 if basename_matches else 1,
                            -os.path.getsize(candidate),
                            candidate,
                        ))
                except Exception:
                    continue
            if metadata_matches:
                metadata_matches.sort(
                    key=lambda item: (
                        item[0], item[1], item[2], item[3], item[4].lower()
                    )
                )
                return metadata_matches[0][4]
        except (ImportError, ModuleNotFoundError, AttributeError):
            pass

    ignored = {"mmproj", "vocoder", "model", "gguf", "f16", "f32", "bf16", "fp16"}

    def tokens(path: str) -> set:
        # Split digits from letters.  A numeric model prefix must share its
        # meaningful alphabetic family tokens with the corresponding
        # projector; treating the prefix and the word as one token makes a
        # valid MMProj disappear.
        values = set(re.findall(r"[a-z]+|\d+", Path(path).stem.lower()))
        return {
            v for v in values
            if v not in ignored
            and not v.isdigit()
            and not re.fullmatch(r"i?q\d.*", v)
        }

    def family_tokens(path: str) -> set:
        # Numeric coincidences such as Q6, 35B or the projection dimension are
        # not evidence that a shared MMProj belongs to this language model.
        # Require at least one meaningful alphabetic family token so a lone
        # Qwen projector cannot be attached to GLM (or another text model).
        generic = {
            "base", "chat", "flash", "instruct", "model", "projector",
            "vision", "visual", "vl",
        }
        return {
            value for value in tokens(path)
            if value.isalpha() and len(value) >= 3 and value not in generic
        }

    model_tokens = tokens(model_path)
    ranked = sorted(
        candidates,
        key=lambda path: (-len(model_tokens & tokens(path)), path.lower()),
    )
    best = ranked[0]
    return best if family_tokens(model_path) & family_tokens(best) else ""

# ════════════════════════════════════════════════════════════
#   MCP SERVER (Node.js — ecossistema ASAEL)
# ════════════════════════════════════════════════════════════
MCP_DIR = os.environ.get("CRONO_MCP_DIR", str(_PROJECT_ROOT / "mcp-crono-matrix"))
try:
    MCP_PORT = int(os.environ.get("CRONO_MCP_PORT", "3001"))
except ValueError:
    MCP_PORT = 3001
MCP_ENTRY = os.environ.get(
    "CRONO_MCP_ENTRY", os.path.join(MCP_DIR, "native_server.mjs")
)

# ════════════════════════════════════════════════════════════
#   DETECÇÃO DE HARDWARE  (portado do launch_model.sh)
# ════════════════════════════════════════════════════════════
class HardwareInfo:
    def __init__(self):
        self.path           = ""
        self.cpu_model        = "Detectando..."
        self.cpu_cores        = 1
        self.cpu_threads      = 1
        self.cpu_sockets      = 1
        self.numa_nodes       = 1
        self.cpu_temp         = 0
        self.ram_total_gb     = 0.0
        self.ram_avail_gb     = 0.0
        self.ram_avail_mb     = 0
        self.swap_total_mb    = 0
        self.swap_free_mb     = 0
        self.swap_used_mb     = 0
        self.swap_zram_total_mb = 0
        self.swap_zram_used_mb = 0
        self.swap_zram_priority = -1
        self.swap_nvme_total_mb = 0
        self.swap_nvme_used_mb = 0
        self.swap_nvme_priority = -1
        self.swap_nvme_path   = str(NVME_SWAP_FILE)
        self.swap_nvme_active = False
        self.swap_nvme_preferred = False
        self.gpu_detected     = False
        self.gpu_model        = "Nenhuma"
        self.gpu_vram_mb      = 0
        self.gpu_vram_free_mb = 0
        self.gpu_vram_gb      = "0.0"
        self.gpu_vram_free_gb = "0.0"
        self.gpu_driver       = ""
        self.gpu_cuda         = ""
        self.gpu_temp         = 0
        self.storage_type     = "unknown"
        self.disk_free_gb     = "?"

    def detect(self):
        self._cpu()
        self._ram()
        self._swap()
        self._gpu()
        self._storage()

    def _run(self, cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    def _cpu(self):
        out = self._run(["env", "LANG=C", "lscpu"])
        for line in out.splitlines():
            if re.match(r"Model name", line, re.I):
                self.cpu_model = line.split(":", 1)[1].strip()
            m = re.match(r"^CPU\(s\):\s+(\d+)", line)
            if m:
                self.cpu_threads = int(m.group(1))
            m2 = re.match(r"Core\(s\) per socket.*?(\d+)", line)
            if m2:
                self.cpu_cores = int(m2.group(1))
            m3 = re.match(r"Socket\(s\):\s+(\d+)", line)
            if m3:
                self.cpu_sockets = int(m3.group(1))
            m4 = re.match(r"NUMA node\(s\):\s+(\d+)", line)
            if m4:
                self.numa_nodes = int(m4.group(1))
        self.cpu_cores *= max(self.cpu_sockets, 1)
        if self.cpu_cores <= 1:
            nc = self._run(["nproc"])
            if nc.isdigit():
                self.cpu_cores = int(nc)
                self.cpu_threads = self.cpu_cores
        sensors = self._run(["sensors"])
        m = re.search(r"Package id 0:?\s+\+(\d+)", sensors)
        if m:
            self.cpu_temp = int(m.group(1))

    def _ram(self):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        self.ram_total_gb = kb / 1048576
                    elif line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        self.ram_avail_gb = kb / 1048576
                        self.ram_avail_mb = kb // 1024
        except Exception:
            pass

    def _swap(self):
        """Read the real kernel swap table and separate ZRAM from NVMe."""
        self.swap_total_mb = 0
        self.swap_used_mb = 0
        self.swap_zram_total_mb = 0
        self.swap_zram_used_mb = 0
        self.swap_zram_priority = -1
        self.swap_nvme_total_mb = 0
        self.swap_nvme_used_mb = 0
        self.swap_nvme_priority = -1
        self.swap_nvme_active = False
        self.swap_nvme_preferred = False
        try:
            rows = Path("/proc/swaps").read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            rows = []
        for row in rows:
            parts = row.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            try:
                size_mb = int(parts[2]) // 1024
                used_mb = int(parts[3]) // 1024
                priority = int(parts[4])
            except ValueError:
                continue
            self.swap_total_mb += size_mb
            self.swap_used_mb += used_mb
            if Path(name).name.startswith("zram"):
                self.swap_zram_total_mb += size_mb
                self.swap_zram_used_mb += used_mb
                self.swap_zram_priority = max(
                    self.swap_zram_priority, priority
                )
                continue
            source = self._run(["findmnt", "-n", "-o", "SOURCE", "-T", name])
            if "nvme" in source.lower() or Path(name) == NVME_SWAP_FILE:
                self.swap_nvme_total_mb += size_mb
                self.swap_nvme_used_mb += used_mb
                self.swap_nvme_priority = max(
                    self.swap_nvme_priority, priority
                )
                self.swap_nvme_path = name
                self.swap_nvme_active = True
        self.swap_free_mb = max(self.swap_total_mb - self.swap_used_mb, 0)
        self.swap_nvme_preferred = bool(
            self.swap_nvme_active
            and (
                self.swap_zram_priority < 0
                or self.swap_nvme_priority > self.swap_zram_priority
            )
        )

    def _gpu(self):
        out = self._run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap,temperature.gpu",
            "--format=csv,noheader,nounits"
        ])
        if not out:
            return
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        if len(parts) >= 3:
            self.gpu_detected     = True
            self.gpu_model        = parts[0]
            try:
                self.gpu_vram_mb      = int(parts[1])
                self.gpu_vram_free_mb = int(parts[2])
                self.gpu_vram_gb      = f"{self.gpu_vram_mb / 1024:.1f}"
                self.gpu_vram_free_gb = f"{self.gpu_vram_free_mb / 1024:.1f}"
            except ValueError:
                pass
            if len(parts) > 3: self.gpu_driver = parts[3]
            if len(parts) > 4: self.gpu_cuda   = parts[4]
        if len(parts) > 5 and parts[5].isdigit():
            self.gpu_temp = int(parts[5])

    def _storage(self):
        for disk in sorted(glob.glob("/sys/block/*")):
            try:
                rota = open(f"{disk}/queue/rotational").read().strip()
                if rota == "0":
                    self.storage_type = "SSD/NVMe"; break
                elif rota == "1":
                    self.storage_type = "HDD"
            except Exception:
                pass
        out = self._run(["df", "-BG", "/home"])
        if out:
            lines = out.splitlines()
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    self.disk_free_gb = parts[3].replace("G", "")

    def identity(self) -> dict:
        return hardware_identity(self)


# ════════════════════════════════════════════════════════════
#   METADADOS DO MODELO  (portado do launch_model.sh)
# ════════════════════════════════════════════════════════════
class ModelMetadata:
    def __init__(self):
        self.path            = ""
        self.ctx_max        = 4096
        self.layers         = 32
        self.full_attention_interval = 0
        self.sliding_window = 0
        self.sliding_window_pattern = []
        self.key_len_swa    = 0
        self.val_len_swa    = 0
        self.shared_kv_layers = 0
        self.embedding_length_per_layer_input = 0
        self.swa_layers     = 0
        self.global_layers  = 0
        self.leading_dense_block_count = 0
        self.kv_layers      = 32
        self.heads          = 32
        self.heads_kv       = 0
        self.head_dim       = 128
        self.key_len        = 0
        self.val_len        = 0
        self.embed          = 4096
        self.arch           = "?"
        self.general_basename = ""
        self.quant          = "Desconhecida"
        self.params_str     = "?"
        self.mmproj_file    = ""
        self.mmproj_size_mb = 0
        self.mmproj_valid   = False
        self.mmproj_has_vision = False
        self.mmproj_has_audio = False
        self.mmproj_has_gen_audio = False
        self.mmproj_tensor_count = 0
        self.vocoder_file   = ""
        self.n_layer_nextn  = 0
        self.has_mtp        = False
        self.supports_reasoning_preserve = False
        self.size_bytes     = 0
        self.size_mb        = 0
        self.size_gb_str    = "0.0"
        self.size_label     = ""
        self.meta_ok        = False
        self.metadata_error = ""
        self.metadata_error_kind = ""
        self.quant_error    = ""
        self.profile_file   = ""
        self.profile        = {}
        self.profile_parameters = {}
        self.profile_error  = ""
        self.layer_heads = []
        self.layer_heads_kv = []
        self.layer_ff = []
        self.layer_kinds = []
        self.layer_weight_bytes = []
        self.expert_weight_bytes_by_layer = []
        self.weight_bytes_by_kind = {}
        self.non_layer_weight_bytes = 0
        self.output_weight_bytes = 0
        self.recurrent_layers = 0
        self.attention_layers = 0
        self.moe_layers = 0
        self.dense_layers = 0
        self.expert_count = 0
        self.expert_used_count = 0
        self.expert_ff = 0
        self.expert_shared_ff = 0
        self.sampling_temp = None
        self.sampling_top_k = None
        self.sampling_top_p = None
        self.sampling_min_p = None
        self.ssm_d_conv = 0
        self.ssm_d_inner = 0
        self.ssm_d_state = 0
        self.ssm_n_group = 0
        self.state_r = 0
        self.state_s = 0
        self.layer_layout_valid = False

    def load(self, path: str):
        if not os.path.isfile(path):
            return
        self.path = path
        try:
            self.size_bytes = _gguf_total_size(path)
        except (OSError, ValueError) as exc:
            self.metadata_error = str(exc)
            self.metadata_error_kind = "gguf_file"
            return
        self.size_mb     = self.size_bytes // 1048576
        sx               = self.size_bytes * 10 // 1073741824
        self.size_gb_str = f"{sx // 10}.{sx % 10}"

        fname = os.path.basename(path)
        m = re.search(r"(?i)(Q[0-9]_[A-Z0-9_]+|IQ[0-9][A-Z0-9_]*|BF16|F16|F32|FP16)", fname)
        self.quant = m.group(1).upper() if m else "Desconhecida"
        # Only accept a parameter-size token at a filename boundary.  Without
        # the trailing boundary, a name beginning with a digit (for example a
        # family name) is incorrectly reported as ``4B``.
        m2 = re.search(r"(?<![A-Za-z0-9])(\d+\.?\d*[BbMm])(?=$|[-_.])", fname)
        self.params_str = m2.group(1).upper() if m2 else "?"

        # Leitura GGUF via biblioteca Python
        try:
            import importlib.util
            if os.path.isdir(GGUF_PY_DIR) and GGUF_PY_DIR not in sys.path:
                sys.path.insert(0, GGUF_PY_DIR)
            spec = importlib.util.find_spec("gguf")
            if spec is not None:
                _patch_gguf_reader_base()
                r = _metadata_gguf_reader(path)
                fields = r.fields
                try:
                    max_type = max(int(t.tensor_type) for t in r.tensors)
                    if max_type > _MAX_SUPPORTED_GGUF_TYPE:
                        self.quant_error = (
                            f"Modelo usa tipo ggml {max_type}, inexistente neste "
                            f"motor (suporta ate {_MAX_SUPPORTED_GGUF_TYPE}). Baixe "
                            f"uma versao deste modelo em Q4_K_M/Q8_0/BF16."
                        )
                except Exception:
                    pass
            else:
                fields = None

            if fields is None:
                raise ModuleNotFoundError("biblioteca gguf não encontrada")

            def _field_value(field):
                if field is None:
                    return None
                try:
                    return field.contents()
                except Exception:
                    pass
                try:
                    raw = field.parts[-1].tobytes()
                    return raw.decode("utf-8", errors="replace").strip("\x00")
                except Exception:
                    return None

            def _field(key_frag, arch_pfx=""):
                exact = f"{arch_pfx}.{key_frag}" if arch_pfx else key_frag
                if exact in fields:
                    return fields[exact]
                suffix = f".{key_frag}".lower()
                for key, field in fields.items():
                    lowered = key.lower()
                    if lowered.endswith(suffix) and (
                        not arch_pfx or lowered.startswith(f"{arch_pfx.lower()}.")
                    ):
                        return field
                return None

            def _value(key_frag, arch_pfx=""):
                return _field_value(_field(key_frag, arch_pfx))

            def _scalar(key_frag, arch_pfx=""):
                value = _value(key_frag, arch_pfx)
                if isinstance(value, (bool, int, float)):
                    return int(value)
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    try:
                        return int(value[0])
                    except (TypeError, ValueError):
                        return None
                return None

            def _number(key_frag, arch_pfx=""):
                value = _value(key_frag, arch_pfx)
                if isinstance(value, (bool, int, float)):
                    return float(value)
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    try:
                        return float(value[0])
                    except (TypeError, ValueError):
                        return None
                return None

            def _layer_array(key_frag, count, arch_pfx=""):
                value = _value(key_frag, arch_pfx)
                if isinstance(value, (list, tuple)):
                    try:
                        values = [int(item) for item in value]
                    except (TypeError, ValueError):
                        return []
                    if len(values) == count:
                        return values
                    if len(values) == 1:
                        return values * count
                    return []
                if isinstance(value, (bool, int, float)):
                    return [int(value)] * count
                return []

            def _layer_bool_array(key_frag, count, arch_pfx=""):
                value = _value(key_frag, arch_pfx)
                if isinstance(value, (list, tuple)):
                    values = [bool(item) for item in value]
                    if len(values) == count:
                        return values
                    if len(values) == 1:
                        return values * count
                if isinstance(value, (bool, int, float)):
                    return [bool(value)] * count
                return []

            af = fields.get("general.architecture")
            if af:
                value = _field_value(af)
                if isinstance(value, str):
                    self.arch = value.upper()
            basename_field = fields.get("general.basename") or fields.get("general.name")
            if basename_field:
                value = _field_value(basename_field)
                if isinstance(value, str):
                    self.general_basename = value
            size_label_field = fields.get("general.size_label")
            if size_label_field:
                value = _field_value(size_label_field)
                if isinstance(value, str) and value.strip():
                    self.size_label = value.strip()
                    # GGUF metadata is authoritative over filename parsing.
                    self.params_str = self.size_label
            chat_template_field = fields.get("tokenizer.chat_template")
            if chat_template_field:
                try:
                    chat_template = _field_value(chat_template_field)
                    self.supports_reasoning_preserve = "reasoning_content" in chat_template
                except Exception:
                    pass

            ap = self.arch.lower()
            found_metadata = set()
            for key, attr, threshold in [
                ("context_length", "ctx_max",         0),
                ("block_count",    "layers",          0),
                ("head_count",     "heads",           0),
                ("head_count_kv",  "heads_kv",        0),
                ("key_length",     "key_len",         0),
                ("value_length",   "val_len",         0),
                ("embedding_length","embed",          0),
                ("full_attention_interval", "full_attention_interval", 0),
                ("attention.sliding_window", "sliding_window", 0),
                ("attention.key_length_swa", "key_len_swa", 0),
                ("attention.value_length_swa", "val_len_swa", 0),
                ("attention.shared_kv_layers", "shared_kv_layers", -1),
                ("embedding_length_per_layer_input", "embedding_length_per_layer_input", -1),
                ("leading_dense_block_count", "leading_dense_block_count", 0),
            ]:
                val = _scalar(key, ap)
                if val is not None and val > threshold:
                    setattr(self, attr, val)
                    found_metadata.add(attr)

            layer_count = max(self.layers, 1)
            self.layer_heads = _layer_array("attention.head_count", layer_count, ap)
            self.layer_heads_kv = _layer_array("attention.head_count_kv", layer_count, ap)
            self.layer_ff = _layer_array("feed_forward_length", layer_count, ap)
            self.sliding_window_pattern = _layer_bool_array(
                "attention.sliding_window_pattern", layer_count, ap
            )
            if not self.layer_heads and self.heads > 0:
                self.layer_heads = [self.heads] * layer_count
            if not self.layer_heads_kv:
                scalar_kv = _scalar("head_count_kv", ap)
                if scalar_kv is not None and scalar_kv > 0:
                    self.layer_heads_kv = [scalar_kv] * layer_count
            if not self.layer_ff:
                scalar_ff = _scalar("feed_forward_length", ap)
                if scalar_ff is not None:
                    self.layer_ff = [scalar_ff] * layer_count
            if self.layer_heads and "heads" not in found_metadata:
                self.heads = max(self.layer_heads, default=self.heads)
                found_metadata.add("heads")

            self.expert_count = _scalar("expert_count", ap) or 0
            self.expert_used_count = _scalar("expert_used_count", ap) or 0
            self.expert_ff = _scalar("expert_feed_forward_length", ap) or 0
            self.expert_shared_ff = _scalar("expert_shared_feed_forward_length", ap) or 0
            # GGUF scalar floats are commonly stored as float32. Values such
            # as 0.95 therefore arrive as 0.949999988079071. Sampling knobs
            # are human-facing decimal parameters, so do not leak binary
            # representation noise into the UI and command preview.
            sampling_temp = _number("sampling.temp", "general")
            sampling_top_p = _number("sampling.top_p", "general")
            sampling_min_p = _number("sampling.min_p", "general")
            self.sampling_temp = (
                round(sampling_temp, 6) if sampling_temp is not None else None
            )
            self.sampling_top_k = _scalar("sampling.top_k", "general")
            self.sampling_top_p = (
                round(sampling_top_p, 6) if sampling_top_p is not None else None
            )
            self.sampling_min_p = (
                round(sampling_min_p, 6) if sampling_min_p is not None else None
            )
            self.ssm_d_conv = _scalar("ssm.conv_kernel", ap) or 0
            self.ssm_d_inner = _scalar("ssm.inner_size", ap) or 0
            self.ssm_d_state = _scalar("ssm.state_size", ap) or 0
            self.ssm_n_group = _scalar("ssm.group_count", ap) or 0

            # MTP: busca por nextn_predict_layers (varia por arquitetura)
            for k, v in fields.items():
                if "nextn_predict_layers" in k:
                    self.n_layer_nextn = _scalar("nextn_predict_layers", ap) or 0
                    break

            if self.layer_heads_kv:
                nonzero_kv = [value for value in self.layer_heads_kv if value > 0]
                self.heads_kv = max(nonzero_kv, default=self.heads_kv)
            if self.heads_kv <= 0:
                self.heads_kv = max(self.heads, 1)
            derived_dim = self.embed // max(self.heads, 1)
            if self.key_len <= 0:
                self.key_len = derived_dim or 128
            if self.val_len <= 0:
                self.val_len = derived_dim or self.key_len
            self.head_dim = max(self.key_len, self.val_len)
            self.has_mtp = self.n_layer_nextn > 0
            required = {"ctx_max", "layers", "heads", "embed"}
            self.meta_ok = self.arch != "?" and required.issubset(found_metadata)
            if not self.meta_ok:
                self.metadata_error = "GGUF sem metadados essenciais de arquitetura/contexto"
                self.metadata_error_kind = "gguf_metadata"

            layer_layout_arch = ap in {"nemotron_h", "nemotron_h_moe"}
            if (
                layer_layout_arch
                and len(self.layer_heads_kv) == layer_count
                and len(self.layer_ff) == layer_count
            ):
                self.layer_kinds = []
                for kv_heads, ff_length in zip(self.layer_heads_kv, self.layer_ff):
                    if kv_heads == 0 and ff_length == 0:
                        kind = "recurrent"
                    elif kv_heads > 0 and ff_length == 0:
                        kind = "attention"
                    elif self.expert_count > 0:
                        kind = "moe"
                    else:
                        kind = "dense"
                    self.layer_kinds.append(kind)
                self.layer_layout_valid = True
                self.recurrent_layers = self.layer_kinds.count("recurrent")
                self.attention_layers = self.layer_kinds.count("attention")
                self.moe_layers = self.layer_kinds.count("moe")
                self.dense_layers = self.layer_kinds.count("dense")
                self.kv_layers = self.attention_layers

            # Gemma 4 26B-A4B alternates five local SWA layers with one
            # global layer. Local and global attention also use different
            # key/value dimensions. Keep this layout for exact KV accounting.
            if ap == "gemma4" and len(self.sliding_window_pattern) == layer_count:
                self.swa_layers = sum(self.sliding_window_pattern)
                self.global_layers = layer_count - self.swa_layers
                self.attention_layers = layer_count
                self.recurrent_layers = 0
                self.moe_layers = layer_count if self.expert_count > 0 else 0
                self.dense_layers = 0 if self.expert_count > 0 else layer_count
                self.layer_kinds = [
                    "moe" if self.expert_count > 0 else "dense"
                    for _ in range(layer_count)
                ]
                self.layer_layout_valid = True
                self.kv_layers = layer_count

            # Qwen3.5/3.6 MoE stores scalar attention metadata, but its
            # recurrent/full-attention cadence is encoded by the architecture.
            # Expose that cadence so KV estimates do not count all 40 blocks.
            if ap in {"qwen35", "qwen35moe"}:
                interval = max(self.full_attention_interval, 1)
                base_layers = max(self.layers - self.n_layer_nextn, 0)
                self.layer_heads = [
                    self.heads if (index >= base_layers or (index + 1) % interval == 0) else 0
                    for index in range(layer_count)
                ]
                self.layer_heads_kv = [
                    self.heads_kv if (index >= base_layers or (index + 1) % interval == 0) else 0
                    for index in range(layer_count)
                ]
                # The dense 9B variant has the same recurrent/full-attention
                # cadence as Qwen3.5, but without expert tensors. The MTP
                # block is attention-bearing and must be included in the KV
                # estimate for both variants.
                if ap == "qwen35moe":
                    self.layer_kinds = [
                        "moe" if index < base_layers else "dense"
                        for index in range(layer_count)
                    ]
                else:
                    self.layer_kinds = [
                        "recurrent" if heads_kv == 0 else "attention"
                        for heads_kv in self.layer_heads_kv
                    ]
                self.layer_layout_valid = True
                self.recurrent_layers = sum(value == 0 for value in self.layer_heads_kv)
                self.attention_layers = sum(value > 0 for value in self.layer_heads_kv)
                self.moe_layers = base_layers if ap == "qwen35moe" else 0
                self.dense_layers = (
                    max(layer_count - base_layers, 0)
                    if ap == "qwen35moe" else 0
                )
                self.kv_layers = self.attention_layers

            if self.ssm_d_conv > 0 and self.ssm_d_inner > 0:
                self.state_r = (self.ssm_d_conv - 1) * (
                    self.ssm_d_inner + 2 * self.ssm_n_group * self.ssm_d_state
                )
                self.state_s = self.ssm_d_state * self.ssm_d_inner

            layer_bytes = [0] * layer_count
            expert_bytes = [0] * layer_count
            for tensor in r.tensors:
                match = re.match(r"^blk\.(\d+)\.", tensor.name)
                if match:
                    index = int(match.group(1))
                    if index < len(layer_bytes):
                        layer_bytes[index] += int(tensor.n_bytes)
                        # Expert matrices are the only tensors moved by
                        # --n-cpu-moe. Record their exact GGUF footprint so
                        # RAM residency decisions are based on this model,
                        # not on a Qwen-derived model-size approximation.
                        if "_exps." in tensor.name:
                            expert_bytes[index] += int(tensor.n_bytes)
                if tensor.name == "output.weight":
                    self.output_weight_bytes += int(tensor.n_bytes)
            self.layer_weight_bytes = layer_bytes
            self.expert_weight_bytes_by_layer = expert_bytes
            self.non_layer_weight_bytes = max(
                self.size_bytes - sum(layer_bytes), 0
            )
            if self.layer_kinds:
                by_kind = {}
                for kind, size in zip(self.layer_kinds, layer_bytes):
                    by_kind[kind] = by_kind.get(kind, 0) + size
                self.weight_bytes_by_kind = by_kind
        except (ModuleNotFoundError, ImportError) as exc:
            self.metadata_error = str(exc)
            missing = str(getattr(exc, "name", "") or "").lower()
            self.metadata_error_kind = (
                "gguf_library" if missing in {"", "gguf"} else "dependency"
            )
        except Exception as exc:
            self.metadata_error = str(exc)
            self.metadata_error_kind = "gguf_parse"

        # Fallback: arquitetura pelo nome do arquivo
        if self.arch == "?":
            m3 = re.search(r"(?i)(llama|gemma|mistral|qwen|phi|falcon|mpt|bloom|gpt|yi|deepseek)", fname)
            if m3:
                self.arch = m3.group(1).upper()
        if self.heads_kv <= 0:
            self.heads_kv = max(self.heads, 1)
        derived_dim = self.embed // max(self.heads, 1)
        if self.key_len <= 0:
            self.key_len = derived_dim or 128
        if self.val_len <= 0:
            self.val_len = derived_dim or self.key_len
        self.head_dim = max(self.key_len, self.val_len)
        base_layers = max(self.layers - self.n_layer_nextn, 1)
        if self.layer_layout_valid:
            self.kv_layers = self.attention_layers
        elif self.full_attention_interval > 1:
            self.kv_layers = math.ceil(base_layers / self.full_attention_interval)
        else:
            self.kv_layers = base_layers

        # Detecta mmproj / vocoder
        self.mmproj_file = _find_companion(
            path, "mmproj", self.general_basename, self.embed
        )
        if self.mmproj_file:
            self.mmproj_size_mb = os.path.getsize(self.mmproj_file) // 1048576
            self._validate_mmproj()
        self.vocoder_file = _find_companion(path, "vocoder")
        self._load_profile(path)

    def identity(self, digest: str = "") -> dict:
        return model_identity(self.path, self.arch, digest)

    def _load_profile(self, model_path: str) -> None:
        model = Path(model_path)
        candidates = [model.with_suffix(".launch.json"), model.parent / "launch_model.json"]
        for profile_path in candidates:
            if not profile_path.is_file():
                continue
            try:
                with profile_path.open(encoding="utf-8") as handle:
                    profile = json.load(handle)
                if not isinstance(profile, dict) or not isinstance(profile.get("parameters", {}), dict):
                    self.profile_error = f"Perfil inválido: {profile_path}"
                    continue
                expected = profile.get("model_file")
                if expected and expected != model.name:
                    continue
                active = profile.get("active_preset")
                presets = profile.get("presets", {})
                preset = presets.get(active, {}) if isinstance(presets, dict) else {}
                effective_parameters = dict(profile["parameters"])
                if isinstance(preset, dict):
                    effective_parameters.update({
                        key: value for key, value in preset.items() if key != "description"
                    })
                self.profile_file = str(profile_path)
                self.profile = profile
                self.profile_parameters = effective_parameters
                return
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                self.profile_error = f"Erro no perfil {profile_path}: {exc}"
                continue

    def _validate_mmproj(self):
        self.mmproj_valid = False
        self.mmproj_has_vision = False
        self.mmproj_has_audio = False
        self.mmproj_has_gen_audio = False
        self.mmproj_tensor_count = 0
        try:
            import importlib.util
            spec = importlib.util.find_spec("gguf")
            if spec is not None:
                _patch_gguf_reader_base()
                r = _metadata_gguf_reader(self.mmproj_file)
                self.mmproj_tensor_count = len(r.tensors)

                def field_value(key):
                    field = r.fields.get(key)
                    if field is None:
                        return None
                    try:
                        return field.contents()
                    except Exception:
                        raw = field.parts[-1].tobytes()
                        try:
                            return raw.decode("utf-8", errors="strict").strip("\x00")
                        except (UnicodeDecodeError, AttributeError):
                            return int.from_bytes(raw, "little")

                def field_bool(key):
                    value = field_value(key)
                    try:
                        return bool(int(value))
                    except (TypeError, ValueError):
                        return str(value).strip().lower() in {"true", "yes", "on"}

                self.mmproj_has_vision = field_bool("clip.has_vision_encoder")
                self.mmproj_has_audio = field_bool("clip.has_audio_encoder")
                self.mmproj_has_gen_audio = field_bool(
                    "clip.has_gen_audio_encoder"
                )

                companion_name = (
                    field_value("general.basename")
                    or field_value("general.name")
                )
                compatible = False
                if self.general_basename and companion_name:
                    normalize = lambda value: re.sub(
                        r"[^a-z0-9]", "", str(value).lower()
                    )
                    compatible = (
                        normalize(companion_name)
                        == normalize(self.general_basename)
                    )
                for key in (
                    "clip.vision.projection_dim",
                    "clip.audio.projection_dim",
                    "clip.gen.audio.projection_dim",
                ):
                    try:
                        compatible = compatible or int(field_value(key)) == self.embed
                    except (TypeError, ValueError):
                        continue
                self.mmproj_valid = bool(
                    compatible and (
                        self.mmproj_has_vision
                        or self.mmproj_has_audio
                        or self.mmproj_has_gen_audio
                    )
                )
            else:
                self.mmproj_valid = False
        except Exception:
            self.mmproj_valid = False


# ════════════════════════════════════════════════════════════
#   CÁLCULO DE PARÂMETROS ÓTIMOS  (portado do launch_model.sh)
# ════════════════════════════════════════════════════════════
class OptimalParams:
    def __init__(
        self, hw: HardwareInfo, meta: ModelMetadata,
        llama_server: str = LLAMA_SERVER, llama_fit_params: str = "",
    ):
        self.hw   = hw
        self.meta = meta
        self.llama_server = str(Path(llama_server).expanduser())
        self.llama_fit_params = str(
            Path(llama_fit_params).expanduser()
            if llama_fit_params
            else Path(self.llama_server).with_name("llama-fit-params")
        )
        self.ctx              = meta.ctx_max
        self.ngl              = 0
        self.ngl_reason       = ""
        self.cache_k          = "f16"
        self.cache_v          = "f16"
        self.cache_reason     = ""
        self.kv_unified       = False
        self.kv_reason        = ""
        self.kv_offload       = "y"
        self.kv_offload_reason = ""
        self.flash            = "auto"
        self.flash_reason     = ""
        self.threads          = 8
        self.threads_reason   = ""
        self.threads_batch    = 0
        self.threads_batch_reason = ""
        self.batch            = 2048
        self.batch_reason     = ""
        self.ubatch           = 256
        self.split_mode       = "layer"
        self.device           = ""
        self.device_reason    = ""
        self.poll             = 50
        self.prio             = 0
        self.numa             = "none"
        self.numa_reason      = ""
        self.repack           = "y"
        self.repack_reason    = ""
        self.load_mode        = "mmap"
        self.load_mode_reason = ""
        self.host_tensor_mb   = 0
        self.host_peak_mb     = 0
        self.memory_shortfall_mb = 0
        # Host-side state grows after the model has loaded.  Keep it separate
        # from tensors so the pre-flight swap plan accounts for the worst
        # long-context state instead of treating the initial RSS as final.
        self.prompt_cache_peak_mb = 0
        self.checkpoint_snapshot_mb = 0
        self.checkpoint_peak_mb = 0
        self.checkpoint_limit_reason = ""
        self.kv_device_mb = 0
        self.host_growth_reason = ""
        self.swap_recommended_gib = 0
        self.swap_plan_reason = ""
        self.tensor_read_lazy = "auto"
        self.tensor_read_lazy_reason = (
            "auto — leitura sob demanda apenas para tensores acima de 4 GiB em mmap"
        )
        self.direct_io        = "n"
        self.no_host          = "n"
        self.vision_enabled   = False
        self.omni             = "n"
        self.omni_reason      = ""
        self.mmproj_offload   = "y"
        self.mmproj_offload_reason = ""
        self.audio            = "n"
        self.mlock            = "n"
        self.no_mmap          = "n"
        self.cont_batching    = "y"
        self.cont_batching_reason = ""
        self.cache_prompt     = "y"
        self.cache_prompt_reason = ""
        self.reuse_port       = "n"
        self.offline          = "n"
        self.fit              = "y"
        self.fit_target       = 256
        self.fit_ctx          = 4096
        self.runtime_overhead_mb = 384
        self.reasoning                = "auto"
        self.reasoning_reason         = ""
        self.reasoning_format         = "auto"
        self.reasoning_format_reason  = ""
        self.reasoning_budget         = -1
        self.reasoning_preserve       = "auto"
        self.reasoning_preserve_reason = "auto — segue a capacidade do template do modelo"
        self.chat_template_kwargs     = ""
        self.mcp_config_file   = ""
        self.mcp_config_json   = ""
        self.mcp_config_reason = ""
        self.temp             = 0.6
        self.top_k            = 20
        self.top_p            = 0.95
        self.repeat_penalty   = 1.00
        self.min_p            = 0.05
        self.sampling_reason  = "perfil geral do launcher"
        self.presence_penalty = 0.00
        self.frequency_penalty = 0.00
        self.repeat_last_n    = 64
        self.seed             = -1
        self.ignore_eos       = "n"
        self.sampler_seq      = "edskypmxt"
        self.dry_multiplier   = 0.00
        self.dry_base         = 1.75
        self.dry_allowed_length = 2
        self.dry_penalty_last_n = -1
        self.top_nsigma       = -1.00
        self.typical_p        = 1.00
        self.xtc_probability  = 0.00
        self.xtc_threshold    = 0.10
        self.dynatemp_range   = 0.00
        self.dynatemp_exp     = 1.00
        self.mirostat         = 0
        self.mirostat_lr      = 0.10
        self.mirostat_ent     = 5.00
        self.adaptive_target  = -1.00
        self.adaptive_decay   = 0.90
        self.image_min_tokens = 0
        self.image_max_tokens = 0
        self.image_min_tokens_reason = ""
        self.mtmd_batch_max   = 1024
        self.mtmd_batch_reason = ""
        self.parallel         = 1
        self.port             = 8080
        self.host             = "127.0.0.1"
        self.swa_full         = "n"
        self.swa_reason       = ""
        self.cache_reuse      = 0
        self.cache_reuse_reason = ""
        self.cpu_moe          = "n"
        self.n_cpu_moe        = 0
        self.n_cpu_moe_reason = "0 — o llama.cpp decide o offload MoE"
        self.n_cpu_ffn        = 0
        self.n_cpu_ffn_reason = "0 — FFN denso segue o offload normal de camadas"
        self.sleep_idle       = -1
        self.jinja              = "y"
        self.slot_similarity    = "0.10"
        self.agentic_max_turns  = 10
        self.agentic_max_tool_preview_lines = 25
        self.spec_type          = "none"
        self.spec_type_reason   = ""
        self.spec_draft_n_max   = 3
        self.spec_draft_n_max_reason = ""
        self.spec_draft_p_min   = 0.0
        self.spec_draft_p_min_reason = ""
        self.spec_draft_n_min   = 0
        self.spec_draft_p_split = 0.10
        self.spec_ngram_mod_n_min = 48
        self.spec_ngram_mod_n_max = 64
        self.spec_ngram_mod_n_match = 24
        self.spec_ngram_min_hits = 1
        self.cache_ram          = 2048
        self.ctx_checkpoints    = 32
        self.checkpoint_min_step = 8192
        self.context_shift      = "n"
        self.warmup             = "y"
        self.timeout            = 3600
        # Level 4 is required by current llama.cpp to emit its authoritative
        # model/context/compute buffer breakdown.  The launcher parses those
        # lines to verify that a requested GPU KV cache really landed in CUDA.
        self.log_verbosity      = 4
        self.metrics            = "n"
        self.log_file           = ""
        self.log_colors         = "auto"
        self.log_prefix         = "y"
        self.log_timestamps     = "y"
        self.perf               = "n"
        self.check_tensors      = "n"
        self.op_offload         = "y"
        self.backend_sampling   = "n"
        self.backend_sampling_reason = "desabilitado — sem backend acelerador"
        self.override_kv        = ""
        self.rope_scaling_type  = ""
        self.rope_scale         = 0
        self.rope_freq_base     = 0
        self.rope_freq_scale    = 0
        self.yarn_orig_ctx      = 0
        self.yarn_ext_factor    = -1.0
        self.yarn_attn_factor   = -1.0
        self.yarn_beta_slow     = -1.0
        self.yarn_beta_fast     = -1.0
        self.api_key            = ""
        self.api_key_file       = ""
        self.ssl_key_file       = ""
        self.ssl_cert_file      = ""
        self.cors_origins       = ""
        self.cors_methods       = ""
        self.cors_headers       = ""
        self.cors_credentials   = "y"
        self.alias              = ""
        self.tags               = ""
        self.threads_http       = -1
        self.sse_ping_interval  = 30
        self.ui_config_file     = ""
        self.slot_save_path     = ""
        self.no_mmproj_auto     = "n"
        self.agentic            = "y"
        self.cache_idle_slots   = "y"
        self.reasoning_budget_message = ""
        self.spm_infill         = "n"
        self.kv_fits_vram       = False
        self.fit_plan_overrides = ""
        self.fit_plan_reason    = ""
        self.fit_plan_source    = "none"
        self.autotune_hit       = None

        # GGUF generation metadata is the universal baseline.  Family
        # adapters below may still choose a task-specific official preset,
        # but an unknown architecture must not silently inherit Qwen values.
        embedded_sampling = []
        if meta.sampling_temp is not None:
            self.temp = float(meta.sampling_temp)
            embedded_sampling.append(f"temp {self.temp}")
        if meta.sampling_top_k is not None:
            self.top_k = int(meta.sampling_top_k)
            embedded_sampling.append(f"top-k {self.top_k}")
        if meta.sampling_top_p is not None:
            self.top_p = float(meta.sampling_top_p)
            embedded_sampling.append(f"top-p {self.top_p}")
        if meta.sampling_min_p is not None:
            self.min_p = float(meta.sampling_min_p)
            embedded_sampling.append(f"min-p {self.min_p}")
        if embedded_sampling:
            self.sampling_reason = (
                "metadados GGUF: " + ", ".join(embedded_sampling)
            )

        # Use task-aware model-card sampling for Qwen variants instead of the
        # generic profile. Qwen3.6 remains thinking-capable by default.
        model_id = f"{getattr(meta, 'general_basename', '')} {getattr(meta, 'path', '')}".lower()
        if "qwen3-coder" in model_id:
            self.temp           = 0.7
            self.top_k           = 20
            self.top_p           = 0.80
            self.min_p           = 0.0
            self.repeat_penalty  = 1.05
            self.reasoning       = "off"
        elif "qwen3.6" in model_id:
            # The GGUF embeds the general-thinking generation_config
            # (temperature 1.0), but Qwen's official model card recommends a
            # separate precise-coding/WebDev preset. Crono Matrix is a local
            # coding control plane, so that task-specific recipe is the
            # automatic default. Explicit UI edits remain preserved.
            self.temp             = 0.6
            self.top_k            = 20
            self.top_p            = 0.95
            self.min_p            = 0.0
            self.repeat_penalty  = 1.0
            self.presence_penalty = 0.0
            self.reasoning       = "auto"
            self.sampling_reason = (
                "Qwen3.6 oficial, thinking para codigo preciso/WebDev: "
                "temp 0.6/top-k 20/top-p 0.95/min-p 0.0/presence 0.0"
            )

        # Official GLM-4.7-Flash defaults.  Its llama.cpp representation uses
        # the DEEPSEEK2 architecture, so architecture alone is not sufficient
        # to select this profile.  Do not apply Qwen's top-k/min-p filters:
        # they are not part of the model card recipe and noticeably constrain
        # a reasoning model that was evaluated at temperature 1.0/top-p 0.95.
        if _is_glm47_flash(meta):
            self.temp            = 1.0
            self.top_k           = 0
            self.top_p           = 0.95
            self.min_p           = 0.0
            self.repeat_penalty  = 1.0
            self.presence_penalty = 0.0
            self.reasoning       = "auto"
            self.fit_ctx         = max(int(meta.ctx_max), 1)
            self.fit_target      = 256
            self.sampling_reason = (
                "GLM-4.7-Flash oficial: temp 1.0, top-p 0.95; "
                "top-k/min-p desativados"
            )

        # Nemotron-H Omni is a metadata-identifiable architecture, not a
        # filename preset.  Its reasoning recipe deliberately does not use
        # Qwen's top-k/min-p filters and its hybrid cache is cheap enough that
        # the native window should be the fit target.
        if meta.arch.lower() in {"nemotron_h", "nemotron_h_moe"}:
            self.temp = (
                float(meta.sampling_temp)
                if meta.sampling_temp is not None else 0.6
            )
            self.top_k = (
                int(meta.sampling_top_k)
                if meta.sampling_top_k is not None else 0
            )
            self.top_p = (
                float(meta.sampling_top_p)
                if meta.sampling_top_p is not None else 0.95
            )
            self.min_p = (
                float(meta.sampling_min_p)
                if meta.sampling_min_p is not None else 0.0
            )
            self.repeat_penalty = 1.0
            self.presence_penalty = 0.0
            self.reasoning = "auto"
            self.reasoning_budget = 16384
            self.chat_template_kwargs = json.dumps(
                {"enable_thinking": True}, separators=(",", ":")
            )
            self.fit_ctx = max(int(meta.ctx_max), 1)
            self.sampling_reason = (
                "Nemotron-H reasoning/GGUF: temp 0.6, top-p 0.95, "
                "top-k/min-p desativados, budget 16384"
            )

        # Laguna XS 2.1 is not a Qwen profile.  Prefer the sampling values
        # embedded by the model author in the GGUF and fall back to the
        # official model-card recipe.  The checkpoint already embeds the
        # YaRN parameters required for its native 256K window, so no external
        # RoPE scaling must be injected by the launcher.
        if meta.arch.lower() == "laguna":
            self.temp = (
                float(meta.sampling_temp)
                if meta.sampling_temp is not None else 1.0
            )
            self.top_k = (
                int(meta.sampling_top_k)
                if meta.sampling_top_k is not None else 20
            )
            self.top_p = (
                float(meta.sampling_top_p)
                if meta.sampling_top_p is not None else 1.0
            )
            self.min_p = (
                float(meta.sampling_min_p)
                if meta.sampling_min_p is not None else 0.0
            )
            self.repeat_penalty = 1.0
            self.presence_penalty = 0.0
            self.reasoning = "auto"
            self.chat_template_kwargs = json.dumps(
                {"enable_thinking": True}, separators=(",", ":")
            )
            # The GGUF is authoritative for the native window.  Do not copy
            # a context target from another model family into this profile.
            self.fit_ctx = max(int(meta.ctx_max), 1)
            self.fit_target = _LAGUNA_RTX3060_RESERVE_MB
            self.sampling_reason = (
                "Laguna XS 2.1 GGUF/oficial: temp 1.0, top-k 20, "
                "top-p 1.0, min-p 0.0, thinking ativo"
            )

        # Official Gemma 4 sampling is standardized across use cases and is
        # embedded in this GGUF. Its template enables thinking through this
        # kwarg instead of Qwen-specific controls.
        if meta.arch.lower() == "gemma4":
            self.temp = (
                float(meta.sampling_temp)
                if meta.sampling_temp is not None else 1.0
            )
            self.top_k = (
                int(meta.sampling_top_k)
                if meta.sampling_top_k is not None else 64
            )
            self.top_p = (
                float(meta.sampling_top_p)
                if meta.sampling_top_p is not None else 0.95
            )
            self.min_p = (
                float(meta.sampling_min_p)
                if meta.sampling_min_p is not None else 0.0
            )
            self.repeat_penalty = 1.0
            self.presence_penalty = 0.0
            # Current llama.cpp deprecates the old enable_thinking template
            # kwarg for Gemma 4. --reasoning on is now the authoritative
            # switch and keeps the UI's explicit on/off control functional.
            self.reasoning = "on"
            self.chat_template_kwargs = ""
            # The GGUF is authoritative for the native window.
            self.fit_ctx = max(int(meta.ctx_max), 1)
            self.fit_target = _GEMMA4_RTX3060_RESERVE_MB
            self.sampling_reason = (
                "Gemma 4 oficial/GGUF: temp 1.0, top-p 0.95, top-k 64, "
                "min-p desativado e reasoning on"
            )

        if (
            meta.arch.lower() == "qwen35moe"
            and hw.gpu_detected
            and 11000 <= hw.gpu_vram_mb < 16384
        ):
            self.fit_target = _QWEN35MOE_RTX3060_RESERVE_MB

    @staticmethod
    def _cache_bytes_per_element(ktype: str) -> float:
        return {
            "f32": 4.0, "f16": 2.0, "bf16": 2.0, "fp16": 2.0,
            "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
        }.get(ktype.lower(), 2.0)

    def _attention_bpt(self, ktype: str) -> int:
        key_bpe = self._cache_bytes_per_element(ktype)
        val_bpe = self._cache_bytes_per_element(self.cache_v)
        layer_kv = getattr(self.meta, "layer_heads_kv", [])
        if getattr(self.meta, "layer_layout_valid", False) and layer_kv:
            total = 0
            for heads_kv in layer_kv[:self.meta.layers]:
                if heads_kv > 0:
                    total += heads_kv * (
                        max(self.meta.key_len, 1) * key_bpe
                        + max(self.meta.val_len, 1) * val_bpe
                    )
            if total > 0:
                return math.ceil(total)
            return 0

        layers = max(self.meta.kv_layers, 1)
        heads = max(self.meta.heads_kv, 1)
        return math.ceil(layers * heads * (
            max(self.meta.key_len, 1) * key_bpe
            + max(self.meta.val_len, 1) * val_bpe
        ))

    def _attention_cache_bytes(self, ctx: int, ktype: str) -> int:
        """Estimate attention cache while respecting Gemma 4's local SWA."""
        tokens = max(int(ctx), 1)
        pattern = getattr(self.meta, "sliding_window_pattern", [])
        layer_kv = getattr(self.meta, "layer_heads_kv", [])
        if (
            self.meta.arch.lower() == "gemma4"
            and len(pattern) == self.meta.layers
            and len(layer_kv) == self.meta.layers
        ):
            key_bpe = self._cache_bytes_per_element(ktype)
            val_bpe = self._cache_bytes_per_element(self.cache_v)
            total = 0.0
            for is_swa, heads_kv in zip(pattern, layer_kv):
                if heads_kv <= 0:
                    continue
                layer_tokens = (
                    min(tokens, max(self.meta.sliding_window, 1))
                    if is_swa and self.swa_full != "y" else tokens
                )
                key_len = (
                    max(self.meta.key_len_swa, 1)
                    if is_swa and self.meta.key_len_swa > 0
                    else max(self.meta.key_len, 1)
                )
                val_len = (
                    max(self.meta.val_len_swa, 1)
                    if is_swa and self.meta.val_len_swa > 0
                    else max(self.meta.val_len, 1)
                )
                total += layer_tokens * heads_kv * (
                    key_len * key_bpe + val_len * val_bpe
                )
            return math.ceil(total)
        return tokens * self._attention_bpt(ktype)

    def _cache_bytes_for_context(self, ctx: int, ktype: str) -> int:
        return (
            self._attention_cache_bytes(ctx, ktype)
            + max(int(ctx), 1) * self._draft_bpt(ktype)
        )

    def _max_context_for_cache_bytes(self, budget_bytes: int, ktype: str) -> int:
        low, high, best = 1, max(int(self.meta.ctx_max), 1), 0
        while low <= high:
            middle = (low + high) // 2
            if self._cache_bytes_for_context(middle, ktype) <= budget_bytes:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _draft_bpt(self, ktype: str) -> int:
        if self.spec_type != "draft-mtp":
            return 0
        layer_kv = getattr(self.meta, "layer_heads_kv", [])
        key_bpe = self._cache_bytes_per_element(ktype)
        val_bpe = self._cache_bytes_per_element(self.cache_v)
        if getattr(self.meta, "layer_layout_valid", False) and layer_kv:
            base_layers = max(self.meta.layers - self.meta.n_layer_nextn, 0)
            heads = sum(max(value, 0) for value in layer_kv[base_layers:])
            if heads > 0:
                return math.ceil(heads * (
                    max(self.meta.key_len, 1) * key_bpe
                    + max(self.meta.val_len, 1) * val_bpe
                ))
        heads = max(self.meta.heads_kv, 1)
        return math.ceil(max(self.meta.n_layer_nextn, 1) * heads * (
            max(self.meta.key_len, 1) * key_bpe
            + max(self.meta.val_len, 1) * val_bpe
        ))

    def _recurrent_state_bytes(self) -> int:
        if not getattr(self.meta, "layer_layout_valid", False):
            return 0
        per_layer = (self.meta.state_r + self.meta.state_s) * 4
        return per_layer * max(self.meta.recurrent_layers, 0) * max(self.parallel, 1)

    def _checkpoint_snapshot_bytes_per_slot(self) -> int:
        """Return a conservative host-RAM bound for one server checkpoint.

        ``llama-server`` serializes context checkpoints into ``std::vector``
        buffers on the host.  They are independent from ``--cache-ram``.
        Hybrid Qwen memory has a partial-state fast path: only its recurrent
        state is serialized.  Ordinary attention/SWA implementations serialize
        their sequence state, so the safe pre-flight bound is the whole KV
        state for a full slot.  This is deliberately a capacity estimate, not
        a request to put the hot KV cache in swap.
        """
        if int(self.ctx_checkpoints) <= 0:
            return 0

        slots = max(int(self.parallel), 1)
        slot_ctx = max(math.ceil(max(int(self.ctx), 1) / slots), 1)

        if getattr(self.meta, "recurrent_layers", 0) > 0:
            per_layer = (self.meta.state_r + self.meta.state_s) * 4
            state_bytes = per_layer * self.meta.recurrent_layers
        else:
            state_bytes = self._attention_cache_bytes(slot_ctx, self.cache_k)

        # MTP keeps a draft context whose partial checkpoint is also copied to
        # host memory by llama-server.  It is normally small, but it must be
        # included in a capacity check so enabling MTP cannot invalidate the
        # swap plan calculated before process start.
        if self.spec_type == "draft-mtp":
            state_bytes += slot_ctx * self._draft_bpt(self.cache_k)

        return max(int(math.ceil(state_bytes)), 0)

    def _plan_context_state_memory(self) -> None:
        """Plan post-load host growth from prompt cache and checkpoints."""
        self.kv_device_mb = math.ceil(
            self._cache_bytes_for_context(self.ctx, self.cache_k) / 1048576
        )
        self.prompt_cache_peak_mb = max(int(self.cache_ram), 0)
        snapshot_bytes = self._checkpoint_snapshot_bytes_per_slot()
        self.checkpoint_snapshot_mb = math.ceil(snapshot_bytes / 1048576)
        # Each checkpoint is an independently allocated host vector. Round
        # each one first so the admission plan cannot miss allocator/page
        # granularity exactly at a MiB boundary.
        self.checkpoint_peak_mb = (
            self.checkpoint_snapshot_mb
            * max(int(self.ctx_checkpoints), 0)
            * max(int(self.parallel), 1)
        )
        self.host_growth_reason = (
            f"cache de prompt até {self.prompt_cache_peak_mb} MB + "
            f"checkpoints até {self.checkpoint_peak_mb} MB "
            f"({max(int(self.ctx_checkpoints), 0)} × "
            f"{self.checkpoint_snapshot_mb} MB/slot)"
        )

    def _gpu_weight_bytes(self, ngl: int) -> int:
        """Estimate the tensors that really remain on the GPU.

        ``--n-gpu-layers all`` assigns every repeating layer plus the output
        layer to the accelerator.  ``--n-cpu-moe N`` is a tensor override: it
        keeps the structural part of those layers on the GPU while moving the
        expert tensors of the first N layers back to the CPU.  Treating N as
        if it offloaded whole layers made the generic VRAM telemetry disagree
        with the measured MoE planner.
        """
        layer_bytes = getattr(self.meta, "layer_weight_bytes", [])
        if len(layer_bytes) != self.meta.layers or not any(layer_bytes):
            per_layer = self.meta.size_bytes / max(self.meta.layers, 1)
            layers = (
                self.meta.layers
                if int(ngl) >= self.meta.layers
                else min(max(int(ngl) - 1, 0), self.meta.layers)
            )
            output = getattr(self.meta, "output_weight_bytes", 0)
            return int(output + layers * per_layer)

        if int(ngl) >= self.meta.layers:
            first_gpu_layer = 0
        else:
            gpu_layers = min(max(int(ngl) - 1, 0), self.meta.layers)
            first_gpu_layer = self.meta.layers - gpu_layers
        total = sum(layer_bytes[first_gpu_layer:])

        # --n-cpu-moe applies to the first N layer indices.  Subtract only
        # experts that would otherwise belong to the GPU layer range.
        expert_bytes = getattr(self.meta, "expert_weight_bytes_by_layer", [])
        if len(expert_bytes) == self.meta.layers and self.n_cpu_moe > 0:
            last_cpu_expert_layer = min(int(self.n_cpu_moe), self.meta.layers)
            if last_cpu_expert_layer > first_gpu_layer:
                total -= sum(
                    expert_bytes[first_gpu_layer:last_cpu_expert_layer]
                )
        if int(ngl) > 0:
            total += getattr(self.meta, "output_weight_bytes", 0)
        return max(int(total), 0)

    def _fit_plan(self, requested_ctx: int | None = None):
        """Ask the installed llama.cpp fit tool for its actual tensor plan."""
        if not self.meta.path or not os.path.isfile(self.llama_fit_params):
            return None
        if not self.hw.gpu_detected or self.hw.gpu_vram_free_mb <= 0:
            return None
        try:
            model_stat = os.stat(self.meta.path)
            fit_stat = os.stat(self.llama_fit_params)
        except OSError:
            return None

        target_ctx = min(
            max(int(requested_ctx or self.meta.ctx_max), 1), self.meta.ctx_max
        )
        key = (
            self.meta.path, model_stat.st_size, model_stat.st_mtime_ns,
            self.llama_fit_params, fit_stat.st_size, fit_stat.st_mtime_ns,
            target_ctx, self.cache_k, self.cache_v, self.batch, self.ubatch,
            self.parallel, self.fit_target, self.fit_ctx, self.swa_full,
            self.spec_type,
            # Placement is a function of available hardware, not just GGUF
            # and cache types. Never reuse a fit after another workload has
            # consumed the memory that justified it.
            # Use the already detected snapshot; hw.identity() also probes
            # the CUDA toolkit and would spawn commands on every cache hit.
            self.hw.cpu_model, self.hw.cpu_cores, self.hw.cpu_threads,
            self.hw.numa_nodes, self.hw.ram_total_gb,
            self.hw.gpu_model, self.hw.gpu_vram_mb,
            self.hw.gpu_driver, self.hw.gpu_cuda,
            self.hw.gpu_vram_free_mb, self.hw.ram_avail_mb,
            tuple(os.environ.get(name, "") for name in (
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES",
            )),
        )
        with _FIT_PLAN_LOCK:
            cached = _FIT_PLAN_CACHE.get(key)
        if cached is not None:
            return cached

        # Context is a first-class model capability.  Ask the native fitter to
        # preserve the requested (normally native) window and then maximize
        # the tensors that remain on the accelerator.  The previous implicit
        # fit did the inverse: it reduced a 262K hybrid model to 4K merely to
        # retain one more expert block on CUDA.  This rule is architecture
        # neutral; llama-fit-params still owns the actual placement decision.
        fit_flash = "off" if (
            _is_glm47_flash(self.meta)
            and self.cache_k in {"f16", "bf16", "fp16"}
            and self.cache_v in {"f16", "bf16", "fp16"}
        ) else "on"
        cmd = [
            self.llama_fit_params,
            "-m", self.meta.path,
            "-c", str(target_ctx),
            "-np", str(max(self.parallel, 1)),
            "-ctk", self.cache_k,
            "-ctv", self.cache_v,
            "-b", str(self.batch),
            "-ub", str(self.ubatch),
            "-fa", fit_flash,
            "--fit", "on",
            "--fit-target", str(max(self.fit_target, 0)),
            "--fit-ctx", str(target_ctx),
        ]
        if self.swa_full == "y":
            cmd.append("--swa-full")
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=90, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if result.returncode != 0:
            return None
        plan_line = next(
            (line.strip() for line in result.stdout.splitlines()
             if line.strip().startswith("-c ")),
            "",
        )
        if not plan_line:
            return None
        try:
            tokens = shlex.split(plan_line)
            values = {}
            index = 0
            while index < len(tokens):
                token = tokens[index]
                if token in {"-c", "-ngl", "-ts", "-ot"} and index + 1 < len(tokens):
                    values[token] = tokens[index + 1]
                    index += 2
                else:
                    index += 1
            planned_ctx = int(values.get("-c", target_ctx))
            planned_ngl = int(values.get("-ngl", 0))
        except (TypeError, ValueError, IndexError):
            return None

        if planned_ctx <= 0 or planned_ngl < 0:
            return None
        plan = {
            "ctx": min(max(planned_ctx, 1), self.meta.ctx_max),
            "ngl": planned_ngl,
            "overrides": values.get("-ot", ""),
            "command": plan_line,
        }
        with _FIT_PLAN_LOCK:
            _FIT_PLAN_CACHE[key] = plan
        return plan

    def _apply_fit_plan(self) -> bool:
        plan = self._fit_plan()
        if not plan:
            return False
        self.ctx = plan["ctx"]
        self.ngl = plan["ngl"]
        self.kv_fits_vram = True
        self.fit_plan_overrides = plan["overrides"]
        self.fit_plan_source = "estimated"
        override_note = (
            "tensor overrides para experts/fragmentos MoE"
            if self.fit_plan_overrides else "sem overrides de tensor"
        )
        self.fit_plan_reason = (
            f"llama-fit-params: ctx {self.ctx} | GPU layers {self.ngl} | "
            f"{override_note}"
        )
        self.ctx_reason = (
            f"{self.fit_plan_reason} | cache {self.cache_k}/{self.cache_v} | "
            f"margem {self.fit_target} MiB"
        )
        self.ngl_reason = (
            f"plano llama-fit-params: {self.ngl} camadas estruturais/output; "
            f"{override_note}"
        )
        if self.fit_plan_overrides:
            self.n_cpu_moe_reason = (
                "llama-fit-params move experts por tensor; "
                "--n-cpu-moe não representa este plano"
            )
            self.cpu_moe = "n"
            self.n_cpu_moe = 0
            self._select_cpu_moe_load_mode()
        return True

    def _bpt(self, ktype: str) -> int:
        return max(math.ceil(self._attention_bpt(ktype) + self._draft_bpt(ktype)), 1)

    def _fit_memory_curve(self):
        if not self.meta.path or not os.path.isfile(self.llama_fit_params):
            return None
        try:
            stat = os.stat(self.meta.path)
            fit_stat = os.stat(self.llama_fit_params)
        except OSError:
            return None
        swa_full = self.swa_full == "y"
        key = (
            self.meta.path, stat.st_size, stat.st_mtime_ns,
            self.llama_fit_params, fit_stat.st_size, fit_stat.st_mtime_ns,
            self.cache_k, self.cache_v, self.batch, self.ubatch,
            self.spec_type, swa_full,
        )
        with _FIT_CURVE_LOCK:
            cached = _FIT_CURVE_CACHE.get(key)
        if cached is not None:
            return cached

        ctx_a = min(65536, self.meta.ctx_max)
        # Compute buffers are not linear near small contexts; probing at the
        # model maximum captures the high-context growth that determines OOM.
        ctx_b = self.meta.ctx_max
        if ctx_b <= ctx_a:
            return None

        points = []
        draft_bpt = self._draft_bpt(self.cache_k)
        mtp_compute_factor = 2 if self.spec_type == "draft-mtp" else 1
        allocator_correction = 512 if self.spec_type == "draft-mtp" else 256
        for ctx in (ctx_a, ctx_b):
            fit_flash = "off" if (
                _is_glm47_flash(self.meta)
                and self.cache_k in {"f16", "bf16", "fp16"}
                and self.cache_v in {"f16", "bf16", "fp16"}
            ) else "on"
            cmd = [
                self.llama_fit_params,
                "-m", self.meta.path,
                "-c", str(ctx),
                "-ctk", self.cache_k,
                "-ctv", self.cache_v,
                "-b", str(self.batch),
                "-ub", str(self.ubatch),
                "-ngl", "all",
                "-fa", fit_flash,
                "--fit", "off",
                "--fit-print", "on",
            ]
            if swa_full:
                cmd.append("--swa-full")
            try:
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=20, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            match = re.search(r"^CUDA\d+\s+(\d+)\s+(\d+)\s+(\d+)\s*$", result.stdout, re.M)
            if not match:
                return None
            model_mb, context_mb, compute_mb = map(int, match.groups())
            total_mb = (
                model_mb + context_mb + compute_mb * mtp_compute_factor
                + (ctx * draft_bpt / 1048576) + allocator_correction
            )
            points.append((ctx, total_mb, compute_mb))

        slope = (points[1][1] - points[0][1]) / (ctx_b - ctx_a)
        if slope <= 0:
            return None
        intercept = points[0][1] - slope * ctx_a
        curve = (intercept, slope, allocator_correction)
        with _FIT_CURVE_LOCK:
            _FIT_CURVE_CACHE[key] = curve
        return curve

    def _estimate_qwen_moe_gpu_mb(self, ctx: int, n_cpu_moe: int):
        """Estimate GPU memory for a hybrid MoE placement.

        The historical method name is retained for compatibility with the
        autotune cache and tests, but llama-fit-params performs the estimate
        from the selected GGUF and therefore also supports Laguna.
        """
        if not self.meta.path or not os.path.isfile(self.llama_fit_params):
            return None
        try:
            model_stat = os.stat(self.meta.path)
            fit_stat = os.stat(self.llama_fit_params)
        except OSError:
            return None
        common_key = (
            self.meta.path, model_stat.st_size, model_stat.st_mtime_ns,
            self.llama_fit_params, fit_stat.st_size, fit_stat.st_mtime_ns,
            "all-structural-layers-v2",
            int(ctx), self.cache_k, self.cache_v,
            int(self.batch), int(self.ubatch), max(int(self.parallel), 1),
            self.swa_full, self.spec_type,
        )
        key = common_key + (int(n_cpu_moe),)
        with _QWEN_MOE_ESTIMATE_LOCK:
            cached = _QWEN_MOE_ESTIMATE_CACHE.get(key)
        if cached is not None:
            return cached
        layers = max(int(self.meta.layers), 1)
        expert_bytes = list(
            getattr(self.meta, "expert_weight_bytes_by_layer", []) or []
        )
        if len(expert_bytes) != layers or not any(expert_bytes):
            # Without exact per-layer tensor sizes a linear extrapolation can
            # select an unsafe split. Let native Fit handle the model instead
            # of repeatedly probing a multi-gigabyte GGUF.
            return None
        cmd = [
            self.llama_fit_params,
            "-m", self.meta.path,
            "-c", str(ctx),
            "-np", str(max(self.parallel, 1)),
            "-ctk", self.cache_k,
            "-ctv", self.cache_v,
            "-b", str(self.batch),
            "-ub", str(self.ubatch),
            # build_cmd maps the complete structural plan to
            # --n-gpu-layers all.  Measure exactly that placement: numeric N
            # in llama.cpp includes the output layer and would leave layer 0
            # on the CPU when N == block_count.
            "-ngl", "all",
            # One real baseline with every expert layer on the CPU. The GPU
            # cost of lower values is recovered from exact GGUF tensor bytes.
            "-ncmoe", str(layers),
            "-fa", "on",
            "--fit", "off",
            "--fit-print", "on",
        ]
        if self.swa_full == "y":
            cmd.append("--swa-full")
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=20, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        match = re.search(r"^CUDA\d+\s+(\d+)\s+(\d+)\s+(\d+)\s*$", result.stdout, re.M)
        if not match:
            return None
        base_model_mb, context_mb, compute_mb = map(int, match.groups())
        if self.spec_type == "draft-mtp":
            compute_mb *= 2
            context_mb += math.ceil(ctx * self._draft_bpt(self.cache_k) / 1048576)
            allocator_mb = 256
        else:
            # llama-fit-params already reports model + context + compute
            # allocations. The independent fit_target is the safety margin;
            # adding another 256 MiB here double-counted that reserve and
            # incorrectly moved one extra expert layer to CPU.
            allocator_mb = 0
        suffix_expert_bytes = [0] * (layers + 1)
        for index in range(layers - 1, -1, -1):
            suffix_expert_bytes[index] = (
                suffix_expert_bytes[index + 1] + max(int(expert_bytes[index]), 0)
            )
        with _QWEN_MOE_ESTIMATE_LOCK:
            for cpu_layers in range(layers + 1):
                gpu_expert_mb = math.ceil(
                    suffix_expert_bytes[cpu_layers] / 1048576
                )
                estimate = (
                    base_model_mb + gpu_expert_mb + context_mb
                    + compute_mb + allocator_mb
                )
                _QWEN_MOE_ESTIMATE_CACHE[common_key + (cpu_layers,)] = estimate
            return _QWEN_MOE_ESTIMATE_CACHE.get(key)

    def _commit_measured_moe_plan(
        self, layers: int, n_cpu_moe: int, target_ctx: int
    ) -> None:
        """Commit a measured hybrid plan as the effective launch profile.

        ``--fit`` and an explicit ``--n-cpu-moe`` are different placement
        mechanisms in llama.cpp.  The old code calculated the right numbers
        but left ``fit=y`` and ``cpu_moe=n`` in the form.  That made the UI
        show ``auto`` for GPU layers and made a later full-form refresh revive
        stale values, even though ``build_cmd`` silently disabled Fit because
        it saw the explicit CPU expert count.

        Once the fitter has selected a concrete expert split, the split is
        the effective plan: show it, send it, and keep the native context
        fixed.  This remains model-independent; ``layers`` and the context
        come exclusively from the selected GGUF and the measured fitter.
        """
        self.ngl = max(int(layers), 0)
        self.n_cpu_moe = max(int(n_cpu_moe), 0)
        self.ctx = min(max(int(target_ctx), 1), max(int(self.meta.ctx_max), 1))
        self.fit_ctx = self.ctx
        self.cpu_moe = "y" if self.n_cpu_moe > 0 else "n"
        # An explicit tensor placement cannot be combined with native Fit.
        # Keeping this synchronized prevents the UI and command preview from
        # describing two different launch plans.
        self.fit = "n"
        self.fit_plan_overrides = ""
        self.kv_fits_vram = True

    def _select_cpu_moe_load_mode(self) -> None:
        """Select a fast resident load only with a safe page-out path.

        With tensor-level CPU MoE overrides, ``--load-mode none`` does not
        leave only the selected expert tensors in system RAM.  The loader's
        non-mmap backing allocation covers the complete GGUF while CUDA owns
        its offloaded buffers as well.  Treating the CUDA budget as RAM saved
        underestimated the observed host allocation by several GiB and could
        leave the kernel with barely 1 GiB available.  In that state idle
        pages are compressed into ZRAM and kswapd can monopolize the machine.

        A preferred NVMe swap can safely hold the cold duplicate of GPU
        tensors and dormant prompt/checkpoint state, provided the actual CPU
        working set still fits physical RAM.  ZRAM is not accepted for that
        role: compressing a multi-GiB model inside RAM creates the feedback
        loop this planner is meant to prevent.
        """
        self._plan_context_state_memory()
        cpu_tensor_mb = self._host_tensor_estimate_mb()
        projector_mb = (
            self.meta.mmproj_size_mb
            if self.vision_enabled and self.meta.mmproj_file
            else 0
        )
        reserve_mb = self._ram_safety_reserve_mb()
        fixed_host_mb = 1024 + self.runtime_overhead_mb
        working_set_mb = (
            cpu_tensor_mb + projector_mb + fixed_host_mb + reserve_mb
        )
        resident_required_mb = (
            max(int(self.meta.size_mb), cpu_tensor_mb)
            + projector_mb
            + self.prompt_cache_peak_mb
            + self.checkpoint_peak_mb
            + fixed_host_mb
            + reserve_mb
        )
        pageout_mb = max(resident_required_mb - self.hw.ram_avail_mb, 0)
        nvme_free_mb = max(
            self.hw.swap_nvme_total_mb - self.hw.swap_nvme_used_mb, 0
        )
        nvme_can_absorb = bool(
            self.hw.swap_nvme_preferred
            and working_set_mb <= self.hw.ram_avail_mb
            and nvme_free_mb >= pageout_mb + 1024
        )
        if pageout_mb == 0:
            self.load_mode = "none"
            self.load_mode_reason = (
                f"none — GGUF residente {self.meta.size_mb} MB + projeção "
                f"{projector_mb} MB + estado host "
                f"{self.prompt_cache_peak_mb + self.checkpoint_peak_mb + fixed_host_mb} MB "
                f"+ reserva {reserve_mb} MB "
                f"<= RAM disponível {self.hw.ram_avail_mb} MB"
            )
        elif nvme_can_absorb:
            self.load_mode = "none"
            self.load_mode_reason = (
                f"none — conjunto ativo {working_set_mb - reserve_mb} MB + "
                f"reserva adaptativa {reserve_mb} MB cabe na RAM; até "
                f"{pageout_mb} MB frios podem paginar no swap NVMe prioritário "
                f"({self.hw.swap_nvme_priority} > ZRAM "
                f"{self.hw.swap_zram_priority})"
            )
        else:
            self.load_mode = "mmap"
            route = (
                f"swap NVMe não prioritário "
                f"({self.hw.swap_nvme_priority} <= ZRAM "
                f"{self.hw.swap_zram_priority})"
                if self.hw.swap_nvme_active and not self.hw.swap_nvme_preferred
                else f"swap NVMe livre {nvme_free_mb} MB"
            )
            self.load_mode_reason = (
                f"mmap — load-mode none exigiria {resident_required_mb} MB "
                f"incluindo GGUF completo, projeção, cache/checkpoints e reserva; "
                f"RAM disponível {self.hw.ram_avail_mb} MB; {route}; experts "
                f"ativos estimados em {cpu_tensor_mb} MB permanecem file-backed"
            )

    def _ram_safety_reserve_mb(self) -> int:
        """Adaptive admission reserve without pinning an arbitrary 2/4 GiB."""
        total_mb = max(int(self.hw.ram_total_gb * 1024), 0)
        if total_mb <= 0:
            return 1024
        return min(2048, max(1024, math.ceil(total_mb * 0.05)))

    def _host_tensor_estimate_mb(self) -> int:
        """Estimate model tensors placed on the host for any architecture."""
        gpu_bytes = self._gpu_weight_bytes(self.ngl)
        placement_estimate = max(
            self.meta.size_mb - math.floor(gpu_bytes / 1048576), 0
        )
        expert_bytes = getattr(self.meta, "expert_weight_bytes_by_layer", [])
        if self.n_cpu_moe > 0 and expert_bytes:
            count = min(int(self.n_cpu_moe), len(expert_bytes))
            expert_estimate = math.ceil(
                sum(expert_bytes[:count]) / 1048576
            )
            # Include the input/non-layer tensors that llama.cpp always keeps
            # on the host; counting only experts understated the active RAM
            # set even after the GPU split itself had been corrected.
            return max(int(placement_estimate), int(expert_estimate))

        estimate = placement_estimate
        if self.fit_plan_overrides:
            # Tensor overrides can move experts while -ngl still reports all
            # structural layers on CUDA. Bound the estimate by the live CUDA
            # budget instead of incorrectly reporting zero host tensors.
            cuda_budget = max(
                self.hw.gpu_vram_free_mb - self.fit_target
                - self.runtime_overhead_mb
                - math.ceil(self._cache_bytes_for_context(self.ctx, self.cache_k) / 1048576),
                0,
            )
            estimate = max(estimate, self.meta.size_mb - cuda_budget)
        return max(int(math.ceil(estimate)), 0)

    def _plan_host_memory(self) -> None:
        """Plan RAM/mmap and NVMe swap capacity before server startup.

        The Linux swap file reserves *disk capacity*, not RAM pages belonging
        to a particular allocation.  The active KV must remain in VRAM/RAM for
        token speed; proactively moving it to NVMe would turn every decode into
        page faults.  What we can and must reserve before launch is enough
        swap-file capacity for the complete post-load host state: CPU tensors,
        prompt cache and llama-server context checkpoint copies.
        """
        self.host_tensor_mb = self._host_tensor_estimate_mb()
        projector_mb = (
            self.meta.mmproj_size_mb
            if self.vision_enabled and self.mmproj_offload == "n" else 0
        )
        kv_host_mb = 0
        if self.kv_offload == "n":
            kv_host_mb = math.ceil(
                self._cache_bytes_for_context(self.ctx, self.cache_k) / 1048576
            )
        self._plan_context_state_memory()
        fixed_host_mb = 1024 + self.runtime_overhead_mb
        reserve_mb = self._ram_safety_reserve_mb()
        # A quantidade padrão de checkpoints não pode ser multiplicada
        # cegamente por um snapshot de KV de vários GiB. Limite-a pela
        # capacidade real RAM + NVMe prioritário, preservando uma margem de
        # paginação. Em modelos híbridos o snapshot é pequeno e os 32
        # checkpoints continuam normalmente intactos.
        self.checkpoint_limit_reason = ""
        if self.checkpoint_snapshot_mb > 0 and self.ctx_checkpoints > 0:
            checkpoint_snapshot_mb = int(self.checkpoint_snapshot_mb)
            nvme_free_for_capacity_mb = (
                max(
                    self.hw.swap_nvme_total_mb - self.hw.swap_nvme_used_mb, 0
                )
                if self.hw.swap_nvme_preferred else 0
            )
            capacity_mb = self.hw.ram_avail_mb + nvme_free_for_capacity_mb
            resident_base_mb = (
                max(int(self.meta.size_mb), self.host_tensor_mb)
                + projector_mb + kv_host_mb + self.prompt_cache_peak_mb
                + fixed_host_mb + reserve_mb
            )
            paging_margin_mb = 1024 if nvme_free_for_capacity_mb > 0 else 0
            max_checkpoints = max(
                (capacity_mb - resident_base_mb - paging_margin_mb)
                // checkpoint_snapshot_mb,
                0,
            )
            if self.ctx_checkpoints > max_checkpoints:
                requested_checkpoints = int(self.ctx_checkpoints)
                self.ctx_checkpoints = int(max_checkpoints)
                self._plan_context_state_memory()
                self.checkpoint_limit_reason = (
                    f"checkpoints limitados de {requested_checkpoints} para "
                    f"{self.ctx_checkpoints}: cada snapshot ocupa "
                    f"{checkpoint_snapshot_mb} MB e a capacidade segura "
                    f"RAM+NVMe é {capacity_mb} MB"
                )
                self.host_growth_reason += f" | {self.checkpoint_limit_reason}"
                if self.load_mode_reason.startswith(
                    "mmap — load-mode none exigiria"
                ):
                    # _select_cpu_moe_load_mode() avaliou a capacidade antes
                    # deste limite. Deixe o plano final decidir novamente com
                    # o número efetivo de checkpoints.
                    self.load_mode_reason = ""
        mmap_host_peak_mb = (
            self.host_tensor_mb
            + projector_mb
            + kv_host_mb
            + self.prompt_cache_peak_mb
            + self.checkpoint_peak_mb
            + fixed_host_mb
        )
        resident_host_peak_mb = (
            max(int(self.meta.size_mb), self.host_tensor_mb)
            + projector_mb
            + kv_host_mb
            + self.prompt_cache_peak_mb
            + self.checkpoint_peak_mb
            + fixed_host_mb
        )

        # A non-mmap load needs host backing for the complete GGUF even when
        # tensor overrides move most structural work to CUDA.  Re-check a
        # family-specific decision here because cache/checkpoint settings can
        # be edited after that decision and increase the post-load peak.
        resident_shortfall_mb = max(
            resident_host_peak_mb + reserve_mb - self.hw.ram_avail_mb, 0
        )
        nvme_free_mb = max(
            self.hw.swap_nvme_total_mb - self.hw.swap_nvme_used_mb, 0
        )
        # Cache/checkpoint entries are dormant copies; they may be paged out.
        # The hot CPU experts, projector, host KV and runtime must still fit in
        # physical RAM or decode would continuously fault from the NVMe.
        active_working_set_mb = (
            self.host_tensor_mb + projector_mb + kv_host_mb
            + fixed_host_mb + reserve_mb
        )
        resident_paging_safe = bool(
            resident_shortfall_mb > 0
            and self.hw.swap_nvme_preferred
            and active_working_set_mb <= self.hw.ram_avail_mb
            and nvme_free_mb >= resident_shortfall_mb + 1024
        )
        if not self.load_mode_reason:
            if resident_shortfall_mb == 0 or resident_paging_safe:
                self.load_mode = "none"
                if resident_paging_safe:
                    self.load_mode_reason = (
                        f"none — conjunto ativo {active_working_set_mb - reserve_mb} MB "
                        f"+ reserva {reserve_mb} MB cabe na RAM; pico frio excedente "
                        f"{resident_shortfall_mb} MB cabe no NVMe prioritário"
                    )
                else:
                    self.load_mode_reason = (
                        f"none — pico host residente {resident_host_peak_mb} MB + "
                        f"reserva {reserve_mb} MB <= RAM disponível "
                        f"{self.hw.ram_avail_mb} MB"
                    )
            else:
                self.load_mode = "mmap"
                self.load_mode_reason = (
                    f"mmap — pico residente excederia a RAM em "
                    f"{resident_shortfall_mb} MB; pesos permanecem "
                    "file-backed no NVMe"
                )
        elif (
            self.load_mode == "none"
            and resident_shortfall_mb > 0
            and not resident_paging_safe
        ):
            previous_reason = self.load_mode_reason
            self.load_mode = "mmap"
            self.load_mode_reason = (
                f"mmap — proteção pós-cálculo: load-mode none excederia "
                f"a RAM segura em {resident_shortfall_mb} MB; decisão anterior: "
                f"{previous_reason}"
            )

        self.host_peak_mb = (
            resident_host_peak_mb if self.load_mode == "none"
            else mmap_host_peak_mb
        )
        self.memory_shortfall_mb = max(
            self.host_peak_mb + reserve_mb - self.hw.ram_avail_mb, 0
        )

        model_gap_mb = max(
            self.meta.size_mb + projector_mb + reserve_mb
            - self.hw.ram_avail_mb,
            0,
        )
        emergency_mb = max(model_gap_mb, self.memory_shortfall_mb)
        if emergency_mb > 0:
            # Four GiB of headroom covers allocator bursts and a temporary
            # projector/context transition. Round to 4 GiB blocks.
            requested_mb = emergency_mb + 4096
            requested_gib = max(8, math.ceil(requested_mb / 4096) * 4)
            try:
                disk_free_gib = int(float(self.hw.disk_free_gb))
            except (TypeError, ValueError):
                disk_free_gib = 0
            reclaimable_gib = math.ceil(self.hw.swap_nvme_total_mb / 1024)
            disk_limit = max(
                (disk_free_gib + reclaimable_gib - 32) // 4 * 4, 0
            )
            self.swap_recommended_gib = min(requested_gib, disk_limit, 64)
        else:
            self.swap_recommended_gib = 0

        current_gib = math.ceil(self.hw.swap_nvme_total_mb / 1024)
        if self.swap_recommended_gib <= 0:
            self.swap_plan_reason = (
                "swap NVMe não necessário para o perfil atual; ZRAM continua "
                "como proteção geral"
            )
        elif current_gib >= self.swap_recommended_gib and self.hw.swap_nvme_preferred:
            self.swap_plan_reason = (
                f"swap NVMe ativo {current_gib} GiB atende a recomendação "
                f"dinâmica de {self.swap_recommended_gib} GiB e tem prioridade "
                f"{self.hw.swap_nvme_priority} acima da ZRAM "
                f"{self.hw.swap_zram_priority} | "
                f"{self.host_growth_reason}"
            )
        elif current_gib >= self.swap_recommended_gib:
            self.swap_plan_reason = (
                f"swap NVMe tem {current_gib} GiB, mas prioridade "
                f"{self.hw.swap_nvme_priority} não supera a ZRAM "
                f"{self.hw.swap_zram_priority}; reaplique antes de usar "
                "load-mode none"
            )
        else:
            self.swap_plan_reason = (
                f"recomendado {self.swap_recommended_gib} GiB no NVMe para "
                f"lacuna host de {emergency_mb} MB; {self.host_growth_reason}; "
                "arquivo pré-alocado antes da carga, sem contar como RAM rápida "
                "nem ampliar o contexto"
            )

    def _adapt_qwen35moe(self, requested_ctx: int | None = None) -> bool:
        """Balance long context and GPU execution for Qwen3.5/3.6 MoE."""
        if self.meta.arch.lower() != "qwen35moe" or not self.hw.gpu_detected:
            return False
        # Context length comes from the selected GGUF, never from a model
        # name or a family-specific constant.
        target_ctx = requested_ctx or self.meta.ctx_max
        target_ctx = min(max(int(target_ctx), 1), self.meta.ctx_max)
        available_mb = max(self.hw.gpu_vram_free_mb - self.fit_target, 0)
        layers = max(self.meta.layers, 1)
        if target_ctx < 1 or available_mb <= 0:
            return False

        max_cpu_moe = layers
        if self._estimate_qwen_moe_gpu_mb(target_ctx, max_cpu_moe) is None:
            return False
        if self._estimate_qwen_moe_gpu_mb(target_ctx, max_cpu_moe) > available_mb:
            return False

        low, high = 0, max_cpu_moe
        while low < high:
            middle = (low + high) // 2
            estimate = self._estimate_qwen_moe_gpu_mb(target_ctx, middle)
            if estimate is not None and estimate <= available_mb:
                high = middle
            else:
                low = middle + 1

        self._commit_measured_moe_plan(layers, low, target_ctx)
        self._select_cpu_moe_load_mode()
        self.ngl_reason = (
            f"todas as {layers} camadas estruturais na GPU; "
            f"{low} camadas de experts na CPU"
        )
        self.n_cpu_moe_reason = (
            f"adaptado por llama-fit-params: {low} camadas CPU, "
            f"contexto alvo {target_ctx}, reserva {self.fit_target} MiB "
            "(estimativa validada por workload local)"
        )
        self.ctx_reason = (
            f"QWEN35MOE adaptado: {target_ctx} tokens | "
            f"{low} camadas MoE CPU | margem VRAM {self.fit_target} MiB"
        )
        self.fit_plan_source = "qwen35moe-measured"
        self.fit_plan_reason = (
            f"plano antecipado: ctx {target_ctx}, {low} camadas MoE CPU; "
            "o comando usa --fit off para preservar esta janela"
        )
        return True

    def _adapt_laguna(self, requested_ctx: int | None = None) -> bool:
        """Fit Laguna XS 2.1's native context without inheriting Qwen rules."""
        if self.meta.arch.lower() != "laguna" or not self.hw.gpu_detected:
            return False
        target_ctx = requested_ctx or self.meta.ctx_max
        target_ctx = min(max(int(target_ctx), 1), self.meta.ctx_max)
        available_mb = max(self.hw.gpu_vram_free_mb - self.fit_target, 0)
        layers = max(self.meta.layers, 1)
        if target_ctx < 1 or available_mb <= 0:
            return False

        # Find the smallest CPU-expert placement that fits the current KV,
        # batch, slot count and live free VRAM.  Structural attention/router
        # layers remain on CUDA; only expert tensors move to host memory.
        maximum = self._estimate_qwen_moe_gpu_mb(target_ctx, layers)
        if maximum is None or maximum > available_mb:
            return False
        low, high = 0, layers
        while low < high:
            middle = (low + high) // 2
            estimate = self._estimate_qwen_moe_gpu_mb(target_ctx, middle)
            if estimate is not None and estimate <= available_mb:
                high = middle
            else:
                low = middle + 1

        self._commit_measured_moe_plan(layers, low, target_ctx)
        self._select_cpu_moe_load_mode()
        # At the native 256K window BF16/Q8 generally places nearly all
        # expert tensors on the host (~25 GiB measured).  mmap avoids a hard
        # resident allocation that would leave no safe RAM margin on 32 GiB.
        if (
            low >= math.ceil(layers * 0.75)
            and self.hw.ram_avail_mb < self.meta.size_mb + 4096
        ):
            self.load_mode = "mmap"
            self.load_mode_reason = (
                "mmap — experts Laguna ocupam quase toda a RAM disponível; "
                "evita alocação residente e OOM do sistema"
            )
        self.ngl_reason = (
            f"{layers}/{layers} camadas estruturais CUDA; "
            f"experts de {low} camadas na CPU"
        )
        self.n_cpu_moe_reason = (
            f"Laguna calculado por llama-fit-params: {low} camadas CPU, "
            f"ctx {target_ctx}, KV {self.cache_k}/{self.cache_v}, "
            f"reserva {self.fit_target} MiB"
        )
        self.ctx_reason = (
            f"Laguna nativo: {target_ctx} tokens | YaRN do GGUF | "
            f"{low} camadas MoE CPU | KV {self.cache_k}/{self.cache_v}"
        )
        self.fit_plan_source = "laguna-measured"
        self.fit_plan_reason = (
            f"plano antecipado: ctx {target_ctx}, {low} camadas MoE CPU; "
            "--fit off preserva o plano calculado"
        )
        return True

    def _adapt_gemma4_moe(self, requested_ctx: int | None = None) -> bool:
        """Keep Gemma 4 routing/attention on CUDA and place only experts on CPU."""
        if not _is_gemma4_moe(self.meta) or not self.hw.gpu_detected:
            return False
        target_ctx = requested_ctx or self.meta.ctx_max
        target_ctx = min(max(int(target_ctx), 1), self.meta.ctx_max)
        available_mb = max(self.hw.gpu_vram_free_mb - self.fit_target, 0)
        layers = max(self.meta.layers, 1)
        if target_ctx < 1 or available_mb <= 0:
            return False

        maximum = self._estimate_qwen_moe_gpu_mb(target_ctx, layers)
        if maximum is None or maximum > available_mb:
            return False
        low, high = 0, layers
        while low < high:
            middle = (low + high) // 2
            estimate = self._estimate_qwen_moe_gpu_mb(target_ctx, middle)
            if estimate is not None and estimate <= available_mb:
                high = middle
            else:
                low = middle + 1

        self._commit_measured_moe_plan(layers, low, target_ctx)
        self._select_cpu_moe_load_mode()
        self.ngl_reason = (
            f"{layers}/{layers} camadas estruturais CUDA; "
            f"experts de {low} camadas na CPU"
        )
        self.n_cpu_moe_reason = (
            f"Gemma 4 calculado por llama-fit-params: {low} camadas CPU, "
            f"ctx {target_ctx}, KV {self.cache_k}/{self.cache_v}, "
            f"reserva {self.fit_target} MiB"
        )
        self.ctx_reason = (
            f"Gemma 4 nativo: {target_ctx} tokens | "
            f"{self.meta.swa_layers} SWA/{self.meta.global_layers} globais | "
            f"{low} camadas MoE CPU | KV {self.cache_k}/{self.cache_v}"
        )
        self.fit_plan_source = "gemma4-measured"
        self.fit_plan_reason = (
            f"plano antecipado: ctx {target_ctx}, {low} camadas MoE CPU; "
            "--fit off preserva a janela e o padrão SWA nativo"
        )
        return True

    def _autotune_runtime_identity(self) -> dict:
        return {
            "build": binary_identity(self.llama_server),
        }

    def resolve_autotune(
        self,
        workload: dict,
        sampler: dict | None = None,
        model_sha256: str = "",
    ) -> dict | None:
        return _AUTOTUNE_CACHE.resolve(
            self.meta.identity(model_sha256),
            self.hw.identity(),
            self._autotune_runtime_identity(),
            workload,
            sampler or {},
        )

    def record_autotune(
        self,
        workload: dict,
        sampler: dict,
        config: dict,
        metrics: dict,
        quality: dict | None = None,
        model_sha256: str = "",
        apply_to_launch: bool = False,
        source: str = "llama-bench",
    ) -> dict:
        digest = model_sha256 or sha256_file(self.meta.path)
        return _AUTOTUNE_CACHE.record(
            self.meta.identity(digest),
            self.hw.identity(),
            self._autotune_runtime_identity(),
            workload,
            sampler,
            config,
            metrics,
            quality,
            apply_to_launch=apply_to_launch,
            source=source,
        )

    def _adapt_qwen3moe(self, requested_ctx: int | None = None) -> bool:
        """Use a high-context, speed-balanced profile for the older Qwen MoE."""
        if self.meta.arch.lower() != "qwen3moe" or not self.hw.gpu_detected:
            return False
        layers = max(self.meta.layers, 1)
        available_mb = max(self.hw.gpu_vram_free_mb - self.fit_target, 0)
        performance_ncm = max(1, min(layers - 1, math.floor(layers * 0.55)))

        if requested_ctx is not None:
            target_ctx = min(max(int(requested_ctx), 1), self.meta.ctx_max)
            if self._estimate_qwen_moe_gpu_mb(target_ctx, performance_ncm) <= available_mb:
                ncm = performance_ncm
            else:
                if self._estimate_qwen_moe_gpu_mb(target_ctx, layers) > available_mb:
                    return False
                low, high = performance_ncm, layers
                while low < high:
                    middle = (low + high) // 2
                    estimate = self._estimate_qwen_moe_gpu_mb(target_ctx, middle)
                    if estimate is not None and estimate <= available_mb:
                        high = middle
                    else:
                        low = middle + 1
                ncm = low
        else:
            ncm = performance_ncm
            if self._estimate_qwen_moe_gpu_mb(self.meta.ctx_max, ncm) <= available_mb:
                target_ctx = self.meta.ctx_max
            else:
                low, high, best = 1, self.meta.ctx_max, 0
                while low <= high:
                    middle = (low + high) // 2
                    estimate = self._estimate_qwen_moe_gpu_mb(middle, ncm)
                    if estimate is not None and estimate <= available_mb:
                        best = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                if best < 1:
                    return False
                target_ctx = max((best // 256) * 256, 1)

        self._commit_measured_moe_plan(layers, ncm, target_ctx)
        self._select_cpu_moe_load_mode()
        self.ngl_reason = (
            f"todas as {layers} camadas estruturais na GPU; "
            f"{ncm} camadas de experts na CPU"
        )
        self.n_cpu_moe_reason = (
            f"adaptado por llama-fit-params: {ncm} camadas CPU, "
            f"contexto alvo {target_ctx} (estimativa; valide por workload)"
        )
        native_note = (
            ""
            if target_ctx >= self.meta.ctx_max
            else f" | nativo {self.meta.ctx_max} excede a VRAM neste perfil"
        )
        self.ctx_reason = (
            f"QWEN3MOE adaptado: {target_ctx} tokens | "
            f"{ncm} camadas MoE CPU | margem VRAM {self.fit_target} MiB"
            f"{native_note}"
        )
        return True

    def _ctx(self):
        hw = self.hw; mt = self.meta
        gpu_kv_available = hw.gpu_detected and hw.gpu_vram_free_mb > 0 and self.ngl > 0
        if gpu_kv_available and self.fit == "y" and self._apply_fit_plan():
            return
        if getattr(mt, "layer_layout_valid", False) and mt.attention_layers == 0:
            self.ctx = mt.ctx_max
            self.kv_fits_vram = True
            self.ctx_reason = (
                f"estado recorrente apenas: {mt.recurrent_layers} camadas SSM; "
                "não há cache KV dependente do contexto"
            )
            return
        curve = self._fit_memory_curve() if gpu_kv_available and self.ngl >= mt.layers else None
        if curve:
            intercept, slope, correction = curve
            vision_mb = 0
            if self.vision_enabled:
                vision_mb = mt.mmproj_size_mb + 256
            usable_mb = hw.gpu_vram_free_mb - self.fit_target - vision_mb
            fitted_ctx = math.floor((usable_mb - intercept) / slope)
            fitted_ctx = max(min(fitted_ctx, mt.ctx_max), 1)
            self.ctx = max((fitted_ctx // 256) * 256, 1)
            self.kv_fits_vram = self.ctx >= 1
            estimated_mb = intercept + slope * self.ctx + vision_mb
            cache_layout = f"{mt.kv_layers} camadas KV"
            if self.spec_type == "draft-mtp":
                cache_layout += " + MTP"
            if self.swa_full == "y":
                cache_layout += " + SWA full"
            self.ctx_reason = (
                f"llama-fit-params: {self.ctx} tokens | uso estimado {estimated_mb:.0f} MB | "
                f"{cache_layout} | correção allocator {correction} MB | "
                f"margem {self.fit_target} MB"
            )
            return
        recurrent_state_mb = math.ceil(self._recurrent_state_bytes() / 1048576)
        vram_total_usable = max(
            hw.gpu_vram_free_mb - self.fit_target - self.runtime_overhead_mb, 0
        ) if gpu_kv_available else 0
        if gpu_kv_available:
            vram_total_usable = max(vram_total_usable - recurrent_state_mb, 0)

        if self.ngl >= mt.layers:
            model_vram_mb = math.ceil(self._gpu_weight_bytes(self.ngl) / 1048576)
            model_ram_mb = 0
        else:
            model_vram_mb = math.ceil(self._gpu_weight_bytes(self.ngl) / 1048576)
            model_ram_mb = int(max(mt.size_mb - model_vram_mb, 0) * 1.05)

        # O peso do projetor já foi descontado; esta margem cobre ativações temporárias.
        mmproj_mb = mt.mmproj_size_mb if self.vision_enabled else 0
        vision_overhead = 256 if self.vision_enabled else 0
        vram_for_kv = vram_total_usable - model_vram_mb - mmproj_mb - vision_overhead

        bpt = self._bpt(self.cache_k)

        if vram_for_kv > 256:
            self.kv_fits_vram = True
            ctx_vram = self._max_context_for_cache_bytes(
                int(vram_for_kv * 1048576), self.cache_k
            )
            self.ctx = min(ctx_vram, mt.ctx_max)
        else:
            self.kv_fits_vram = False
            ram_reserved = model_ram_mb + 1024
            ram_for_kv = max(hw.ram_avail_mb - ram_reserved, 0)
            ctx_ram = self._max_context_for_cache_bytes(
                int(ram_for_kv * 1048576), self.cache_k
            )
            self.ctx = min(ctx_ram, mt.ctx_max)

        self.ctx = min(max(self.ctx, 1), max(mt.ctx_max, 1))
        if self.kv_fits_vram:
            cache_mb = self._cache_bytes_for_context(
                self.ctx, self.cache_k
            ) // 1048576
            self.ctx_reason = (
                f"VRAM: {self.ctx} tokens | {cache_mb} MB "
                f"de {int(vram_for_kv)} MB disponíveis | {mt.kv_layers} camadas KV | "
                f"workspace {self.runtime_overhead_mb} MB | estado recorrente "
                f"{recurrent_state_mb} MB | margem {self.fit_target} MB"
            )
        else:
            self.ctx_reason = f"RAM: {self.ctx} tokens | modelo usa {model_ram_mb} MB de RAM"

    def calculate(self) -> None:
        # First pass determines whether the model can be fully offloaded. Batch
        # sizing then refines the CUDA workspace before the final memory pass.
        self._ngl()
        self._batch()
        self._update_runtime_overhead()
        self._ngl()
        self._cache()
        self._spec()
        self._swa_full()
        if (
            not self._adapt_laguna()
            and not self._adapt_gemma4_moe()
            and not self._adapt_qwen35moe()
            and not self._adapt_qwen3moe()
        ):
            self._ctx()
        self._flash()
        self._threads()
        self._threads_batch()
        self._kv_unified()
        self._kv_offload()
        self._numa()
        self._repack()
        self._device()
        self._backend_sampling()
        self._poll()
        self._omni()
        self._mmproj_offload()
        self._mtmd_batch()
        self._cache_reuse()
        self._image_tokens()
        self._reasoning_preserve()
        # ``ctx`` is the effective window selected from the current hardware
        # plan.  Keep the fit floor synchronized with it so the command and
        # the running server cannot silently fall back to the constructor's
        # generic 4096-token floor.  The upper bound is always the context
        # declared by the selected GGUF metadata.
        self.fit_ctx = min(
            max(int(self.ctx), 1), max(int(self.meta.ctx_max), 1)
        )
        self._plan_host_memory()

    def recalculate_memory(self, cache_k: str, cache_v: str, batch: int, ubatch: int) -> None:
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.cache_reason = f"selecionado pelo usuário: K={cache_k}, V={cache_v}"
        self.batch = batch
        self.ubatch = ubatch
        self.batch_reason = f"selecionado pelo usuário: batch={batch}, micro-batch={ubatch}"
        self._update_runtime_overhead()
        self._ngl()
        if (
            not self._adapt_laguna()
            and not self._adapt_gemma4_moe()
            and not self._adapt_qwen35moe()
            and not self._adapt_qwen3moe()
        ):
            self._ctx()
        self._flash()
        self._kv_offload()
        self.fit_ctx = min(
            max(int(self.ctx), 1), max(int(self.meta.ctx_max), 1)
        )
        self._plan_host_memory()

    def _update_runtime_overhead(self) -> None:
        # Calibrated on CUDA with MTP: 2048/512 consumes about 384 MiB of
        # graph/runtime workspace. Both logical and physical batch contribute.
        base = 128
        batch_part = math.ceil(128 * max(self.batch, 1) / 2048)
        ubatch_part = math.ceil(128 * max(self.ubatch, 1) / 512)
        self.runtime_overhead_mb = max(base + batch_part + ubatch_part, 160)

    def _ngl(self) -> None:
        hw = self.hw; mt = self.meta
        if hw.gpu_detected and hw.gpu_vram_free_mb > 512:
            mmproj_mb = mt.mmproj_size_mb if self.vision_enabled else 0
            usable = (
                hw.gpu_vram_free_mb - self.fit_target
                - self.runtime_overhead_mb - mmproj_mb
            )
            model_needed = mt.size_mb
            if usable >= model_needed:
                self.ngl = 999
                self.fit = "n"
                self.ngl_reason = (
                    f"all — modelo inteiro + workspace cabem na VRAM "
                    f"({hw.gpu_vram_free_gb} GB livre)"
                )
            elif mt.size_mb > 0 and mt.layers > 0:
                self.fit = "y"
                layer_bytes = getattr(mt, "layer_weight_bytes", [])
                if len(layer_bytes) == mt.layers and any(layer_bytes):
                    total = getattr(mt, "output_weight_bytes", 0)
                    lf = 1 if total else 0
                    for size in reversed(layer_bytes):
                        if total + size > usable * 1048576:
                            break
                        total += size
                        lf += 1
                    self.ngl = lf
                    self.ngl_reason = (
                        f"{lf}/{mt.layers} camadas finais cabem na VRAM "
                        "(tamanhos reais por bloco)"
                    )
                else:
                    mpl = max(mt.size_mb // mt.layers, 1)
                    lf  = min(max(usable // mpl, 0), mt.layers)
                    self.ngl = lf
                    self.ngl_reason = f"{lf}/{mt.layers} camadas cabem na VRAM livre"
        else:
            self.ngl = 0
            self.ngl_reason = "sem GPU / VRAM insuficiente — modo CPU only"

    def _cache(self) -> None:
        q = self.meta.quant.upper()
        hw = self.hw; mt = self.meta
        full_gpu = hw.gpu_detected and self.ngl >= mt.layers
        vram_avail = hw.gpu_vram_free_mb if hw.gpu_detected else 0
        residual = (
            vram_avail - mt.size_mb
            - (mt.mmproj_size_mb if self.vision_enabled else 0)
            - self.runtime_overhead_mb - self.fit_target
            if full_gpu else 0
        )
        memory_tight = not full_gpu or residual < 2048
        # f16 é o novo padrão do llama.cpp — melhor precisão com Flash Attention
        if _is_glm47_flash(self.meta):
            # GLM's MLA cache requires identical K/V types.  Q8_0 is the only
            # practical way to retain its native 202,752-token window on the
            # 12 GiB RTX 3060 while keeping the structural layers on CUDA.
            # F16/F16 remains available manually as the maximum-precision,
            # lower-context/lower-GPU-offload alternative.
            ck=cv="q8_0"; r=(
                "q8_0/q8_0 — MLA exige K=V; preserva os 202752 tokens "
                "nativos e o maior offload CUDA na RTX 3060"
            )
        elif self.meta.arch.lower() == "laguna":
            # Quality-first cache for code/reasoning.  On the local RTX 3060
            # at 262144 tokens llama-fit-params measured BF16/Q8 at
            # 10695 MiB with all expert blocks on CPU.  Q8/Q8 remains a valid
            # manual speed/memory profile and triggers a fresh placement.
            ck="bf16"; cv="q8_0"; r=(
                "bf16/q8_0 — K conservado para qualidade; V Q8 preserva "
                "o contexto nativo de 262144 tokens"
            )
        elif _is_gemma4_moe(self.meta):
            # Local 262K fit measurements: BF16/Q8 keeps one additional
            # expert layer on CUDA versus F16/F16 while preserving K.
            ck="bf16"; cv="q8_0"; r=(
                "bf16/q8_0 — Gemma 4 26B-A4B medido em 262K; "
                "K preservado e uma camada MoE extra permanece na GPU"
            )
        elif self.meta.arch.lower() == "qwen35moe":
            # Empirical profile for Qwen3.6-35B-A3B on the Xeon E5-2682 v4 +
            # RTX 3060: keep K in BF16 (the more sensitive side of attention)
            # and quantize V to Q8_0.  This preserved the reasoning score in
            # our fixed-seed comparison while keeping essentially full speed.
            ck="bf16"; cv="q8_0"; r=f"bf16/q8_0 — perfil empírico de qualidade/velocidade para {q}"
        elif self.meta.arch.lower() == "qwen35":
            # Ornith-1.5-9B is a dense Qwen3.5 hybrid model. Keep the more
            # sensitive K cache in BF16 and V in Q8_0, matching the proven
            # quality profile used for the user's Qwen3.6 model. Its model
            # profile moves KV to RAM so the native 262K window remains
            # available on a 12 GB card.
            ck="bf16"; cv="q8_0"; r=f"bf16/q8_0 — K preservado e V quantizado para {q} híbrido"
        elif self.meta.arch.lower() == "qwen3moe":
            ck=cv="q5_1"; r=f"q5_1 — cache equilibrado para {q} híbrido QWEN MoE"
        elif re.match(r"Q[23]_|IQ[23]_", q): ck=cv="q8_0"; r=f"q8_0 — modelo {q} baixa precisão"
        elif re.match(r"Q4_|IQ4_",        q):
            if not memory_tight:
                ck=cv="f16"; r=f"f16 — modelo cabe na VRAM ({vram_avail} MB livre), cache de alta precisão"
            else:
                ck=cv="q8_0"; r=f"q8_0 — reduz uso de VRAM preservando boa precisão com {q}"
        elif re.match(r"Q5_|IQ5_", q) and memory_tight:
            ck=cv="q8_0"; r=f"q8_0 — VRAM residual limitada; prioriza contexto com modelo {q}"
        elif re.match(r"Q5_|IQ5_",        q): ck=cv="f16";  r=f"f16 — melhor precisão para {q}"
        elif re.match(r"Q6_", q) and memory_tight:
            ck=cv="q8_0"; r=f"q8_0 — modelo {q} híbrido; preserva contexto e VRAM"
        elif re.match(r"Q6_",             q): ck=cv="f16";  r=f"f16 — melhor precisão para {q}"
        elif re.match(r"Q8_|IQ8_",        q): ck=cv="f16";  r=f"f16 — FA obrigatório; mesma precisão que {q}"
        elif q in ("BF16","F16","FP16"):       ck=cv="f16";  r=f"f16 — mesma precisão que {q}"
        else:                                  ck=cv="f16";  r=f"f16 — padrão (FA compatível)"
        self.cache_k = ck; self.cache_v = cv; self.cache_reason = r

    def _flash(self) -> None:
        if not self.hw.gpu_detected:
            self.flash = "n"; self.flash_reason = "desabilitado — sem GPU"
        elif self.cache_k not in ("f16","bf16","fp16") or self.cache_v not in ("f16","bf16","fp16"):
            self.flash = "y"
            self.flash_reason = (
                f"OBRIGATÓRIO para KV quantizado ({self.cache_k}/{self.cache_v})"
            )
        elif _is_glm47_flash(self.meta):
            self.flash = "n"
            self.flash_reason = (
                "desabilitado com KV F16/BF16 — recomendação de desempenho "
                "do quantizador GLM; KV quantizado ainda exige Flash Attention"
            )
        else:
            self.flash = "y"
            self.flash_reason = "habilitado — força CUDA Flash Attention para máximo throughput"

    def _threads(self) -> None:
        if self.meta.arch.lower() == "laguna" and self.hw.gpu_detected:
            self.threads = min(max(self.hw.cpu_cores, 1), 16)
            self.threads_reason = (
                f"{self.threads} threads físicas — Laguna executa experts MoE na CPU"
            )
            return
        if (self.meta.arch.lower() in {"qwen35", "qwen35moe"} or _is_glm47_flash(self.meta) or _is_gemma4_moe(self.meta)) and self.hw.gpu_detected and self.hw.cpu_cores >= 16:
            self.threads = 12
            self.threads_reason = "12 threads — perfil medido no Xeon E5-2682 v4 com RTX 3060"
            return
        if self.ngl >= self.meta.layers:
            t = min(max(self.hw.cpu_cores, 1), 16)
            self.threads = t
            self.threads_reason = f"{t} threads — GPU domina geração, CPU faz prefill/batch"
        else:
            t = min(max(self.hw.cpu_cores, 1), 32)
            self.threads = t
            self.threads_reason = f"{t} cores físicos — melhor para geração híbrida"

    def _batch(self) -> None:
        if (self.meta.arch.lower() in {"qwen35", "qwen35moe"} or _is_gemma4_moe(self.meta)) and self.hw.gpu_detected:
            self.batch = 2048; self.ubatch = 512
            self.batch_reason = "2048/512 — perfil medido; passou em carga com contexto 256K"
            return
        if self.ngl >= self.meta.layers:
            self.batch = 2048; self.ubatch = 512
            self.batch_reason = "2048/512 — padrão seguro para 12 GB"
        elif self.hw.gpu_vram_mb >= 8192:
            self.batch = 2048; self.ubatch = 512
            self.batch_reason = "2048/512 — GPU parcial"
        else:
            self.batch = 2048; self.ubatch = 256
            self.batch_reason = "2048 — conservador / CPU"

    # ── Calcula o contexto máximo baseado na VRAM disponível ──────────────────────────────────
    def _kv_unified(self) -> None:
        self.kv_unified = self.parallel > 1
        self.kv_reason = (
            "buffer KV único compartilhado entre slots"
            if self.kv_unified else "desnecessário com apenas um slot"
        )

    def _poll(self) -> None:
        self.poll = 50 if self.hw.gpu_detected else 100

    def _threads_batch(self) -> None:
        if self.meta.arch.lower() == "laguna" and self.hw.gpu_detected:
            self.threads_batch = min(max(self.hw.cpu_threads, 1), 32)
            self.threads_batch_reason = (
                f"{self.threads_batch} threads — prefill híbrido Laguna"
            )
            return
        if (self.meta.arch.lower() in {"qwen35", "qwen35moe"} or _is_glm47_flash(self.meta) or _is_gemma4_moe(self.meta)) and self.hw.gpu_detected and self.hw.cpu_cores >= 16:
            self.threads_batch = 16
            self.threads_batch_reason = "16 threads — perfil medido para prefill no Xeon E5-2682 v4"
            return
        if self.ngl >= self.meta.layers:
            t = min(max(self.hw.cpu_threads, 1), 32)
            self.threads_batch = t
            self.threads_batch_reason = f"{t} threads — prefill escala com hyperthreading"
        elif self.hw.cpu_cores >= 16:
            self.threads_batch = min(self.hw.cpu_threads, 32)
            self.threads_batch_reason = f"{self.threads_batch} threads — CPU-bound"
        else:
            self.threads_batch = 0
            self.threads_batch_reason = "0 = mesmo que --threads (padrão)"

    def _kv_offload(self) -> None:
        if self.ngl > 0 and self.hw.gpu_detected and self.kv_fits_vram:
            self.kv_offload = "y"
            self.kv_offload_reason = "habilitado — GPU gerencia o cache KV"
        else:
            self.kv_offload = "n"
            self.kv_offload_reason = "desabilitado — CPU only"

    def _numa(self) -> None:
        if self.hw.numa_nodes > 1:
            self.numa = "distribute"
            self.numa_reason = f"distribute — {self.hw.numa_nodes} nós NUMA"
        else:
            self.numa = "none"
            self.numa_reason = "none — apenas um nó NUMA detectado"

    def _repack(self) -> None:
        self.repack = "y"
        self.repack_reason = "habilitado — pesos reorganizados para acesso eficiente na GPU"

    def _device(self) -> None:
        if self.hw.gpu_detected:
            self.device = "CUDA0"
            self.device_reason = f"CUDA0 — {self.hw.gpu_model}"
        else:
            self.device = ""
            self.device_reason = "sem GPU — CPU only"

    def _backend_sampling(self) -> None:
        if self.hw.gpu_detected:
            self.backend_sampling = "y"
            self.backend_sampling_reason = (
                "habilitado — amostragem executada pelo backend CUDA "
                "quando suportada"
            )
        else:
            self.backend_sampling = "n"
            self.backend_sampling_reason = "desabilitado — sem GPU/backend acelerador"

    def _mmproj_offload(self) -> None:
        if self.vision_enabled and self.meta.mmproj_file and self.meta.mmproj_valid and self.hw.gpu_detected:
            # The projector is allocated after the language model.  Decide by
            # measured resources, not by a list of model names: whenever the
            # model alone is larger than the usable device, native Fit will
            # consume nearly all VRAM and the MMProj must remain on CPU.
            usable_vram_mb = max(
                self.hw.gpu_vram_free_mb
                - self.fit_target
                - self.runtime_overhead_mb,
                0,
            )
            model_requires_hybrid_placement = (
                self.meta.size_mb > usable_vram_mb
            )
            projector_would_exceed_device = (
                self.meta.size_mb + self.meta.mmproj_size_mb
                > usable_vram_mb
            )
            if model_requires_hybrid_placement or projector_would_exceed_device:
                self.mmproj_offload = "n"
                self.mmproj_offload_reason = (
                    f"desabilitado — modelo {self.meta.size_mb} MB + MMProj "
                    f"{self.meta.mmproj_size_mb} MB excedem os "
                    f"{usable_vram_mb} MB CUDA utilizáveis; MMProj na CPU "
                    "preserva pesos, KV e workspace na GPU"
                )
                return
            estimated_mb = None
            if self.meta.arch.lower() == "qwen35moe" and self.n_cpu_moe > 0:
                estimated_mb = self._estimate_qwen_moe_gpu_mb(self.ctx, self.n_cpu_moe)
            available_mb = max(self.hw.gpu_vram_free_mb - self.fit_target, 0)
            projector_mb = max(int(self.meta.mmproj_size_mb or 0), 0)
            if estimated_mb is not None and estimated_mb + projector_mb > available_mb:
                self.mmproj_offload = "n"
                self.mmproj_offload_reason = (
                    f"desabilitado — mmproj de {projector_mb} MB não cabe na margem CUDA "
                    f"({estimated_mb} + {projector_mb} > {available_mb} MB); CPU evita OOM"
                )
            else:
                self.mmproj_offload = "y"
                self.mmproj_offload_reason = "habilitado — projetor cabe na margem CUDA estimada"
        elif self.meta.mmproj_file and self.meta.mmproj_valid:
            self.mmproj_offload = "n"
            self.mmproj_offload_reason = "desabilitado — visão disponível, mas inativa"
        else:
            self.mmproj_offload = "n"
            self.mmproj_offload_reason = "desabilitado — sem projetor ou sem GPU"

    def _mtmd_batch(self) -> None:
        if self.vision_enabled and self.meta.mmproj_file:
            if self.meta.arch.lower() == "gemma4":
                self.mtmd_batch_max = 512
                self.mtmd_batch_reason = (
                    "512 tokens — cobre o máximo efetivo de 280 tokens "
                    "do projetor Gemma 4 no libmtmd atual"
                )
            else:
                self.mtmd_batch_max = 1024
                self.mtmd_batch_reason = "1024 tokens — padrão para modelos multimodais"
        elif self.meta.mmproj_file:
            self.mtmd_batch_max = 0
            self.mtmd_batch_reason = "0 — visão disponível, mas inativa"
        else:
            self.mtmd_batch_max = 0
            self.mtmd_batch_reason = "0 — modelo não multimodal"

    def _cache_reuse(self) -> None:
        self.cache_reuse = 0
        self.cache_reuse_reason = "0 — desabilitado (economiza VRAM)"

    def _image_tokens(self) -> None:
        mt = self.meta
        # Sempre lê os metadados; decide image-min-tokens conforme a arquitetura.
        # Qwen-VL (e derivados) exigem >=1024 image tokens para grounding,
        # conforme aviso emitido pelo próprio llama-server.
        if self.vision_enabled and mt.mmproj_file and "QWEN" in (mt.arch or "").upper():
            self.image_min_tokens = 1024
            self.image_min_tokens_reason = (
                "1024 — Qwen-VL exige >=1024 image tokens p/ grounding "
                "(aviso do llama-server)"
            )
        elif self.vision_enabled and mt.mmproj_file:
            self.image_min_tokens = 0
            self.image_min_tokens_reason = (
                "0 — multimodal, mas arquitetura sem exigência conhecida de image tokens"
            )
        elif mt.mmproj_file:
            self.image_min_tokens = 0
            self.image_min_tokens_reason = "0 — visão disponível, mas inativa"
        else:
            self.image_min_tokens = 0
            self.image_min_tokens_reason = "0 — modelo sem visão (sem mmproj)"

    def _swa_full(self) -> None:
        arch = self.meta.arch.lower()
        if arch == "laguna":
            self.swa_full = "n"
            self.swa_reason = (
                f"desabilitado — preserva SWA de {self.meta.sliding_window or 512} "
                "tokens; SWA full desperdiçaria VRAM"
            )
        elif arch == "gemma4":
            self.swa_full = "n"
            self.swa_reason = (
                f"desabilitado — preserva {self.meta.swa_layers or 0} camadas "
                f"SWA de {self.meta.sliding_window or 1024} tokens; "
                "--swa-full aumentaria a KV sem benefício para o contexto nativo"
            )
        elif "gemma" in arch:
            self.swa_full = "y"
            self.swa_reason = "habilitado — Gemma usa SWA full cache"
        else:
            self.swa_full = "n"
            self.swa_reason = "desabilitado — modelo sem SWA"

    def _omni(self) -> None:
        has_vision = bool(self.meta.mmproj_has_vision)
        has_audio_input = bool(self.meta.mmproj_has_audio)
        has_generated_audio = bool(self.meta.mmproj_has_gen_audio)
        # Backward compatibility for synthetic metadata and old cached
        # profiles created before modality-aware MMProj inspection.
        if self.meta.mmproj_valid and not (
            has_vision or has_audio_input or has_generated_audio
        ):
            has_vision = True
        modality_names = []
        if has_vision:
            modality_names.extend(("imagem", "vídeo"))
        if has_audio_input:
            modality_names.append("áudio de entrada")
        if has_generated_audio:
            modality_names.append("áudio de saída")
        modality_text = ", ".join(modality_names) or "multimodal"
        if self.vision_enabled and self.meta.mmproj_file and self.meta.mmproj_valid:
            self.omni = "y"
            self.omni_reason = (
                f"{modality_text} ativos: "
                f"{os.path.basename(self.meta.mmproj_file)}"
            )
        elif self.meta.mmproj_file and self.meta.mmproj_valid:
            self.omni = "n"
            self.omni_reason = (
                f"{modality_text} disponíveis, desativados para maximizar contexto: "
                f"{os.path.basename(self.meta.mmproj_file)}"
            )
        elif self.meta.mmproj_file:
            self.omni = "n"
            self.omni_reason = f"projetor inválido/incompatível: {os.path.basename(self.meta.mmproj_file)}"
        else:
            self.omni = "n"
            self.omni_reason = "nenhum projetor detectado"
        # ``audio`` controls a separate output vocoder. Audio *input* is part
        # of the MMProj and becomes active together with ``omni``.
        self.audio = "y" if self.meta.vocoder_file else "n"

    def _spec(self) -> None:
        mt = self.meta
        if _is_glm47_flash(mt):
            self.spec_type = "none"
            self.spec_type_reason = (
                "none — este GGUF não inclui cabeça MTP; n-gram não teve "
                "ganho validado para raciocínio GLM"
            )
            self.spec_draft_n_max = 0
            self.spec_draft_n_max_reason = "0 — sem draft validado"
            self.spec_draft_p_min = 0.0
            self.spec_draft_p_min_reason = "0.0 — sem draft"
        elif _is_gemma4_moe(mt):
            self.spec_type = "none"
            self.spec_type_reason = (
                "none — o GGUF alvo não contém o assistant/drafter Gemma 4; "
                "n-gram ainda não foi validado neste hardware"
            )
            self.spec_draft_n_max = 0
            self.spec_draft_n_max_reason = "0 — sem drafter compatível carregado"
            self.spec_draft_p_min = 0.0
            self.spec_draft_p_min_reason = "0.0 — sem draft"
        elif mt.arch.lower() in {"qwen35", "qwen35moe"} and self.hw.gpu_vram_mb < 16384:
            self.spec_type = "none"
            if mt.arch.lower() == "qwen35":
                self.spec_type_reason = (
                    "none — MTP cria um segundo contexto CUDA; em GPU abaixo de 16 GB "
                    "o buffer extra pode causar OOM com batch alto"
                )
            else:
                self.spec_type_reason = (
                    "none — MTP tem baixa aceitação com CPU-MoE nesta VRAM; "
                    "decode direto é mais rápido"
                )
            self.spec_draft_n_max = 0
            self.spec_draft_n_max_reason = "0 — MTP adaptativo desabilitado"
            self.spec_draft_p_min = 0.0
            self.spec_draft_p_min_reason = "0.0 — sem draft"
        elif mt.has_mtp:
            self.spec_type = "draft-mtp"
            self.spec_type_reason = f"draft-mtp — modelo tem {mt.n_layer_nextn} cabeças MTP (self-speculative)"
            self.spec_draft_n_max = 3
            self.spec_draft_n_max_reason = "3 — padrão llama.cpp para MTP"
            self.spec_draft_p_min = 0.0
            self.spec_draft_p_min_reason = "0.0 — sem filtro de probabilidade (greedy)"
        elif self.hw.gpu_detected and self.hw.gpu_vram_free_mb >= 4096:
            self.spec_type = "ngram-mod"
            self.spec_type_reason = "ngram-mod — especulação sem modelo extra"
            self.spec_draft_n_max = 0
            self.spec_draft_n_max_reason = "gerenciado pelo --spec-default"
            self.spec_draft_p_min = 0.0
            self.spec_draft_p_min_reason = "0.0 — sem filtro"
        else:
            self.spec_type = "none"
            self.spec_type_reason = "none — sem GPU/VRAM insuficiente para speculative"
            self.spec_draft_n_max = 0
            self.spec_draft_n_max_reason = "0 — desabilitado"

    def _reasoning_preserve(self) -> None:
        if self.meta.supports_reasoning_preserve:
            self.reasoning_preserve = "y"
            self.reasoning_preserve_reason = (
                "habilitado — o chat template preserva reasoning_content no histórico"
            )
        else:
            self.reasoning_preserve = "auto"
            self.reasoning_preserve_reason = "auto — capacidade não confirmada no template"

    # ── Monta o comando final ────────────────────────────────
    def build_cmd(self, final: dict) -> list:
        fit_val = final.get("fit", "y")
        fit_on = isinstance(fit_val, str) and fit_val.lower() == "y"
        try:
            n_cpu_moe = int(final.get("n_cpu_moe", 0))
        except (TypeError, ValueError):
            n_cpu_moe = 0
        explicit_cpu_moe = str(final.get("cpu_moe", "n")).lower() == "y" or n_cpu_moe > 0
        if fit_on and explicit_cpu_moe:
            # llama.cpp rejects --fit after --n-cpu-moe because the latter
            # installs tensor overrides. The adaptive plan is already fitted.
            fit_on = False
        # llama.cpp only reduces the context during --fit when n_ctx is left
        # unset. Passing --ctx-size here would mark it as user-defined and
        # force the fitter to keep an oversized context, which then makes it
        # spill model layers to host RAM. With Fit enabled, --fit-ctx is the
        # minimum context and llama.cpp chooses the largest context that fits.
        cmd = [
            self.llama_server,
            "-m",            final.get("model_path", ""),
        ]
        # If the Fit floor equals the requested context, the user/model
        # profile is explicitly asking for a fixed window. Pass it through so
        # /props, the process command and the UI all describe the same value;
        # Fit may still move tensors to CPU to satisfy that fixed window.
        fixed_fit_context = False
        if fit_on:
            try:
                fixed_fit_context = int(final.get("fit_ctx", 0)) >= int(final["ctx"])
            except (TypeError, ValueError, KeyError):
                fixed_fit_context = False
        if not fit_on or fixed_fit_context:
            cmd += ["--ctx-size", str(final["ctx"])]
        cmd += [
            "--parallel",    str(final["parallel"]),
            "--cache-type-k",final["cache_k"],
            "--cache-type-v",final["cache_v"],
            "--batch-size",  str(final["batch"]),
            "--ubatch-size", str(final["ubatch"]),
            "--threads",     str(final["threads"]),
            "--host",        final["host"],
            "--port",        str(final["port"]),
        ]
        # 'auto' is the default sentinel and still lets llama.cpp's --fit adapt VRAM.
        if fit_on:
            cmd += ["--n-gpu-layers", "auto"]
        else:
            ngl = final["ngl"]
            if isinstance(ngl, int) and ngl >= self.meta.layers:
                ngl = "all"
            cmd += ["--n-gpu-layers", str(ngl)]

        # Threads batch (0 = same as threads)
        try:
            tb = int(final.get("threads_batch", 0))
        except (TypeError, ValueError):
            tb = 0
        if tb > 0:
            cmd += ["--threads-batch", str(tb)]

        # KV unified
        if final.get("kv_unified"):
            cmd.append("--kv-unified")

        # KV offload is always explicit. Besides making the generated command
        # auditable, this prevents LLAMA_ARG_KV_OFFLOAD inherited by the web
        # or desktop process from silently changing the selected placement.
        if final.get("kv_offload") == "n":
            cmd.append("--no-kv-offload")
        else:
            cmd.append("--kv-offload")

        # Split mode (GPU distribution)
        if final.get("split_mode"):
            cmd += ["--split-mode", final["split_mode"]]

        # Device (CUDA device IDs)
        dev = final.get("device", "")
        if dev and dev.lower() != "auto":
            cmd += ["--device", dev]

        # Poll
        if final.get("poll"):
            cmd += ["--poll", str(final["poll"])]

        # Flash attention
        flash_val = final.get("flash", "auto")
        if flash_val.lower() == "y":
            cmd += ["--flash-attn", "on"]
        elif flash_val.lower() == "n":
            cmd += ["--flash-attn", "off"]
        elif flash_val.lower() == "auto":
            cmd += ["--flash-attn", "auto"]

        # NUMA
        numa_val = final.get("numa", "none")
        if numa_val and numa_val.lower() != "none":
            cmd += ["--numa", numa_val]

        # Repack
        if final.get("repack") == "n":
            cmd.append("--no-repack")

        # No host buffer
        if final.get("no_host") == "y":
            cmd.append("--no-host")

        # Direct I/O
        if final.get("direct_io") == "y":
            cmd.append("--direct-io")

        # SWA full cache (Gemma models)
        if final.get("swa_full") == "y":
            cmd.append("--swa-full")

        # A manual fit-off launch still needs the tensor-level MoE placement
        # calculated by llama-fit-params for hybrid models.
        if (
            not fit_on
            and self.fit_plan_overrides
            and int(final.get("ctx", 0)) == self.ctx
        ):
            cmd += ["--override-tensor", self.fit_plan_overrides]

        # Cache reuse (KV shifting)
        cr = int(final.get("cache_reuse", 0))
        if cr > 0:
            cmd += ["--cache-reuse", str(cr)]

        # Sampling: valores explícitos preservam 0/1 como opções de desativação.
        cmd += ["--temp", str(final.get("temp", 0.8))]
        cmd += ["--top-k", str(final.get("top_k", 40))]
        cmd += ["--top-p", str(final.get("top_p", 0.95))]
        cmd += ["--min-p", str(final.get("min_p", 0.05))]
        rp_val = float(final.get("repeat_penalty", 0))
        if rp_val > 0 and rp_val != 1.0:
            cmd += ["--repeat-penalty", str(rp_val)]
        pp_val = float(final.get("presence_penalty", 0))
        if pp_val != 0.0:
            cmd += ["--presence-penalty", str(pp_val)]

        # Load mode (substitui --mlock, --no-mmap, --direct-io)
        lm = final.get("load_mode", "mmap")
        if lm and lm != "mmap":
            cmd += ["--load-mode", lm]
        elif final.get("mlock") == "y":
            cmd += ["--load-mode", "mlock"]
        elif final.get("no_mmap") == "y":
            cmd += ["--load-mode", "none"]
        elif final.get("direct_io") == "y":
            cmd += ["--load-mode", "dio"]
        tensor_read_lazy = str(final.get("tensor_read_lazy", "auto")).lower()
        if tensor_read_lazy != "auto":
            lazy_flag = _server_lazy_mode_flag(self.llama_server)
            if not lazy_flag:
                raise ValueError(
                    "O llama-server selecionado nao suporta --lazy-mode "
                    "nem --tensor-read-lazy"
                )
            cmd += [lazy_flag, tensor_read_lazy]

        # Sampling avançado
        seed = int(final.get("seed", -1))
        if seed >= 0:
            cmd += ["--seed", str(seed)]
        if final.get("ignore_eos") == "y":
            cmd.append("--ignore-eos")
        sl = final.get("sampler_seq", "")
        if sl:
            cmd += ["--sampler-seq", sl]
        rln = int(final.get("repeat_last_n", 64))
        if rln != 64:
            cmd += ["--repeat-last-n", str(rln)]
        fp_val = float(final.get("frequency_penalty", 0))
        if fp_val != 0.0:
            cmd += ["--frequency-penalty", str(fp_val)]
        # DRY sampling
        dry_mult = float(final.get("dry_multiplier", 0))
        if dry_mult > 0:
            cmd += ["--dry-multiplier", str(dry_mult)]
            cmd += ["--dry-base", str(final.get("dry_base", 1.75))]
            cmd += ["--dry-allowed-length", str(final.get("dry_allowed_length", 2))]
            dpn = int(final.get("dry_penalty_last_n", -1))
            if dpn >= 0:
                cmd += ["--dry-penalty-last-n", str(dpn)]
        # Top-n-sigma
        tns = float(final.get("top_nsigma", -1))
        if tns >= 0:
            cmd += ["--top-n-sigma", str(tns)]
        # Typical-p
        tp = float(final.get("typical_p", 1.0))
        if tp < 1.0:
            cmd += ["--typical", str(tp)]
        # XTC
        xtc_p = float(final.get("xtc_probability", 0))
        if xtc_p > 0:
            cmd += ["--xtc-probability", str(xtc_p)]
            cmd += ["--xtc-threshold", str(final.get("xtc_threshold", 0.10))]
        # Dynamic temperature
        dr = float(final.get("dynatemp_range", 0))
        if dr > 0:
            cmd += ["--dynatemp-range", str(dr)]
            cmd += ["--dynatemp-exp", str(final.get("dynatemp_exp", 1.0))]
        # Mirostat
        mir = int(final.get("mirostat", 0))
        if mir > 0:
            cmd += ["--mirostat", str(mir)]
            cmd += ["--mirostat-lr", str(final.get("mirostat_lr", 0.10))]
            cmd += ["--mirostat-ent", str(final.get("mirostat_ent", 5.00))]
        # Adaptive-p
        at = float(final.get("adaptive_target", -1))
        if at >= 0:
            cmd += ["--adaptive-target", str(at)]
            cmd += ["--adaptive-decay", str(final.get("adaptive_decay", 0.90))]

        # RoPE scaling
        rst = final.get("rope_scaling_type", "")
        if rst:
            cmd += ["--rope-scaling", rst]
        rs = float(final.get("rope_scale", 0))
        if rs > 0:
            cmd += ["--rope-scale", str(rs)]
        rfb = float(final.get("rope_freq_base", 0))
        if rfb > 0:
            cmd += ["--rope-freq-base", str(rfb)]
        rfs = float(final.get("rope_freq_scale", 0))
        if rfs > 0:
            cmd += ["--rope-freq-scale", str(rfs)]
        # YaRN
        yoc = int(final.get("yarn_orig_ctx", 0))
        if yoc > 0:
            cmd += ["--yarn-orig-ctx", str(yoc)]
        yef = float(final.get("yarn_ext_factor", -1))
        if yef >= 0:
            cmd += ["--yarn-ext-factor", str(yef)]
        yaf = float(final.get("yarn_attn_factor", -1))
        if yaf >= 0:
            cmd += ["--yarn-attn-factor", str(yaf)]
        ybs = float(final.get("yarn_beta_slow", -1))
        if ybs >= 0:
            cmd += ["--yarn-beta-slow", str(ybs)]
        ybf = float(final.get("yarn_beta_fast", -1))
        if ybf >= 0:
            cmd += ["--yarn-beta-fast", str(ybf)]

        # API authentication / SSL
        ak = final.get("api_key", "").strip()
        if ak:
            cmd += ["--api-key", ak]
        akf = final.get("api_key_file", "").strip()
        if akf:
            cmd += ["--api-key-file", akf]
        skf = final.get("ssl_key_file", "").strip()
        if skf:
            cmd += ["--ssl-key-file", skf]
        scf = final.get("ssl_cert_file", "").strip()
        if scf:
            cmd += ["--ssl-cert-file", scf]

        # CORS
        co = final.get("cors_origins", "").strip()
        if co:
            cmd += ["--cors-origins", co]
        cm = final.get("cors_methods", "").strip()
        if cm:
            cmd += ["--cors-methods", cm]
        ch = final.get("cors_headers", "").strip()
        if ch:
            cmd += ["--cors-headers", ch]
        if final.get("cors_credentials") == "n":
            cmd.append("--no-cors-credentials")

        # Server advanced
        th = int(final.get("threads_http", -1))
        if th > 0:
            cmd += ["--threads-http", str(th)]
        sspi = int(final.get("sse_ping_interval", 30))
        if sspi != 30:
            cmd += ["--sse-ping-interval", str(sspi)]
        if final.get("reuse_port") == "y":
            cmd.append("--reuse-port")
        if final.get("offline") == "y":
            cmd.append("--offline")

        # Cont batching / cache prompt
        if final.get("cont_batching") == "n":
            cmd.append("--no-cont-batching")
        if final.get("cache_prompt") == "n":
            cmd.append("--no-cache-prompt")
        if final.get("cache_idle_slots") == "n":
            cmd.append("--no-cache-idle-slots")

        # Model aliases / tags
        al = final.get("alias", "").strip()
        if not al and str(final.get("agent_compat", "n")).lower() == "y":
            # Clientes locais precisam de um model id estável em /v1/models.
            # O stem do GGUF evita que a extensão e o caminho físico virem parte do id.
            al = Path(final.get("model_path", "")).stem
        if al:
            cmd += ["--alias", al]
        tg = final.get("tags", "").strip()
        if tg:
            cmd += ["--tags", tg]

        # Misc server flags
        if final.get("perf") == "y":
            cmd.append("--perf")
        if final.get("check_tensors") == "y":
            cmd.append("--check-tensors")
        if final.get("op_offload") == "n":
            cmd.append("--no-op-offload")
        backend_sampling = str(final.get("backend_sampling", "auto")).lower()
        if backend_sampling == "y" or (
            backend_sampling == "auto" and self.hw.gpu_detected
        ):
            cmd.append("--backend-sampling")
        okv = final.get("override_kv", "").strip()
        if okv:
            cmd += ["--override-kv", okv]
        ssp = final.get("slot_save_path", "").strip()
        if ssp:
            cmd += ["--slot-save-path", ssp]
        if final.get("spm_infill") == "y":
            cmd.append("--spm-infill")

        # Log options
        lf = final.get("log_file", "").strip()
        if lf:
            cmd += ["--log-file", lf]
        lc = final.get("log_colors", "auto")
        if lc != "auto":
            cmd += ["--log-colors", lc]
        if final.get("log_prefix") == "n":
            cmd.append("--no-log-prefix")
        if final.get("log_timestamps") == "n":
            cmd.append("--no-log-timestamps")

        # CPU MoE offload
        # These flags are not additive. --cpu-moe means *all* expert tensors;
        # emitting it together with --n-cpu-moe N silently defeats the
        # measured partial split and leaves the intended VRAM unused.
        if n_cpu_moe > 0:
            cmd += ["--n-cpu-moe", str(n_cpu_moe)]
        elif final.get("cpu_moe") == "y":
            cmd.append("--cpu-moe")
        n_cpu_ffn = int(final.get("n_cpu_ffn", 0))
        if n_cpu_ffn > 0:
            cmd += ["--n-cpu-ffn", str(n_cpu_ffn)]

        # Reasoning budget message
        rbm = final.get("reasoning_budget_message", "").strip()
        if rbm:
            cmd += ["--reasoning-budget-message", rbm]

        # Reasoning / Chain-of-Thought
        reas = final.get("reasoning", "auto")
        if reas.lower() == "on":
            cmd += ["--reasoning", "on"]
        elif reas.lower() == "off":
            cmd += ["--reasoning", "off"]
        elif reas.lower() == "auto":
            cmd += ["--reasoning", "auto"]
        rf = final.get("reasoning_format", "auto")
        if rf and rf.lower() != "auto":
            cmd += ["--reasoning-format", rf]
        if int(final.get("reasoning_budget", -1)) >= 0:
            cmd += ["--reasoning-budget", str(final["reasoning_budget"])]

        # Chat template kwargs (JSON)
        # Caminho inequívoco para reasoning_effort low/medium/high do GPT-OSS;
        # o campo OAI reasoning_effort só trata "none" diretamente.
        ctk = str(final.get("chat_template_kwargs", "") or "").strip()
        if ctk:
            try:
                json.loads(ctk)
            except (ValueError, TypeError) as exc:
                raise ValueError("chat_template_kwargs: informe um JSON valido") from exc
            cmd += ["--chat-template-kwargs", ctk]
        rp = final.get("reasoning_preserve", "auto")
        if rp == "y":
            cmd.append("--reasoning-preserve")
        elif rp == "n":
            cmd.append("--no-reasoning-preserve")

        # MCP servers (stdio support)
        mcp_native_json = final.get("mcp_native_json", "").strip()
        mcp_file = final.get("mcp_config_file", "").strip()
        mcp_json = final.get("mcp_config_json", "").strip()
        if mcp_native_json:
            cmd += ["--mcp-servers-json", mcp_native_json]
        if mcp_file:
            cmd += ["--mcp-servers-config", mcp_file]
        if mcp_json:
            cmd += ["--mcp-servers-json", mcp_json]

        # Multimodal
        if final.get("omni") == "y" and self.meta.mmproj_file and self.meta.mmproj_valid:
            if final.get("no_mmproj_auto") == "y":
                cmd.append("--no-mmproj-auto")
            cmd += ["--mmproj", self.meta.mmproj_file]
            force_cpu_mmproj = (
                final.get("mmproj_offload") == "n"
                or (
                    self.meta.arch.lower() in {"qwen35", "qwen35moe", "gemma4"}
                    and self.hw.gpu_vram_mb < 16384
                )
            )
            if force_cpu_mmproj:
                cmd.append("--no-mmproj-offload")
                # ``--no-mmproj-offload`` define use_gpu=false. Reforce o
                # destino também no seletor novo do mtmd para que variáveis de
                # ambiente ou um device geral do LLM nunca façam o projetor
                # voltar à CUDA. O Qwen 35B usa esta via para manter o MMProj
                # na RAM e reservar a RTX 3060 para os pesos/estado do LLM.
                if _server_supports_flag(self.llama_server, "--mmproj-device"):
                    cmd += ["--mmproj-device", "none"]
            if int(final.get("image_min_tokens", 0)) > 0:
                cmd += ["--image-min-tokens", str(final["image_min_tokens"])]
            if int(final.get("image_max_tokens", 0)) > 0:
                cmd += ["--image-max-tokens", str(final["image_max_tokens"])]
            mtmd = int(final.get("mtmd_batch_max", 1024))
            if mtmd > 0:
                cmd += ["--mtmd-batch-max-tokens", str(mtmd)]
        elif final.get("omni") != "y":
            # The launcher has an explicit vision toggle. Prevent llama-server
            # from discovering a neighbouring projector behind that toggle.
            cmd.append("--no-mmproj-auto")

        # Audio (vocoder)
        if final.get("audio") == "y" and self.meta.vocoder_file:
            if not _server_supports_flag(self.llama_server, "--model-vocoder"):
                raise ValueError(
                    "Este llama-server nao oferece --model-vocoder; audio foi recusado "
                    "antes da inicializacao para evitar um comando invalido."
                )
            cmd += ["--model-vocoder", self.meta.vocoder_file]

        # Jinja template engine
        if final.get("jinja") == "n":
            cmd.append("--no-jinja")

        # Sleep idle
        si = int(final.get("sleep_idle", -1))
        if si > 0:
            cmd += ["--sleep-idle-seconds", str(si)]

        # Slot prompt similarity
        ss = final.get("slot_similarity", "")
        if ss:
            cmd += ["--slot-prompt-similarity", str(ss)]

        # Media path
        media = final.get("media_path", MEDIA_PATH)
        if media and os.path.isdir(media):
            cmd += ["--media-path", media]

        # Prompt cache and server operation
        cmd += ["--cache-ram", str(final.get("cache_ram", 2048))]
        cmd += ["--ctx-checkpoints", str(final.get("ctx_checkpoints", 32))]
        cmd += ["--checkpoint-min-step", str(final.get("checkpoint_min_step", 8192))]
        if final.get("context_shift") == "y":
            cmd.append("--context-shift")
        if final.get("warmup") == "n":
            cmd.append("--no-warmup")
        cmd += ["--timeout", str(final.get("timeout", 3600))]
        cmd += ["--log-verbosity", str(final.get("log_verbosity", 3))]
        if final.get("metrics") == "y":
            cmd.append("--metrics")

        # Agent / Tools
        if final.get("agentic") == "y":
            cmd.append("--agent")
        else:
            tools = final.get("tools", "all")
            if tools == "readonly":
                cmd += ["--tools", "read_file,file_glob_search,grep_search,get_info"]
            elif tools == "all":
                cmd += ["--tools", "all"]

        # Agentic max turns (via UI config)
        amt = int(final.get("agentic_max_turns", 10))
        if amt >= 0:
            ui_config = json.dumps({
                "agenticMaxTurns": amt,
                "agenticMaxToolPreviewLines": int(final.get("agentic_max_tool_preview_lines", 25)),
            })
            cmd += ["--ui-config", ui_config]
        uicf = final.get("ui_config_file", "").strip()
        if uicf:
            cmd += ["--ui-config-file", uicf]

        # Speculative decoding (MTP / n-gram / draft models)
        spec_type = final.get("spec_type", "none")
        if spec_type not in SPECULATIVE_TYPES:
            raise ValueError(
                f"spec_type invalido: {spec_type}; use um de "
                + ", ".join(SPECULATIVE_TYPES)
            )
        if spec_type in ("ngram-mod", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-cache"):
            cmd += ["--spec-type", spec_type]
            nmin = int(final.get("spec_ngram_mod_n_min", 48))
            nmax = int(final.get("spec_ngram_mod_n_max", 64))
            nmatch = int(final.get("spec_ngram_mod_n_match", 24))
            nhits = int(final.get("spec_ngram_min_hits", 1))
            if spec_type == "ngram-mod":
                if nmax != 64: cmd += ["--spec-ngram-mod-n-max", str(nmax)]
                if nmin != 48: cmd += ["--spec-ngram-mod-n-min", str(nmin)]
                if nmatch != 24: cmd += ["--spec-ngram-mod-n-match", str(nmatch)]
            elif spec_type == "ngram-simple":
                cmd += ["--spec-ngram-simple-size-n", str(nmatch)]
                cmd += ["--spec-ngram-simple-size-m", str(nmax)]
                if nhits > 1: cmd += ["--spec-ngram-simple-min-hits", str(nhits)]
            elif spec_type == "ngram-map-k":
                cmd += ["--spec-ngram-map-k-size-n", str(nmatch)]
                cmd += ["--spec-ngram-map-k-size-m", str(nmax)]
                if nhits > 1: cmd += ["--spec-ngram-map-k-min-hits", str(nhits)]
            elif spec_type == "ngram-map-k4v":
                cmd += ["--spec-ngram-map-k4v-size-n", str(nmatch)]
                cmd += ["--spec-ngram-map-k4v-size-m", str(nmax)]
                if nhits > 1: cmd += ["--spec-ngram-map-k4v-min-hits", str(nhits)]
        elif spec_type and spec_type != "none":
            cmd += ["--spec-type", spec_type]
            if spec_type.startswith("draft-"):
                cmd += ["--spec-draft-type-k", final["cache_k"]]
                cmd += ["--spec-draft-type-v", final["cache_v"]]
            sn = int(final.get("spec_draft_n_max", 0))
            if sn > 0:
                cmd += ["--spec-draft-n-max", str(sn)]
            snmin = int(final.get("spec_draft_n_min", 0))
            if snmin > 0:
                cmd += ["--spec-draft-n-min", str(snmin)]
            sp = float(final.get("spec_draft_p_min", 0.0))
            if sp > 0.0:
                cmd += ["--spec-draft-p-min", str(sp)]
            sps = float(final.get("spec_draft_p_split", 0.10))
            if sps != 0.10:
                cmd += ["--spec-draft-p-split", str(sps)]

        # Fit (auto-param adjustment)
        cmd += ["--fit", "on" if fit_on else "off"]
        if fit_on:
            cmd += ["--fit-target", str(final.get("fit_target", 1024))]
            cmd += ["--fit-ctx", str(final.get("fit_ctx", 4096))]

        return [str(arg) for arg in cmd]


# ════════════════════════════════════════════════════════════
#   HUGGINGFACE HUB CLIENT  (reconstruído do LM Studio)
# ════════════════════════════════════════════════════════════
HF_API   = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co"

HF_TRUSTED = {
    "lmstudio-community": 3, "bartowski": 2, "TheBloke": 1,
    "mradermacher": 1, "MaziyarPanahi": 1,
}


def _hf_headers() -> dict:
    headers = {"User-Agent": "crono-hf/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _hf_fetch_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _hf_parse_url(input_str: str):
    try:
        u = urllib.parse.urlparse(input_str)
        if u.hostname != "huggingface.co":
            return None
        segs = [s for s in u.path.split("/") if s]
        if len(segs) < 2:
            return None
        file_name = None
        revision = "main"
        if len(segs) > 4 and segs[2] in {"blob", "resolve"}:
            revision = urllib.parse.unquote(segs[3])
            file_name = "/".join(segs[4:])
        elif len(segs) > 2:
            file_name = "/".join(segs[2:])
        return {"user": segs[0], "repo": segs[1], "file": file_name, "revision": revision}
    except Exception:
        return None


def _hf_clean_filename(name: str):
    return re.sub(r"-I?Q\d[_0-9A-Za-z]{0,6}", "", name)


def _hf_find_breakpoints(name: str):
    pts = []
    in_b = False
    for i, ch in enumerate(name):
        if ch in "-.":
            if not in_b:
                pts.append(i)
                in_b = True
        else:
            in_b = False
    return pts


def _hf_format_bytes(b: int):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def _hf_display_name(name: str) -> str:
    """Mostra o nome do arquivo GGUF sem a extensão .gguf (preserva internamente)."""
    return name[:-5] if name.lower().endswith(".gguf") else name


def _hf_base_model_name(name: str) -> str:
    """Remove apenas sufixos de formato, sem alterar a identidade do modelo."""
    value = name
    suffix = re.compile(
        r"(?i)(?:[._-](?:"
        r"gguf|ggml|fp8(?:[._-]dynamic)?|bf16|fp16|f16|f32|"
        r"awq|gptq|exl2|hqq|aqlm|mlx|int8|int4|"
        r"bnb(?:[._-](?:4|8)bit)?|bitsandbytes|(?:4|8)bit|quantized"
        r"))$"
    )
    while True:
        cleaned = suffix.sub("", value)
        if cleaned == value:
            return value
        value = cleaned


def _hf_format_params(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value:,}"


class HuggingFaceHub:
    """Busca, consulta e download de modelos do HuggingFace (estilo LM Studio)."""

    def __init__(self):
        self._response_lock = threading.Lock()
        self._active_responses = set()

    def cancel_downloads(self) -> None:
        with self._response_lock:
            responses = list(self._active_responses)
        for response in responses:
            try:
                response.close()
            except Exception:
                pass

    def search(self, term: str, limit: int = 20) -> list:
        data = _hf_fetch_json(
            f"{HF_API}/models?search={urllib.parse.quote(term)}&full=true&sort=likes"
        )
        return data[:limit]

    def latest_models(
        self, *, search: str = "", pipeline_tag: str = "",
        filter_tag: str = "", limit: int = 20,
    ) -> list:
        """Lista lançamentos/atualizações mais recentes pela API pública.

        A busca normal continua ordenada por popularidade. Este caminho é
        deliberadamente separado para o radar: ``lastModified`` identifica um
        lançamento ou atualização recente e ``full=true`` traz tags suficientes
        para classificá-lo sem baixar o card/modelo.
        """
        try:
            bounded_limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            bounded_limit = 20
        query = {
            "sort": "lastModified", "direction": "-1", "full": "true",
            "limit": str(bounded_limit),
        }
        if search.strip():
            query["search"] = search.strip()
        if pipeline_tag.strip():
            query["pipeline_tag"] = pipeline_tag.strip()
        if filter_tag.strip():
            query["filter"] = filter_tag.strip()
        return _hf_fetch_json(f"{HF_API}/models?{urllib.parse.urlencode(query)}")

    def model_info(self, user: str, repo: str, revision: str = "main") -> dict:
        if revision and revision != "main":
            quoted = urllib.parse.quote(revision, safe="")
            return _hf_fetch_json(
                f"{HF_API}/models/{user}/{repo}/revision/{quoted}?blobs=true"
            )
        return _hf_fetch_json(f"{HF_API}/models/{user}/{repo}?blobs=true")

    def gguf_files(self, user: str, repo: str) -> list:
        info = self.model_info(user, repo)
        return [s for s in info.get("siblings", []) if s["rfilename"].endswith(".gguf")]

    def resolve_candidates(self, filename: str) -> list:
        clean = _hf_clean_filename(filename)
        pts = _hf_find_breakpoints(clean)
        pts.append(len(clean))
        for i in range(len(pts) - 1, -1, -1):
            term = clean[:pts[i]]
            if len(term) < 3:
                continue
            repos = _hf_fetch_json(
                f"{HF_API}/models?search={urllib.parse.quote(term)}&full=true&sort=likes"
            )
            candidates = []
            for r in repos:
                if any(
                    s["rfilename"].lower() == filename.lower()
                    for s in r.get("siblings", [])
                ):
                    sp = r["id"].split("/")
                    if len(sp) == 2:
                        candidates.append(sp)
            if candidates:
                candidates.sort(
                    key=lambda x: HF_TRUSTED.get(x[0], 0), reverse=True
                )
                return candidates
        return []

    def download(
        self, user: str, repo: str, filename: str, dest_dir: str,
        on_progress=None, expected_size: int = 0, expected_sha256: str = "",
        cancel_event=None, revision: str = "main", promote: bool = True,
    ) -> str:
        clean_name = filename.replace("\\", "/").lstrip("/")
        quoted_revision = urllib.parse.quote(revision or "main", safe="")
        url = f"{HF_RESOLVE}/{user}/{repo}/resolve/{quoted_revision}/{urllib.parse.quote(clean_name, safe='/')}"
        req = urllib.request.Request(url, headers=_hf_headers())
        os.makedirs(dest_dir, exist_ok=True)
        base = Path(dest_dir).resolve()
        dest_path = (base / clean_name).resolve()
        if base not in dest_path.parents:
            raise ValueError("nome de arquivo remoto inválido")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        partial = dest_path.with_name(dest_path.name + ".part")
        src = None
        try:
            src = urllib.request.urlopen(req, timeout=30)
            with self._response_lock:
                self._active_responses.add(src)
            with src:
                total = int(src.headers.get("Content-Length", 0))
                downloaded = 0
                digest = hashlib.sha256()
                start = _time.monotonic()
                chunk_size = 1024 * 1024
                with partial.open("wb") as f:
                    while True:
                        if cancel_event and cancel_event.is_set():
                            raise InterruptedError("download cancelado")
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if on_progress:
                            elapsed = max(_time.monotonic() - start, 0.001)
                            on_progress(downloaded, total, downloaded / elapsed)
            if total and downloaded != total:
                raise IOError(f"download incompleto: {downloaded} de {total} bytes")
            if expected_size and downloaded != expected_size:
                raise IOError(f"tamanho divergente: {downloaded} de {expected_size} bytes")
            if expected_sha256 and digest.hexdigest().lower() != expected_sha256.lower():
                raise IOError("SHA-256 do arquivo baixado não confere")
            if promote:
                os.replace(partial, dest_path)
        except Exception:
            try:
                partial.unlink()
            except OSError:
                pass
            raise
        finally:
            if src is not None:
                with self._response_lock:
                    self._active_responses.discard(src)
        return str(dest_path if promote else partial)

    def search_by_url_or_id(self, input_str: str):
        parsed = _hf_parse_url(input_str)
        if parsed:
            return parsed
        if "/" in input_str:
            parts = input_str.split("/")
            return {
                "user": parts[0], "repo": parts[1],
                "file": "/".join(parts[2:]) if len(parts) > 2 else None,
                "revision": "main",
            }
        return None
