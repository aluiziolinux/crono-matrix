import json
import hashlib
import os
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

from jinja2 import Environment, FileSystemLoader

from autotune_cache import AutotuneCache
from launch_model_core import (
    HardwareInfo, ModelMetadata, OptimalParams, _find_companion,
    _is_auxiliary_gguf,
)
from web.services import EvalRunner, LauncherWebState, _resolve_node_runtime


class _LiveProcess:
    pid = 4242

    def poll(self):
        return None


class LauncherRuntimeTests(unittest.TestCase):
    def test_hardware_swap_detects_nvme_priority_above_zram(self):
        hardware = HardwareInfo()
        table = (
            "Filename Type Size Used Priority\n"
            "/dev/zram0 partition 32688124 0 100\n"
            f"{hardware.swap_nvme_path} file 16777212 0 110\n"
        )
        with mock.patch(
            "launch_model_core.Path.read_text", return_value=table
        ), mock.patch.object(
            hardware, "_run", return_value="/dev/nvme0n1p2"
        ):
            hardware._swap()

        self.assertEqual(hardware.swap_zram_priority, 100)
        self.assertEqual(hardware.swap_nvme_priority, 110)
        self.assertTrue(hardware.swap_nvme_preferred)

    def test_dynamic_swap_reapplies_existing_file_with_unsafe_priority(self):
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.auto_nvme_swap_default = "y"
        state.opt = mock.Mock(swap_recommended_gib=12)
        state.hardware = HardwareInfo()
        state.hardware.swap_nvme_total_mb = 16 * 1024
        state.hardware.swap_nvme_active = True
        state.hardware.swap_nvme_preferred = False

        expected = {"reapplied": True}
        with mock.patch.object(
            state, "configure_nvme_swap", return_value=expected
        ) as configure:
            result = state.ensure_dynamic_swap()

        self.assertEqual(result, expected)
        configure.assert_called_once_with("create", 16)

    def test_resident_launch_rejects_zram_first_route(self):
        hardware, metadata = self._qwen35_fixture()
        hardware.swap_nvme_priority = 10
        hardware.swap_nvme_preferred = False
        optimal = OptimalParams(hardware, metadata)
        optimal.n_cpu_moe = 33
        optimal.ctx = 262144
        optimal._plan_host_memory()
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.hardware = hardware
        state.meta = metadata
        state.opt = optimal

        with self.assertRaisesRegex(ValueError, "prioridade 10"):
            state._validate_resident_memory_route({
                "load_mode": "none", "n_cpu_moe": 33,
                "ctx": 262144, "cache_k": "bf16", "cache_v": "q8_0",
                "kv_offload": "y", "cache_ram": 2048,
                "ctx_checkpoints": 32, "omni": "n",
            })

    def test_mlock_is_rejected_when_model_cannot_fit_with_reserve(self):
        hardware, metadata = self._qwen35_fixture()
        hardware.ram_avail_mb = 20 * 1024
        optimal = OptimalParams(hardware, metadata)
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.hardware = hardware
        state.meta = metadata
        state.opt = optimal

        with self.assertRaisesRegex(ValueError, "sem possibilidade de paginação"):
            state._validate_resident_memory_route({
                "load_mode": "mmap+mlock", "omni": "n",
            })

    def test_autotune_resolution_never_hashes_model_during_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "large.gguf"
            model_path.write_bytes(b"metadata-only-fixture")
            stat = model_path.stat()
            cache_path = root / "autotune.json"
            cache_path.write_text(json.dumps({
                "version": 1,
                "records": [{
                    "record_id": "validated-local-record",
                    "status": "validated",
                    "apply_to_launch": True,
                    "model": {
                        "path": str(model_path.resolve()),
                        "name": model_path.name,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": "deadbeef",
                        "architecture": "qwen35moe",
                    },
                    "hardware": {"gpu_model": "RTX"},
                    "runtime": {"build": "local"},
                    "workload": {"ctx": 262144},
                    "sampler": {"seed": 42},
                    "config": {"ctx": 262144},
                    "metrics": {"score": 1.0},
                    "quality": {"validated": True, "stable": True},
                }],
            }), encoding="utf-8")
            cache = AutotuneCache(cache_path)
            query = {
                "path": str(model_path.resolve()),
                "name": model_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": "",
                "architecture": "qwen35moe",
            }
            with mock.patch(
                "autotune_cache.sha256_file",
                side_effect=AssertionError("selection must not hash GGUF"),
            ):
                hit = cache.resolve(
                    query,
                    {"gpu_model": "RTX"},
                    {"build": "local"},
                    {"ctx": 262144},
                    {"seed": 42},
                )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["record_id"], "validated-local-record")

    def test_process_snapshot_reports_external_listener_without_claiming_pid(self):
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.proc = None
        state.process_state = "idle"
        state.process_error = ""
        state.params = {"host": "127.0.0.1", "port": 0, "mcp_native": "n"}
        state.model_path = ""
        state.started_at = None
        state.exit_code = None
        state.mcp_state = "disabled"
        state.mcp_tools = 0
        state.mcp_error = ""
        state.runtime_effective = {"ready": False}
        state.memory_guard = {}
        state.agent_global_default = "n"
        state.agent_info = {"global_enabled": False}

        # The managed test environment forbids opening listening sockets.
        # Mock only the OS probe so this test exercises the state transition
        # and the safety invariant (no foreign PID is claimed).
        with mock.patch.object(
            LauncherWebState, "_port_is_listening", return_value=True
        ):
            snapshot = state.process_snapshot()

        self.assertEqual(snapshot["state"], "external")
        self.assertTrue(snapshot["external_conflict"])
        self.assertFalse(snapshot["running"])
        self.assertIsNone(snapshot["pid"])
        self.assertIn("processo externo", snapshot["error"])

    def test_runtime_buffer_parser_confirms_cuda_kv_and_recurrent_state(self):
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.runtime_effective = {
            "memory_buffers": {
                "requested": "gpu",
                "kv": {"devices": {}, "total_mb": 0.0, "gpu_mb": 0.0,
                       "cpu_mb": 0.0, "placement": "pending", "confirmed": None},
                "rs": {"devices": {}, "total_mb": 0.0, "gpu_mb": 0.0,
                       "cpu_mb": 0.0, "placement": "pending", "confirmed": None},
            }
        }

        state._capture_runtime_buffer(
            "llama_kv_cache:      CUDA0 KV buffer size =  3982.00 MiB\n"
        )
        state._capture_runtime_buffer(
            "llama_memory_recurrent: CUDA0 RS buffer size = 63.00 MiB\n"
        )

        memory = state.runtime_effective["memory_buffers"]
        self.assertEqual(memory["kv"]["placement"], "gpu")
        self.assertEqual(memory["kv"]["gpu_mb"], 3982.0)
        self.assertTrue(memory["kv"]["confirmed"])
        self.assertEqual(memory["rs"]["placement"], "gpu")
        self.assertEqual(memory["rs"]["gpu_mb"], 63.0)

    def test_runtime_buffer_parser_detects_cpu_kv_mismatch(self):
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.runtime_effective = {
            "memory_buffers": {
                "requested": "gpu",
                "kv": {"devices": {}, "total_mb": 0.0, "gpu_mb": 0.0,
                       "cpu_mb": 0.0, "placement": "pending", "confirmed": None},
                "rs": {"devices": {}, "total_mb": 0.0, "gpu_mb": 0.0,
                       "cpu_mb": 0.0, "placement": "pending", "confirmed": None},
            }
        }

        state._capture_runtime_buffer(
            "llama_kv_cache: CPU KV buffer size = 3982.00 MiB\n"
        )

        kv = state.runtime_effective["memory_buffers"]["kv"]
        self.assertEqual(kv["placement"], "cpu")
        self.assertFalse(kv["confirmed"])

    @staticmethod
    def _origin_check_state(hf, model_path):
        state = LauncherWebState.__new__(LauncherWebState)
        state.model_update_lock = threading.RLock()
        state.model_update_results = {}
        state.model_update_thread = None
        state.model_update = {
            "state": "idle", "started_at": "", "checked_at": "",
            "total": 0, "completed": 0, "current": "", "error": "",
        }
        state.hf = hf
        state.models = [{"path": str(model_path)}]
        state.models_dir = str(Path(model_path).parent)
        state.model_path = ""
        state.scan_models = mock.Mock()
        return state

    def test_local_gguf_update_check_confirms_hash_and_persists_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "4BeastsOfApocalypse.Q6_K.gguf"
            content = b"synthetic-gguf-for-origin-check"
            model.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            hf = mock.Mock()
            hf.resolve_candidates.return_value = [["mradermacher", "4BeastsOfApocalypse-GGUF"]]
            hf.model_info.return_value = {
                "sha": "commit-1", "lastModified": "2026-09-01T10:00:00Z",
                "siblings": [{
                    "rfilename": model.name, "size": len(content),
                    "lfs": {"sha256": digest},
                }],
            }
            state = self._origin_check_state(hf, model)

            result = state._verify_one_model_update(str(model))

            self.assertEqual(result["state"], "current")
            self.assertEqual(result["repo_id"], "mradermacher/4BeastsOfApocalypse-GGUF")
            origin = json.loads(
                Path(f"{model}.crono-origin.json").read_text(encoding="utf-8")
            )
            self.assertEqual(origin["remote_sha256"], digest)
            self.assertEqual(origin["downloaded_sha256"], digest)
            self.assertEqual(origin["commit"], "commit-1")

    def test_local_gguf_update_check_distinguishes_remote_update_from_local_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model-Q6_K.gguf"
            local = b"old-model-content"
            remote = b"new-model-content"
            model.write_bytes(local)
            old_digest = hashlib.sha256(local).hexdigest()
            new_digest = hashlib.sha256(remote).hexdigest()
            LauncherWebState._write_model_origin(model, {
                "repo_id": "example/model-GGUF", "filename": model.name,
                "downloaded_size": len(local), "downloaded_sha256": old_digest,
                "remote_size": len(local), "remote_sha256": old_digest,
            })
            hf = mock.Mock()
            hf.model_info.return_value = {
                "sha": "commit-2", "siblings": [{
                    "rfilename": model.name, "size": len(local),
                    "lfs": {"sha256": new_digest},
                }],
            }
            state = self._origin_check_state(hf, model)
            self.assertEqual(state._verify_one_model_update(str(model))["state"], "outdated")

            model.write_bytes(b"tampered-content!")
            # Same size, but neither the recorded download nor the remote
            # artifact: this is local divergence, not a remote update.
            result = state._verify_one_model_update(str(model))
            self.assertEqual(result["state"], "different")

    def test_local_gguf_update_check_marks_ambiguous_and_unassociated_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "unknown.Q6_K.gguf"
            model.write_bytes(b"gguf")
            hf = mock.Mock()
            state = self._origin_check_state(hf, model)
            hf.resolve_candidates.return_value = []
            self.assertEqual(state._verify_one_model_update(str(model))["state"], "unassociated")
            hf.resolve_candidates.return_value = [["one", "repo"], ["two", "repo"]]
            self.assertEqual(state._verify_one_model_update(str(model))["state"], "ambiguous")

    def test_llama_scope_is_isolated_without_memory_ceiling(self):
        state = LauncherWebState.__new__(LauncherWebState)
        with mock.patch("web.services.shutil.which", return_value="/usr/bin/systemd-run"), mock.patch(
            "web.services.Path.is_file", return_value=True
        ):
            command, unit, memory_high, headroom = state._scoped_llama_command(
                ["llama-server", "--ctx-size", "262144"], 26000, "none"
            )
        self.assertTrue(unit.startswith("crono-llama-"))
        self.assertEqual(headroom, 0)
        self.assertEqual(memory_high, 0)
        self.assertFalse(any("MemoryHigh=" in item for item in command))
        self.assertEqual(command[-2:], ["--ctx-size", "262144"])

    def test_mmap_load_also_avoids_memory_high_churn(self):
        state = LauncherWebState.__new__(LauncherWebState)
        with mock.patch("web.services.shutil.which", return_value="/usr/bin/systemd-run"), mock.patch(
            "web.services.Path.is_file", return_value=True
        ):
            _command, _unit, memory_high, headroom = state._scoped_llama_command(
                ["llama-server"], 26000, "mmap"
            )
        self.assertEqual(headroom, 0)
        self.assertEqual(memory_high, 0)
        self.assertFalse(any("MemoryHigh=" in item for item in _command))

    def test_post_load_scope_transition_updates_kernel_and_telemetry(self):
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.logs = deque(maxlen=20)
        state.log_seq = 0
        state.llama_scope_unit = "crono-llama-test"
        state.memory_guard = {
            "scope_phase": "loading",
            "scope_headroom_mb": 0,
            "memory_high_mb": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            cgroup_dir = Path(directory)
            (cgroup_dir / "memory.current").write_text(
                str(22000 * 1048576), encoding="utf-8"
            )
            (cgroup_dir / "memory.high").write_text(
                str(22000 * 1048576), encoding="utf-8"
            )
            with mock.patch.object(
                state, "_process_memory_control_dir", return_value=cgroup_dir
            ), mock.patch.object(state, "_mem_available_mb", return_value=2000):
                state._tune_scope_for_inference(_LiveProcess())
            self.assertEqual(state.memory_guard["scope_phase"], "inference")
            self.assertEqual(state.memory_guard["scope_headroom_mb"], 0)
            self.assertEqual(state.memory_guard["memory_high_mb"], 0)
            self.assertEqual(
                (cgroup_dir / "memory.high").read_text(encoding="utf-8"),
                "max\n",
            )

    def test_native_c99_ram_guard_compiles(self):
        state = LauncherWebState.__new__(LauncherWebState)
        binary = state._ensure_native_memory_guard()
        self.assertTrue(binary.is_file())
        self.assertTrue(os.access(binary, os.X_OK))

    def test_importance_matrix_is_not_exposed_as_a_model(self):
        self.assertTrue(_is_auxiliary_gguf("Tiel-Coder-35B-A3B.imatrix.gguf"))
        self.assertTrue(_is_auxiliary_gguf("model-importance-matrix.gguf"))
        self.assertFalse(_is_auxiliary_gguf("Tiel-Coder-35B-A3B-Q6_K_XL.gguf"))

    @staticmethod
    def _qwen35_fixture():
        hardware = HardwareInfo()
        hardware.cpu_cores = 16
        hardware.cpu_threads = 32
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 27673
        hardware.swap_zram_total_mb = 32 * 1024
        hardware.swap_zram_priority = 100
        hardware.swap_nvme_total_mb = 16 * 1024
        hardware.swap_nvme_priority = 110
        hardware.swap_nvme_active = True
        hardware.swap_nvme_preferred = True
        hardware.disk_free_gb = "65"
        hardware.gpu_detected = True
        hardware.gpu_model = "NVIDIA GeForce RTX 3060"
        hardware.gpu_vram_mb = 12288
        hardware.gpu_vram_free_mb = 11468
        hardware.gpu_vram_free_gb = "11.2"

        metadata = ModelMetadata()
        metadata.arch = "QWEN35MOE"
        metadata.ctx_max = 262144
        metadata.layers = 40
        metadata.moe_layers = 40
        metadata.size_mb = 27193
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.expert_weight_bytes_by_layer = [630 * 1048576] * 40
        metadata.path = "/models/Qwen3.6-35B-A3B-Q6_K.gguf"
        metadata.general_basename = "Qwen_Qwen3.6"
        metadata.sampling_temp = 1.0
        metadata.sampling_top_k = 20
        metadata.sampling_top_p = 0.95
        metadata.sampling_min_p = None
        return hardware, metadata

    @staticmethod
    def _laguna_fixture():
        hardware, _ = LauncherRuntimeTests._qwen35_fixture()
        metadata = ModelMetadata()
        metadata.meta_ok = True
        metadata.arch = "LAGUNA"
        metadata.general_basename = "Laguna XS 2.1"
        metadata.ctx_max = 262144
        metadata.layers = 40
        metadata.moe_layers = 40
        metadata.expert_count = 256
        metadata.expert_used_count = 8
        metadata.sliding_window = 512
        metadata.size_mb = 26229
        metadata.path = "/models/Laguna-XS-2.1-Q6_K.gguf"
        metadata.sampling_temp = 1.0
        metadata.sampling_top_k = 20
        metadata.sampling_top_p = 1.0
        metadata.sampling_min_p = 0.0
        metadata.supports_reasoning_preserve = True
        return hardware, metadata

    @staticmethod
    def _gemma4_fixture():
        hardware, _ = LauncherRuntimeTests._qwen35_fixture()
        hardware.ram_avail_mb = 25288
        metadata = ModelMetadata()
        metadata.meta_ok = True
        metadata.arch = "GEMMA4"
        metadata.general_basename = "Gemma 4 26B A4B"
        metadata.ctx_max = 262144
        metadata.layers = 30
        metadata.kv_layers = 30
        metadata.attention_layers = 30
        metadata.moe_layers = 30
        metadata.dense_layers = 0
        metadata.expert_count = 128
        metadata.expert_used_count = 8
        metadata.expert_ff = 704
        metadata.embed = 2816
        metadata.heads = 16
        metadata.heads_kv = 8
        metadata.layer_heads_kv = ([8] * 5 + [2]) * 5
        metadata.key_len = 512
        metadata.val_len = 512
        metadata.key_len_swa = 256
        metadata.val_len_swa = 256
        metadata.sliding_window = 1024
        metadata.sliding_window_pattern = ([True] * 5 + [False]) * 5
        metadata.swa_layers = 25
        metadata.global_layers = 5
        metadata.layer_layout_valid = True
        metadata.size_mb = 25616
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.path = "/models/gemma-4-26B-A4B-it-Q8_0.gguf"
        # Exact Q8 expert tensor footprint observed in each local GGUF layer.
        metadata.expert_weight_bytes_by_layer = [808845824] * 30
        metadata.sampling_temp = 1.0
        metadata.sampling_top_k = 64
        metadata.sampling_top_p = 0.95
        metadata.sampling_min_p = None
        return hardware, metadata

    def test_resolve_source_root_uses_canonical_rtx3060_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "llama.cpp"
            binary_dir = source / "build-rtx3060" / "bin"
            binary_dir.mkdir(parents=True)
            for name in ("llama-server", "llama-fit-params"):
                binary = binary_dir / name
                binary.write_text("#!/bin/sh\n", encoding="utf-8")
                os.chmod(binary, 0o755)

            display, server, fit_params = LauncherWebState._resolve_llama_cpp(
                str(source), require=True,
            )
            self.assertEqual(display, str(source.resolve()))
            self.assertEqual(server, str((binary_dir / "llama-server").resolve()))
            self.assertEqual(fit_params, str((binary_dir / "llama-fit-params").resolve()))

    def test_resolve_source_root_prefers_portable_crono_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "llama.cpp"
            portable_dir = source / "build-crono" / "bin"
            legacy_dir = source / "build-rtx3060" / "bin"
            portable_dir.mkdir(parents=True)
            legacy_dir.mkdir(parents=True)
            for binary_dir in (portable_dir, legacy_dir):
                for name in ("llama-server", "llama-fit-params"):
                    binary = binary_dir / name
                    binary.write_text("#!/bin/sh\n", encoding="utf-8")
                    os.chmod(binary, 0o755)

            _display, server, fit_params = LauncherWebState._resolve_llama_cpp(
                str(source), require=True,
            )

            self.assertEqual(server, str((portable_dir / "llama-server").resolve()))
            self.assertEqual(
                fit_params, str((portable_dir / "llama-fit-params").resolve())
            )

    def test_shared_mmproj_requires_matching_model_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary) / "models"
            glm_dir = models / "GLM-4.7-Flash-Q6_K_L"
            qwen_dir = models / "Qwen3.6-35B-A3B-Q6_K"
            beasts_dir = models / "4BeastsOfApocalypse.Q6_K"
            glm_dir.mkdir(parents=True)
            qwen_dir.mkdir()
            beasts_dir.mkdir()
            glm_model = glm_dir / "zai-org_GLM-4.7-Flash-Q6_K_L.gguf"
            qwen_model = qwen_dir / "Qwen3.6-35B-A3B-Q6_K.gguf"
            beasts_model = beasts_dir / "4BeastsOfApocalypse.Q6_K.gguf"
            qwen_mmproj = models / "mmproj-Qwen3.6-35B-A3B-BF16.gguf"
            beasts_mmproj = beasts_dir / "mmproj-4BeastsOfApocalypse-BF16.gguf"
            for path in (glm_model, qwen_model, beasts_model, qwen_mmproj, beasts_mmproj):
                path.touch()

            self.assertEqual(_find_companion(str(glm_model), "mmproj"), "")
            self.assertEqual(
                Path(_find_companion(str(qwen_model), "mmproj")), qwen_mmproj,
            )
            # A numeric prefix must not prevent a valid same-family MMProj
            # from being associated with the model.
            self.assertEqual(
                Path(_find_companion(str(beasts_model), "mmproj")), beasts_mmproj,
            )

    def test_templates_compile_and_idle_action_response_renders(self):
        template_root = Path(__file__).resolve().parents[1] / "web" / "templates"
        environment = Environment(loader=FileSystemLoader(str(template_root)))
        environment.filters["bytes"] = str
        environment.filters["params"] = str
        for name in environment.list_templates(extensions=("html",)):
            environment.get_template(name)

        process = {
            "state": "idle", "pid": None, "model": "", "ready": False,
            "running": False, "error": "", "mcp_state": "disabled",
            "mcp_tools": 0, "mcp_error": "", "agent_global_default": False,
            "last_command": "",
            "runtime_effective": {
                "ready": False, "context_window": 0,
                "requested_context": 0, "total_slots": 0,
            },
            "agent_compat": {
                "global_enabled": False, "enabled": False, "model": "",
                "endpoint": "", "agent_env": "", "agent_metadata": "",
                "context_window": 0, "auto_compact_token_limit": 0,
                "modalities": [], "reasoning_enabled": False,
                "supports_reasoning_effort": False,
                "global_config": "", "global_error": "",
                "opencode_config": "",
            },
        }
        rendered = environment.get_template(
            "partials/action_response.html"
        ).render(process=process, message="", error="")
        self.assertIn("ATIVAR MODO UNIVERSAL", rendered)
        self.assertIn("Aguardando comando", rendered)
        eval_rendered = environment.get_template(
            "partials/eval_runner.html"
        ).render(eval_state=EvalRunner().snapshot())
        self.assertIn("CONFIGURAÇÃO REPRODUZÍVEL DE INFERÊNCIA", eval_rendered)
        self.assertIn('name="reasoning_effort"', eval_rendered)
        radar_rendered = environment.get_template(
            "partials/hf_radar.html"
        ).render(radar={
            "enabled": True, "watchlist": "qwen,gemma", "initialized": True,
            "last_refresh": "2026-09-01T00:00:00Z", "error": "",
            "refreshing": False, "unread_count": 0, "items": [],
        })
        self.assertIn("RADAR DE LANÇAMENTOS", radar_rendered)
        self.assertIn("PREFERÊNCIAS DO RADAR", radar_rendered)

    def test_alpha_eval_configuration_is_validated_and_reproducible(self):
        config = EvalRunner._normalize_run_config(
            repeats="3", seed="42", scale="medium", mode="auto",
            reasoning_effort="high", reasoning_budget="auto", sampling="fixed",
            temperature="0.6", top_k="20", top_p="0.95", min_p="0.05",
            repeat_penalty="1.0", max_tokens="65536", timeout="600",
            xctx_scale="1", os_filter="linux", judge_url="", judge_model="",
        )
        self.assertEqual(config["max_tokens"], 65536)
        self.assertEqual(config["reasoning_effort"], "high")
        self.assertEqual(config["sampling"], "fixed")
        self.assertEqual(config["timeout"], 600)
        with self.assertRaises(ValueError):
            EvalRunner._normalize_run_config(
                mode="think", reasoning_effort="medium",
            )

    def test_global_mode_repairs_stale_compatibility_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.json"
            settings.write_text(json.dumps({
                "agent_global": "y",
                "agent_compat": "n",
            }), encoding="utf-8")
            state = LauncherWebState(settings_file=settings)
            self.assertEqual(state.agent_global_default, "y")
            self.assertEqual(state.agent_compat_default, "y")
            state.set_agent_global(True)
            saved = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual(saved["agent_global"], "y")
            self.assertEqual(saved["agent_compat"], "y")

    def test_node_runtime_preflight_detects_dynamic_linker_failure(self):
        broken = mock.Mock(returncode=127, stdout="", stderr="libada.so.3: cannot open shared object file")
        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch("web.services.shutil.which", return_value="/usr/bin/node"), mock.patch(
                "web.services.subprocess.run", return_value=broken
            ):
                path, version, diagnostics = _resolve_node_runtime()
        self.assertEqual(path, "")
        self.assertEqual(version, "")
        self.assertIn("libada.so.3", diagnostics)

    def test_node_runtime_preflight_returns_absolute_working_binary(self):
        working = mock.Mock(returncode=0, stdout="v26.8.1\n", stderr="")
        with mock.patch.dict(os.environ, {"CRONO_NODE_BIN": "/opt/node/bin/node"}, clear=False):
            with mock.patch("web.services.shutil.which", return_value="/opt/node/bin/node"), mock.patch(
                "web.services.subprocess.run", return_value=working
            ) as run:
                path, version, diagnostics = _resolve_node_runtime()
        self.assertEqual(path, "/opt/node/bin/node")
        self.assertEqual(version, "v26.8.1")
        self.assertEqual(diagnostics, "")
        run.assert_called_once_with(
            ["/opt/node/bin/node", "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

    def test_hf_radar_baselines_then_alerts_only_new_release(self):
        base = {
            "id": "Qwen/Qwen3.8-Flash-Next",
            "lastModified": "2026-09-01T10:00:00.000Z",
            "pipeline_tag": "text-generation",
            "tags": ["transformers", "qwen3", "reasoning"],
            "downloads": 100, "likes": 10,
        }
        gguf = {
            "id": "mradermacher/Qwen3.8-Flash-Next-GGUF",
            "lastModified": "2026-09-01T10:01:00.000Z",
            "tags": ["gguf", "llama.cpp", "qwen3", "mixture-of-experts"],
            "downloads": 10, "likes": 2,
        }

        def first_feed(**kwargs):
            return [gguf] if kwargs.get("filter_tag") == "gguf" else [base]

        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.json"
            state = LauncherWebState(settings_file=settings)
            with mock.patch.object(state.hf, "latest_models", side_effect=first_feed):
                baseline = state.refresh_hf_radar(force=True)
            self.assertTrue(baseline["initialized"])
            self.assertEqual(baseline["unread_count"], 0)
            self.assertTrue(any(item["id"] == gguf["id"] for item in baseline["items"]))

            new_release = {
                "id": "google/gemma-5-27B-A4B-it",
                "lastModified": "2026-09-01T11:00:00.000Z",
                "pipeline_tag": "text-generation",
                "tags": ["transformers", "gemma", "mixture-of-experts", "agent"],
                "downloads": 0, "likes": 1,
            }

            def second_feed(**kwargs):
                if kwargs.get("filter_tag") == "gguf":
                    return [gguf]
                return [base, new_release]

            with mock.patch.object(state.hf, "latest_models", side_effect=second_feed):
                refreshed = state.refresh_hf_radar(force=True)
            self.assertEqual(refreshed["unread_count"], 1)
            alerted = next(item for item in refreshed["items"] if item["id"] == new_release["id"])
            self.assertTrue(alerted["unread"])
            self.assertEqual(alerted["event"], "NOVO")
            self.assertEqual(state.mark_hf_radar_read()["unread_count"], 0)
            saved = json.loads(settings.read_text(encoding="utf-8"))
            self.assertTrue(saved["hf_radar_initialized"])
            self.assertIn(new_release["id"], saved["hf_radar_seen"])

    def test_preview_redacts_api_and_mcp_secrets(self):
        display = LauncherWebState._display_command([
            "llama-server", "--api-key", "top-secret",
            "--mcp-servers-json", '{"token":"also-secret"}',
            "--ctx-size", "4096",
        ])
        self.assertNotIn("top-secret", display)
        self.assertNotIn("also-secret", display)
        self.assertIn("<redacted>", display)
        self.assertIn("--ctx-size 4096", display)

    def test_universal_profile_is_client_neutral_and_has_no_legacy_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".crono-agent"
            env_file = root / "agent-local.env.sh"
            metadata_file = root / "agent-local.json"
            state = LauncherWebState.__new__(LauncherWebState)
            state.meta = ModelMetadata()
            state.runtime_effective = {}
            final = {
                "agent_compat": "y", "host": "127.0.0.1", "port": 8080,
                "mcp_native": "n", "model_path": "/models/model.gguf",
                "alias": "local-model", "ctx": 32768, "reasoning": "auto",
                "omni": "n", "api_key": "", "api_key_file": "",
            }

            with mock.patch("web.services.AGENT_COMPAT_DIR", root), mock.patch(
                "web.services.AGENT_ENV_FILE", env_file
            ), mock.patch("web.services.AGENT_METADATA_FILE", metadata_file):
                info = state._write_agent_compatibility(final, 8080, 32768)

            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            self.assertEqual(info["endpoint"], "http://127.0.0.1:8080/v1")
            self.assertEqual(metadata["api"], "openai-compatible")
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["agent-local.env.sh", "agent-local.json", "model-catalog.json"],
            )

    def test_universal_profile_publishes_runtime_props_and_native_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".crono-agent"
            env_file = root / "agent-local.env.sh"
            metadata_file = root / "agent-local.json"
            state = LauncherWebState.__new__(LauncherWebState)
            state.meta = ModelMetadata()
            state.runtime_effective = {
                "props": {
                    "model_alias": "runtime-vision-model",
                    "model_ftype": "Q6_K",
                    "model_path": "/models/runtime-vision-model.gguf",
                    "build_info": "test-build",
                    "modalities": {"vision": True, "video": True, "audio": False},
                    "media_marker": "<__media__>",
                    "default_generation_settings": {
                        "n_ctx": 131072,
                        "params": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "reasoning_format": "deepseek",
                        },
                    },
                    "chat_template": "template",
                    "chat_template_tool_use": "tool-template",
                    "chat_template_caps": {
                        "supports_tools": True,
                        "supports_tool_calls": True,
                        "supports_parallel_tool_calls": True,
                        "supports_system_role": True,
                        "supports_preserve_reasoning": True,
                        "supports_reasoning_effort": False,
                        "supports_string_content": True,
                        "supports_typed_content": False,
                        "supports_object_arguments": True,
                    },
                    "endpoint_slots": True,
                    "endpoint_props": False,
                    "endpoint_metrics": True,
                    "cors_proxy_enabled": True,
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                }
            }
            state._fetch_runtime_payload = lambda url, key: [{
                "display_name": "Read file",
                "tool": "read_file",
                "type": "server",
                "permissions": {"write": False},
                "uses_cwd": True,
                "definition": {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {"type": "object"}},
                },
            }]
            final = {
                "agent_compat": "y", "host": "127.0.0.1", "port": 8080,
                "mcp_native": "n", "model_path": "/models/stale.gguf",
                "alias": "stale-alias", "ctx": 262144, "reasoning": "auto",
                "reasoning_budget": -1, "agentic": "y", "tools": "all",
                "omni": "y", "api_key": "", "api_key_file": "",
            }

            with mock.patch("web.services.AGENT_COMPAT_DIR", root), mock.patch(
                "web.services.AGENT_ENV_FILE", env_file
            ), mock.patch("web.services.AGENT_METADATA_FILE", metadata_file):
                info = state._write_agent_compatibility(final, 8080, 131072)

            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            catalog = json.loads(
                (root / "model-catalog.json").read_text(encoding="utf-8")
            )
            model = catalog["models"][0]

            self.assertEqual(info["model"], "runtime-vision-model")
            self.assertEqual(info["context_window"], 131072)
            self.assertEqual(info["modalities"], ["text", "image", "video"])
            self.assertTrue(info["capabilities"]["attachment"])
            self.assertTrue(info["capabilities"]["tool_call"])
            self.assertTrue(info["capabilities"]["input"]["video"])
            self.assertTrue(info["capabilities"]["interleaved"])
            self.assertEqual(info["server_capabilities"]["native_tool_count"], 1)
            self.assertEqual(info["native_tools"][0]["name"], "read_file")
            self.assertEqual(
                metadata["reasoning"]["supported_efforts"],
                ["off", "low", "medium", "high", "max"],
            )
            self.assertEqual(
                model["supported_reasoning_levels"][0]["effort"], "off"
            )
            self.assertEqual(
                model["supported_reasoning_levels"][-1]["effort"], "max"
            )
            self.assertEqual(metadata["runtime"]["model_ftype"], "Q6_K")
            self.assertEqual(metadata["endpoints"]["responses"], "http://127.0.0.1:8080/v1/responses")
            self.assertEqual(model["max_output_tokens"], 131072)
            self.assertEqual(model["truncation_policy"]["limit"], 131072)
            self.assertEqual(model["experimental_supported_tools"], ["read_file"])
            opencode_model = state._opencode_managed_provider()["models"][
                "runtime-vision-model"
            ]
            self.assertEqual(opencode_model["modalities"]["input"], [
                "text", "image", "video"
            ])
            self.assertEqual(opencode_model["limit"], {
                "context": 131072, "output": 131072,
            })
            self.assertTrue(opencode_model["temperature"])
            self.assertTrue(opencode_model["attachment"])
            self.assertTrue(opencode_model["tool_call"])
            self.assertEqual(
                opencode_model["interleaved"], {"field": "reasoning_content"}
            )
            self.assertEqual(
                opencode_model["variants"]["medium"]["reasoning_budget_tokens"],
                2048,
            )

    def test_universal_profile_detects_reasoning_when_runtime_format_is_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".crono-agent"
            state = LauncherWebState.__new__(LauncherWebState)
            state.meta = ModelMetadata()
            state.runtime_effective = {"props": {
                "model_alias": "plain-model",
                "default_generation_settings": {
                    "n_ctx": 8192,
                    "params": {"reasoning_format": "none"},
                },
                "chat_template": "<|im_start|>assistant\\n<think>\\n",
                "chat_template_caps": {
                    "supports_preserve_reasoning": True,
                    "supports_reasoning_effort": False,
                },
            }}
            state._fetch_runtime_payload = lambda url, key: []
            final = {
                "agent_compat": "y", "host": "127.0.0.1", "port": 8080,
                "mcp_native": "n", "model_path": "/models/plain.gguf",
                "alias": "plain-model", "ctx": 8192, "reasoning": "auto",
                "omni": "n", "api_key": "", "api_key_file": "",
            }
            with mock.patch("web.services.AGENT_COMPAT_DIR", root), mock.patch(
                "web.services.AGENT_ENV_FILE", root / "agent.env"
            ), mock.patch("web.services.AGENT_METADATA_FILE", root / "agent.json"):
                info = state._write_agent_compatibility(final, 8080, 8192)
            self.assertTrue(info["reasoning_enabled"])
            self.assertFalse(info["supports_reasoning_effort"])
            self.assertEqual(
                info["capabilities"]["interleaved"],
                {"field": "reasoning_content"},
            )
            self.assertEqual(
                info["server_capabilities"]["reasoning_detection"]["source"],
                "chat_template",
            )

    def test_universal_profile_does_not_invent_reasoning_for_plain_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".crono-agent"
            state = LauncherWebState.__new__(LauncherWebState)
            state.meta = ModelMetadata()
            state.runtime_effective = {"props": {
                "model_alias": "plain-model",
                "default_generation_settings": {
                    "n_ctx": 8192,
                    "params": {"reasoning_format": "none"},
                },
                "chat_template": "{{ messages[0].content }}",
                "chat_template_caps": {},
            }}
            state._fetch_runtime_payload = lambda url, key: []
            final = {
                "agent_compat": "y", "host": "127.0.0.1", "port": 8080,
                "mcp_native": "n", "model_path": "/models/plain.gguf",
                "alias": "plain-model", "ctx": 8192, "reasoning": "auto",
                "omni": "n", "api_key": "", "api_key_file": "",
            }
            with mock.patch("web.services.AGENT_COMPAT_DIR", root), mock.patch(
                "web.services.AGENT_ENV_FILE", root / "agent.env"
            ), mock.patch("web.services.AGENT_METADATA_FILE", root / "agent.json"):
                info = state._write_agent_compatibility(final, 8080, 8192)
            self.assertFalse(info["reasoning_enabled"])
            self.assertFalse(info["capabilities"]["reasoning"])
            self.assertEqual(info["capabilities"]["interleaved"], False)

    def test_readiness_uses_props_context_as_runtime_truth(self):
        state = LauncherWebState.__new__(LauncherWebState)
        state.lock = threading.RLock()
        state.proc = _LiveProcess()
        state.params = {"ctx": 262144}
        state.runtime_effective = {}
        state.process_state = "starting"
        state.process_error = ""
        state.log_seq = 0
        state.logs = deque(maxlen=20)
        state.agent_info = {"global_error": ""}
        state._agent_api_key = lambda final: "local"
        state._fetch_runtime_json = lambda url, key: (
            {"status": "ok"} if url.endswith("/health") else {
                "default_generation_settings": {"n_ctx": 109824},
                "total_slots": 1,
                "model_path": "/models/test.gguf",
            }
        )
        published = {}
        state._write_agent_compatibility = (
            lambda final, port, effective: published.update(context=effective, port=port)
        )

        final = {
            "host": "127.0.0.1", "port": 8080, "ctx": 262144,
            "parallel": 1, "model_path": "/models/test.gguf",
            "timeout": 60, "agent_global": "n",
        }
        state._wait_for_server_ready(state.proc, final, 8080)

        self.assertEqual(state.process_state, "running")
        self.assertEqual(state.params["effective_ctx"], 109824)
        self.assertEqual(state.runtime_effective["context_window"], 109824)
        self.assertEqual(published["context"], 109824)

    def test_opencode_adapter_declares_vision_and_restores_original_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "opencode.jsonc"
            state_file = root / "opencode-state.json"
            original = """{
  // configuração do usuário
  \"theme\": \"matrix\",
  \"provider\": {
    \"crono\": {\"name\": \"perfil antigo\"},
  },
}\n"""
            config.write_text(original, encoding="utf-8")
            launcher = LauncherWebState.__new__(LauncherWebState)
            launcher.agent_info = {
                "model": "Qwen3.6-35B-A3B-Q6_K",
                "endpoint": "http://127.0.0.1:8080/v1",
                "context_window": 262144,
                "modalities": ["text", "image"],
                "reasoning_enabled": True,
                "supports_reasoning_effort": True,
            }
            with mock.patch("web.services.OPENCODE_GLOBAL_CONFIG", config), mock.patch(
                "web.services.OPENCODE_GLOBAL_STATE", state_file
            ):
                launcher._activate_opencode_global()
                active = json.loads(config.read_text(encoding="utf-8"))
                model = active["provider"]["crono"]["models"]["Qwen3.6-35B-A3B-Q6_K"]
                self.assertEqual(active["model"], "crono/Qwen3.6-35B-A3B-Q6_K")
                self.assertEqual(model["modalities"]["input"], ["text", "image"])
                self.assertEqual(model["limit"]["context"], 262144)
                self.assertIs(model["reasoning"], True)
                self.assertEqual(model["interleaved"], {"field": "reasoning_content"})
                self.assertEqual(
                    model["variants"],
                    {
                        "off": {
                            "reasoningEffort": "none",
                            "reasoning_format": "auto",
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        "low": {
                            "reasoningEffort": "low",
                            "reasoning_format": "auto",
                            "chat_template_kwargs": {"enable_thinking": True},
                            "thinking_budget_tokens": 512,
                            "reasoning_budget_tokens": 512,
                            "reasoning_control": True,
                        },
                        "medium": {
                            "reasoningEffort": "medium",
                            "reasoning_format": "auto",
                            "chat_template_kwargs": {"enable_thinking": True},
                            "thinking_budget_tokens": 2048,
                            "reasoning_budget_tokens": 2048,
                            "reasoning_control": True,
                        },
                        "high": {
                            "reasoningEffort": "high",
                            "reasoning_format": "auto",
                            "chat_template_kwargs": {"enable_thinking": True},
                            "thinking_budget_tokens": 8192,
                            "reasoning_budget_tokens": 8192,
                            "reasoning_control": True,
                        },
                        "max": {
                            "reasoning_format": "auto",
                            "chat_template_kwargs": {"enable_thinking": True},
                            "reasoning_control": True,
                        },
                    },
                )
                launcher._deactivate_opencode_global()
                self.assertEqual(config.read_text(encoding="utf-8"), original)

    def test_opencode_adapter_reconciles_stale_legacy_crono_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "opencode.jsonc"
            state_file = root / "opencode-state.json"
            original = {
                "$schema": "https://opencode.ai/config.json",
                "disabled_providers": ["teste"],
                "provider": {
                    "teste": {
                        "name": "gemma-4",
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": "http://127.0.0.1:8080/v1"},
                        "models": {"gemma-4": {"name": "gemma-4"}},
                    },
                    "crono": {
                        "name": "crono-llama.cpp",
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {
                            "baseURL": "http://127.0.0.1:8080/v1",
                            "headers": {"Ornith-1.0": "0"},
                        },
                        "models": {
                            "Ornith-1.0-9B": {"name": "Ornith-1.0-9B"},
                        },
                    },
                },
            }
            original_text = json.dumps(original, ensure_ascii=False, indent=2) + "\n"
            config.write_text(original_text, encoding="utf-8")

            launcher = LauncherWebState.__new__(LauncherWebState)
            launcher.agent_info = {
                "model": "4BeastsOfApocalypse.Q6_K",
                "endpoint": "http://127.0.0.1:8080/v1",
                "context_window": 262144,
                "modalities": ["text"],
                "reasoning_enabled": True,
            }
            stale_provider = {
                "name": "Crono Matrix - llama.cpp",
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": "http://127.0.0.1:8081/v1"},
                "models": {"old-model": {"name": "old-model"}},
            }
            state_file.write_text(json.dumps({
                "active": True,
                "config_path": str(config),
                "original_exists": True,
                "backup_content": "{}\n",
                "active_sha256": "stale-state",
                "managed_provider": stale_provider,
                "managed_model": "crono/old-model",
            }), encoding="utf-8")

            with mock.patch("web.services.OPENCODE_GLOBAL_CONFIG", config), mock.patch(
                "web.services.OPENCODE_GLOBAL_STATE", state_file
            ):
                launcher._activate_opencode_global()
                active = json.loads(config.read_text(encoding="utf-8"))
                self.assertEqual(
                    active["model"], "crono/4BeastsOfApocalypse.Q6_K"
                )
                self.assertEqual(
                    active["provider"]["crono"]["models"]
                    ["4BeastsOfApocalypse.Q6_K"]["name"],
                    "4BeastsOfApocalypse.Q6_K",
                )
                saved = json.loads(state_file.read_text(encoding="utf-8"))
                self.assertTrue(saved["reconciled_legacy_provider"])
                self.assertEqual(saved["backup_content"], original_text)

                launcher._deactivate_opencode_global()
                self.assertEqual(config.read_text(encoding="utf-8"), original_text)

    def test_opencode_adapter_rejects_remote_provider_during_stale_state_reconcile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "opencode.jsonc"
            state_file = root / "opencode-state.json"
            config.write_text(json.dumps({
                "provider": {
                    "crono": {
                        "name": "external-crono",
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": "https://api.example.com/v1"},
                        "models": {"remote": {"name": "remote"}},
                    },
                },
            }), encoding="utf-8")
            state_file.write_text(json.dumps({
                "active": True,
                "managed_provider": {"name": "previous"},
                "backup_content": "{}\n",
            }), encoding="utf-8")

            launcher = LauncherWebState.__new__(LauncherWebState)
            launcher.agent_info = {
                "model": "local-model",
                "endpoint": "http://127.0.0.1:8080/v1",
                "context_window": 8192,
                "modalities": ["text"],
                "reasoning_enabled": False,
            }

            with mock.patch("web.services.OPENCODE_GLOBAL_CONFIG", config), mock.patch(
                "web.services.OPENCODE_GLOBAL_STATE", state_file
            ):
                with self.assertRaisesRegex(ValueError, "alterado fora"):
                    launcher._activate_opencode_global()

    def test_opencode_adapter_exposes_llama_reasoning_budgets_without_template_effort(self):
        launcher = LauncherWebState.__new__(LauncherWebState)
        launcher.agent_info = {
            "model": "local-model",
            "endpoint": "http://127.0.0.1:8080/v1",
            "context_window": 32768,
            "modalities": ["text"],
            "reasoning_enabled": True,
            "supports_reasoning_effort": False,
        }

        model = launcher._opencode_managed_provider()["models"]["local-model"]

        self.assertIs(model["reasoning"], True)
        self.assertEqual(model["interleaved"], {"field": "reasoning_content"})
        self.assertEqual(
            list(model["variants"]), ["off", "low", "medium", "high", "max"],
        )
        self.assertEqual(model["variants"]["low"]["thinking_budget_tokens"], 512)
        self.assertEqual(model["variants"]["medium"]["thinking_budget_tokens"], 2048)
        self.assertEqual(model["variants"]["high"]["thinking_budget_tokens"], 8192)
        self.assertNotIn("thinking_budget_tokens", model["variants"]["max"])

    def test_opencode_adapter_omits_reasoning_when_runtime_disables_it(self):
        launcher = LauncherWebState.__new__(LauncherWebState)
        launcher.agent_info = {
            "model": "plain-model",
            "endpoint": "http://127.0.0.1:8080/v1",
            "context_window": 8192,
            "modalities": ["text"],
            "reasoning_enabled": False,
            "supports_reasoning_effort": False,
        }

        model = launcher._opencode_managed_provider()["models"]["plain-model"]

        self.assertNotIn("reasoning", model)
        self.assertNotIn("interleaved", model)
        self.assertNotIn("variants", model)

    def test_qwen35_rtx3060_uses_measured_vram_knee(self):
        hardware, metadata = self._qwen35_fixture()
        optimal = OptimalParams(hardware, metadata)
        # Linearized from the real 262K fit-print measurements:
        # ncm=33 -> 11205 MiB; ncm=34 -> 10575 MiB.
        optimal._estimate_qwen_moe_gpu_mb = (
            lambda ctx, n_cpu_moe: 31995 - 630 * n_cpu_moe
        )

        self.assertEqual(optimal.fit_target, 256)
        self.assertTrue(optimal._adapt_qwen35moe())
        self.assertEqual(optimal.ctx, 262144)
        self.assertEqual(optimal.n_cpu_moe, 33)
        self.assertEqual(optimal.ngl, 40)
        self.assertEqual(optimal.fit_ctx, 262144)
        self.assertEqual(optimal.fit, "n")
        self.assertEqual(optimal.cpu_moe, "y")
        self.assertEqual(optimal.load_mode, "none")
        self.assertEqual(optimal.fit_plan_source, "qwen35moe-measured")

        state = LauncherWebState()
        values = state._optimal_values(optimal, metadata)
        self.assertEqual(
            {
                key: values[key]
                for key in ("ctx", "ngl", "n_cpu_moe", "fit", "cpu_moe", "fit_ctx")
            },
            {
                "ctx": 262144, "ngl": 40, "n_cpu_moe": 33,
                "fit": "n", "cpu_moe": "y", "fit_ctx": 262144,
            },
        )

    def test_parameter_snapshot_distinguishes_planner_from_effective_command(self):
        hardware, metadata = self._qwen35_fixture()
        state = LauncherWebState()
        state.hardware = hardware
        state.hardware_ready = True
        state.meta = metadata
        state.model_path = metadata.path

        optimal = OptimalParams(hardware, metadata)
        optimal.ctx = 18523
        optimal.ngl = 15
        optimal.fit = "y"
        optimal.kv_offload = "y"
        optimal.load_mode = "mmap"
        optimal.ctx_reason = "VRAM: 18523 tokens"
        optimal.ngl_reason = "15/40 camadas finais cabem na VRAM"
        optimal.kv_offload_reason = "habilitado — GPU gerencia o cache KV"
        optimal.load_mode_reason = "mmap — host planejado"
        state.opt = optimal
        state.params = state._optimal_values(optimal, metadata)
        # Simulate the user's intentional full-context/host choices after the
        # memory planner has produced its VRAM-only estimate.
        state.params.update({
            "ctx": 262144,
            "fit": "y",
            "fit_target": 256,
            "fit_ctx": 262144,
            "kv_offload": "n",
            "load_mode": "none",
            "cache_k": "bf16",
            "cache_v": "q8_0",
            "temp": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "min_p": 0.0,
        })
        state.profile_vram_free_mb = hardware.gpu_vram_free_mb

        snapshot = state.parameter_snapshot()
        facts = snapshot["command_facts"]
        self.assertEqual(facts["context"], 262144)
        self.assertEqual(facts["planner_context"], 18523)
        self.assertEqual(facts["gpu_layers"], "auto")
        self.assertEqual(facts["kv_offload"], "CPU (--no-kv-offload)")
        self.assertEqual(facts["load_mode"], "none")
        reasons = dict(snapshot["reasons"])
        self.assertIn("comando: --n-gpu-layers auto", reasons["GPU layers"])
        self.assertIn("comando efetivo: CPU", reasons["KV offload"])
        self.assertIn("comando/UI: load-mode none", reasons["Load mode"])

    def test_restore_optimal_profile_rebuilds_stale_vram_plan(self):
        hardware, metadata = self._qwen35_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "Qwen3.6-35B-A3B-Q6_K.gguf"
            model_path.touch()
            metadata.path = str(model_path)
            state = LauncherWebState(
                settings_file=Path(temporary) / "settings.json"
            )
            state.hardware = hardware
            state.hardware_ready = True
            state.meta = metadata
            state.model_path = str(model_path)
            state.params = {"omni": "n"}
            state.profile_vram_free_mb = 820

            def make_optimal(meta):
                optimal = OptimalParams(hardware, meta)
                optimal._estimate_qwen_moe_gpu_mb = (
                    lambda _ctx, n_cpu_moe: 31995 - 630 * n_cpu_moe
                )
                return optimal

            state._new_optimal_params = make_optimal
            snapshot = state.restore_optimal_profile()

        self.assertEqual(snapshot["values"]["ctx"], 262144)
        self.assertEqual(snapshot["values"]["n_cpu_moe"], 33)
        self.assertEqual(snapshot["values"]["load_mode"], "none")
        self.assertEqual(
            snapshot["planning_hardware"]["gpu_vram_free_mb"], 11468
        )

    def test_qwen36_uses_official_precise_coding_sampling(self):
        hardware, metadata = self._qwen35_fixture()
        optimal = OptimalParams(hardware, metadata)

        # The GGUF fixture embeds general-thinking temp=1.0, but the launcher
        # deliberately selects Qwen's task-specific precise-coding recipe.
        self.assertEqual(optimal.temp, 0.6)
        self.assertEqual(optimal.top_k, 20)
        self.assertEqual(optimal.top_p, 0.95)
        self.assertEqual(optimal.min_p, 0.0)
        self.assertEqual(optimal.presence_penalty, 0.0)
        self.assertIn("Qwen3.6", optimal.sampling_reason)

    def test_qwen36_command_overrides_llama_server_min_p_default(self):
        hardware, metadata = self._qwen35_fixture()
        optimal = OptimalParams(hardware, metadata)
        state = LauncherWebState()
        final = state._optimal_values(optimal, metadata)
        final["model_path"] = metadata.path

        command = optimal.build_cmd(final)

        self.assertEqual(command[command.index("--temp") + 1], "0.6")
        self.assertEqual(command[command.index("--top-k") + 1], "20")
        self.assertEqual(command[command.index("--top-p") + 1], "0.95")
        self.assertEqual(command[command.index("--min-p") + 1], "0.0")
        self.assertIn("--kv-offload", command)

    def test_partial_cpu_moe_does_not_emit_all_cpu_moe_override(self):
        hardware, metadata = self._qwen35_fixture()
        optimal = OptimalParams(hardware, metadata)
        state = LauncherWebState()
        final = state._optimal_values(optimal, metadata)
        final.update({
            "model_path": metadata.path,
            "cpu_moe": "y",
            "n_cpu_moe": 33,
        })

        command = optimal.build_cmd(final)

        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "33")
        self.assertNotIn("--cpu-moe", command)
        self.assertIn("--kv-offload", command)

    def test_native_server_tools_are_independent_from_crono_mcp(self):
        hardware, metadata = self._qwen35_fixture()
        optimal = OptimalParams(hardware, metadata)
        state = LauncherWebState()
        final = state._optimal_values(optimal, metadata)
        final.update({
            "model_path": metadata.path,
            "agentic": "y",
            "mcp_native": "n",
            "mcp_native_json": "",
        })

        command_without_mcp = optimal.build_cmd(final)

        self.assertIn("--agent", command_without_mcp)
        self.assertNotIn("--mcp-servers-json", command_without_mcp)

        final["mcp_native"] = "y"
        final["mcp_native_json"] = '{"mcpServers":{"crono-matrix":{}}}'
        command_with_mcp = optimal.build_cmd(final)

        self.assertIn("--agent", command_with_mcp)
        self.assertIn("--mcp-servers-json", command_with_mcp)

    def test_readonly_native_tools_use_names_exposed_by_server(self):
        hardware, metadata = self._qwen35_fixture()
        optimal = OptimalParams(hardware, metadata)
        state = LauncherWebState()
        final = state._optimal_values(optimal, metadata)
        final.update({
            "model_path": metadata.path,
            "agentic": "n",
            "tools": "readonly",
        })

        command = optimal.build_cmd(final)
        selected = command[command.index("--tools") + 1]

        self.assertEqual(
            selected, "read_file,file_glob_search,grep_search,get_info"
        )
        self.assertNotIn("get_datetime", selected)

    def test_lazy_tensor_flag_tracks_selected_llama_server_version(self):
        hardware, metadata = self._qwen35_fixture()
        optimal = OptimalParams(hardware, metadata)
        state = LauncherWebState()
        final = state._optimal_values(optimal, metadata)
        final.update({
            "model_path": metadata.path,
            "tensor_read_lazy": "on",
            "load_mode": "mmap",
        })

        with mock.patch(
            "launch_model_core._server_lazy_mode_flag", return_value="--lazy-mode"
        ):
            current = optimal.build_cmd(final)
        self.assertEqual(current[current.index("--lazy-mode") + 1], "on")

        with mock.patch(
            "launch_model_core._server_lazy_mode_flag",
            return_value="--tensor-read-lazy",
        ):
            legacy = optimal.build_cmd(final)
        self.assertEqual(
            legacy[legacy.index("--tensor-read-lazy") + 1], "on"
        )

    def test_laguna_uses_own_sampling_and_native_context_profile(self):
        hardware, metadata = self._laguna_fixture()
        optimal = OptimalParams(hardware, metadata)

        self.assertEqual(optimal.temp, 1.0)
        self.assertEqual(optimal.top_k, 20)
        self.assertEqual(optimal.top_p, 1.0)
        self.assertEqual(optimal.min_p, 0.0)
        self.assertEqual(optimal.fit_ctx, 262144)
        self.assertEqual(optimal.chat_template_kwargs, '{"enable_thinking":true}')
        self.assertIn("Laguna XS 2.1", optimal.sampling_reason)

    def test_laguna_bf16_q8_native_context_uses_measured_vram_knee(self):
        hardware, metadata = self._laguna_fixture()
        optimal = OptimalParams(hardware, metadata)

        def estimate(_ctx, n_cpu_moe):
            # Local fit-print at 262144: BF16/Q8 requires ncm=40;
            # Q8/Q8 reaches ncm=36 with the same 256 MiB reserve.
            base = 36151 if optimal.cache_k == "bf16" else 33705
            return base - 630 * n_cpu_moe

        optimal._estimate_qwen_moe_gpu_mb = estimate
        optimal.calculate()

        self.assertEqual((optimal.cache_k, optimal.cache_v), ("bf16", "q8_0"))
        self.assertEqual(optimal.ctx, 262144)
        self.assertEqual(optimal.ngl, 40)
        self.assertEqual(optimal.n_cpu_moe, 40)
        self.assertEqual(optimal.load_mode, "mmap")
        self.assertEqual(optimal.fit_plan_source, "laguna-measured")
        self.assertEqual(optimal.swa_full, "n")

        optimal.recalculate_memory("q8_0", "q8_0", 2048, 512)
        self.assertEqual(optimal.ctx, 262144)
        self.assertEqual(optimal.n_cpu_moe, 36)

    def test_laguna_web_recalculation_preserves_user_kv_and_replans(self):
        hardware, metadata = self._laguna_fixture()
        state = LauncherWebState()
        state.hardware = hardware
        state.hardware_ready = True
        state.meta = metadata
        state.model_path = metadata.path

        def make_optimal(meta):
            optimal = OptimalParams(hardware, meta)

            def estimate(_ctx, n_cpu_moe):
                base = 36151 if optimal.cache_k == "bf16" else 33705
                return base - 630 * n_cpu_moe

            optimal._estimate_qwen_moe_gpu_mb = estimate
            return optimal

        state._new_optimal_params = make_optimal
        initial = make_optimal(metadata)
        initial.calculate()
        state.opt = initial
        state.params = state._optimal_values(initial, metadata)

        self.assertEqual(state.params["ctx"], 262144)
        self.assertEqual(state.params["temp"], 1.0)
        self.assertEqual(state.params["top_p"], 1.0)
        self.assertEqual(state.params["n_cpu_moe"], 40)

        full_form = dict(state.params)
        full_form.update({
            "cache_k": "q8_0", "cache_v": "q8_0",
            "recalculate_field": "cache_k",
        })
        values = state.recalculate_memory(full_form)["values"]

        self.assertEqual((values["cache_k"], values["cache_v"]), ("q8_0", "q8_0"))
        self.assertEqual(values["ctx"], 262144)
        self.assertEqual(values["n_cpu_moe"], 36)
        self.assertEqual(values["temp"], 1.0)
        self.assertEqual(values["top_p"], 1.0)

    def test_laguna_profile_reaches_llama_server_command(self):
        hardware, metadata = self._laguna_fixture()
        optimal = OptimalParams(hardware, metadata)
        optimal._estimate_qwen_moe_gpu_mb = (
            lambda _ctx, n_cpu_moe: 36151 - 630 * n_cpu_moe
        )
        optimal.calculate()
        state = LauncherWebState()
        final = state._optimal_values(optimal, metadata)
        final["model_path"] = metadata.path
        final["alias"] = "Laguna-XS-2.1-Q6_K"

        command = optimal.build_cmd(final)

        def argument(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(argument("--ctx-size"), "262144")
        self.assertEqual(argument("--cache-type-k"), "bf16")
        self.assertEqual(argument("--cache-type-v"), "q8_0")
        self.assertEqual(argument("--n-cpu-moe"), "40")
        self.assertEqual(argument("--temp"), "1.0")
        self.assertEqual(argument("--top-k"), "20")
        self.assertEqual(argument("--top-p"), "1.0")
        self.assertEqual(argument("--min-p"), "0.0")
        self.assertEqual(
            argument("--chat-template-kwargs"), '{"enable_thinking":true}'
        )
        self.assertIn("--reasoning-preserve", command)
        self.assertIn("--fit", command)
        self.assertEqual(argument("--fit"), "off")

    def test_gemma4_uses_official_sampling_and_native_thinking(self):
        hardware, metadata = self._gemma4_fixture()
        optimal = OptimalParams(hardware, metadata)

        self.assertEqual(optimal.temp, 1.0)
        self.assertEqual(optimal.top_k, 64)
        self.assertEqual(optimal.top_p, 0.95)
        self.assertEqual(optimal.min_p, 0.0)
        self.assertEqual(optimal.fit_ctx, 262144)
        self.assertEqual(optimal.reasoning, "on")
        self.assertEqual(optimal.chat_template_kwargs, "")
        self.assertIn("Gemma 4", optimal.sampling_reason)

    def test_gemma4_swa_cache_accounting_and_measured_rtx3060_plan(self):
        hardware, metadata = self._gemma4_fixture()
        optimal = OptimalParams(hardware, metadata)
        optimal._estimate_qwen_moe_gpu_mb = (
            lambda _ctx, n_cpu_moe: round(31525 - 771.5 * n_cpu_moe)
        )
        optimal.calculate()

        self.assertEqual((optimal.cache_k, optimal.cache_v), ("bf16", "q8_0"))
        self.assertEqual(optimal.ctx, 262144)
        self.assertEqual(optimal.ngl, 30)
        self.assertEqual(optimal.n_cpu_moe, 27)
        self.assertEqual(optimal.load_mode, "none")
        self.assertEqual(optimal.swa_full, "n")
        self.assertEqual(optimal.spec_type, "none")
        self.assertEqual((optimal.threads, optimal.threads_batch), (12, 16))
        self.assertEqual((optimal.batch, optimal.ubatch), (2048, 512))
        self.assertLess(
            optimal._cache_bytes_for_context(262144, "bf16") / 1048576,
            4300,
        )

    def test_gemma4_command_and_cpu_mmproj_preserve_native_context(self):
        hardware, metadata = self._gemma4_fixture()
        metadata.mmproj_file = "/models/mmproj-gemma-4-26B-A4B-it-BF16.gguf"
        metadata.mmproj_size_mb = 1139
        metadata.mmproj_valid = True
        optimal = OptimalParams(hardware, metadata)
        optimal.vision_enabled = True
        optimal._estimate_qwen_moe_gpu_mb = (
            lambda _ctx, n_cpu_moe: round(31525 - 771.5 * n_cpu_moe)
        )
        optimal.calculate()

        # With vision enabled, CPU experts remain the hot set and the cold
        # resident excess has an explicitly higher-priority NVMe route.
        self.assertEqual(optimal.n_cpu_moe, 27)
        self.assertEqual(optimal.load_mode, "none")
        self.assertEqual(optimal.ctx_checkpoints, 2)
        self.assertEqual(optimal.swap_recommended_gib, 20)
        self.assertEqual(optimal.mmproj_offload, "n")
        self.assertEqual(optimal.mtmd_batch_max, 512)

        state = LauncherWebState()
        final = state._optimal_values(optimal, metadata)
        final.update({
            "model_path": metadata.path,
            "omni": "y",
            "no_mmproj_auto": "y",
        })
        with mock.patch("launch_model_core._server_supports_flag", return_value=True):
            command = optimal.build_cmd(final)

        def argument(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(argument("--ctx-size"), "262144")
        self.assertEqual(argument("--cache-type-k"), "bf16")
        self.assertEqual(argument("--cache-type-v"), "q8_0")
        self.assertEqual(argument("--n-cpu-moe"), "27")
        self.assertEqual(argument("--temp"), "1.0")
        self.assertEqual(argument("--top-k"), "64")
        self.assertEqual(argument("--top-p"), "0.95")
        self.assertEqual(argument("--min-p"), "0.0")
        self.assertNotIn("--chat-template-kwargs", command)
        self.assertEqual(argument("--reasoning"), "on")
        self.assertEqual(argument("--mtmd-batch-max-tokens"), "512")
        self.assertIn("--no-mmproj-offload", command)
        self.assertEqual(argument("--mmproj-device"), "none")
        self.assertNotIn("--swa-full", command)
        self.assertEqual(argument("--fit"), "off")

    def test_glm47_profile_uses_native_context_and_official_sampling(self):
        hardware, _ = self._qwen35_fixture()
        metadata = ModelMetadata()
        metadata.arch = "DEEPSEEK2"
        metadata.general_basename = "GLM 4.7 Flash"
        metadata.path = "/models/zai-org_GLM-4.7-Flash-Q6_K_L.gguf"
        metadata.ctx_max = 202752
        metadata.layers = 47
        metadata.size_mb = 23823

        optimal = OptimalParams(hardware, metadata)

        self.assertEqual(optimal.fit_ctx, 202752)
        self.assertEqual(optimal.temp, 1.0)
        self.assertEqual(optimal.top_p, 0.95)
        self.assertEqual(optimal.top_k, 0)
        self.assertEqual(optimal.min_p, 0.0)
        self.assertIn("GLM-4.7-Flash oficial", optimal.sampling_reason)

        optimal.cache_k = optimal.cache_v = "q8_0"
        optimal._flash()
        self.assertEqual(optimal.flash, "y")
        optimal.cache_k = optimal.cache_v = "f16"
        optimal._flash()
        self.assertEqual(optimal.flash, "n")

    def test_glm47_fixed_fit_context_reaches_server_command(self):
        hardware, _ = self._qwen35_fixture()
        metadata = ModelMetadata()
        metadata.arch = "DEEPSEEK2"
        metadata.general_basename = "GLM 4.7 Flash"
        metadata.path = "/models/GLM-4.7-Flash-Q6_K_L.gguf"
        metadata.ctx_max = 202752
        metadata.layers = 47
        optimal = OptimalParams(hardware, metadata)
        final = {
            "model_path": metadata.path,
            "ctx": 202752, "fit_ctx": 202752, "fit": "y",
            "fit_target": 256, "parallel": 1,
            "cache_k": "q8_0", "cache_v": "q8_0",
            "batch": 2048, "ubatch": 512, "threads": 12,
            "threads_batch": 16, "host": "127.0.0.1", "port": 8080,
            "ngl": 48, "flash": "y", "temp": 1.0,
            "top_k": 0, "top_p": 0.95, "min_p": 0.0,
            "load_mode": "none", "spec_type": "none",
        }

        command = optimal.build_cmd(final)

        self.assertEqual(command[command.index("--ctx-size") + 1], "202752")
        self.assertEqual(command[command.index("--fit-ctx") + 1], "202752")
        self.assertEqual(command[command.index("--temp") + 1], "1.0")
        self.assertEqual(command[command.index("--top-k") + 1], "0")
        self.assertEqual(command[command.index("--min-p") + 1], "0.0")
        self.assertEqual(command[command.index("--load-mode") + 1], "none")

    def test_glm47_web_recalculation_keeps_kv_symmetric(self):
        hardware, _ = self._qwen35_fixture()
        metadata = ModelMetadata()
        metadata.meta_ok = True
        metadata.arch = "DEEPSEEK2"
        metadata.general_basename = "GLM 4.7 Flash"
        metadata.path = "/models/GLM-4.7-Flash-Q6_K_L.gguf"
        metadata.ctx_max = 202752
        metadata.layers = 47
        metadata.size_mb = 23823
        metadata.size_bytes = metadata.size_mb * 1048576

        state = LauncherWebState()
        state.hardware = hardware
        state.hardware_ready = True
        state.meta = metadata
        state.model_path = metadata.path

        def make_optimal(meta):
            optimal = OptimalParams(hardware, meta)
            optimal._fit_plan = lambda requested_ctx=None: {
                "ctx": 202752, "ngl": 48,
                "overrides": "blk\\.7\\.ffn_.*=CPU",
                "command": "-c 202752 -ngl 48",
            }
            return optimal

        state._new_optimal_params = make_optimal
        initial = make_optimal(metadata)
        initial.calculate()
        state.opt = initial
        state.params = state._optimal_values(initial, metadata)

        full_form = dict(state.params)
        full_form.update({
            "cache_k": "f16", "cache_v": "q8_0",
            "recalculate_field": "cache_k",
        })
        values = state.recalculate_memory(full_form)["values"]

        self.assertEqual(values["cache_k"], "f16")
        self.assertEqual(values["cache_v"], "f16")
        self.assertEqual(values["ctx"], 202752)
        self.assertEqual(values["flash"], "n")

    def test_qwen_moe_estimate_cache_includes_parallel_slots(self):
        hardware, metadata = self._qwen35_fixture()
        metadata.expert_weight_bytes_by_layer = [630 * 1048576] * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata.path = str(root / "model.gguf")
            fit_binary = root / "llama-fit-params"
            Path(metadata.path).write_bytes(b"gguf")
            fit_binary.write_bytes(b"fit")
            optimal = OptimalParams(
                hardware, metadata, llama_fit_params=str(fit_binary)
            )
            completed = mock.Mock(stdout="CUDA0 1553 3980 1006\n", returncode=0)
            with mock.patch("launch_model_core.subprocess.run", return_value=completed) as run:
                first = optimal._estimate_qwen_moe_gpu_mb(262144, 33)
                second = optimal._estimate_qwen_moe_gpu_mb(262144, 33)
                neighbor = optimal._estimate_qwen_moe_gpu_mb(262144, 34)
                optimal.parallel = 2
                third = optimal._estimate_qwen_moe_gpu_mb(262144, 33)

            self.assertEqual(first, 10949)
            self.assertEqual(second, first)
            self.assertEqual(neighbor, 10319)
            self.assertEqual(third, first)
            self.assertEqual(run.call_count, 2)
            self.assertIn("-np", run.call_args.args[0])
            self.assertIn("2", run.call_args.args[0])
            self.assertEqual(
                run.call_args.args[0][run.call_args.args[0].index("-ncmoe") + 1],
                "40",
            )

    def test_dynamic_nvme_swap_is_sized_from_model_memory_gap(self):
        hardware = HardwareInfo()
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 20 * 1024
        hardware.disk_free_gb = "80"
        hardware.swap_zram_total_mb = 32 * 1024
        metadata = ModelMetadata()
        metadata.size_mb = 30 * 1024
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.layers = 40
        metadata.ctx_max = 262144

        optimal = OptimalParams(hardware, metadata)
        optimal.ngl = 0
        optimal.ctx = 4096
        optimal._plan_host_memory()

        self.assertEqual(optimal.load_mode, "mmap")
        # In addition to the model gap, the pre-flight plan must reserve the
        # configured 2 GiB prompt cache.  This capacity used to be invisible
        # until the server had already been running for a while.
        self.assertEqual(optimal.swap_recommended_gib, 20)
        self.assertIn("sem contar como RAM rápida", optimal.swap_plan_reason)

    def test_host_plan_reserves_hybrid_context_checkpoints_separately(self):
        hardware = HardwareInfo()
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 20 * 1024
        hardware.disk_free_gb = "80"
        metadata = ModelMetadata()
        metadata.size_mb = 1024
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.layers = 4
        metadata.ctx_max = 262144
        metadata.layer_layout_valid = True
        metadata.recurrent_layers = 2
        # Two recurrent layers × 1 Mi elements × fp32 = 8 MiB per slot.
        metadata.state_r = 1024 * 1024
        metadata.state_s = 0

        optimal = OptimalParams(hardware, metadata)
        optimal.ctx = 262144
        optimal.parallel = 2
        optimal.cache_ram = 256
        optimal.ctx_checkpoints = 4
        optimal._plan_host_memory()

        self.assertEqual(optimal.checkpoint_snapshot_mb, 8)
        self.assertEqual(optimal.checkpoint_peak_mb, 64)
        self.assertEqual(optimal.prompt_cache_peak_mb, 256)
        self.assertIn("cache de prompt até 256 MB", optimal.host_growth_reason)
        self.assertIn("checkpoints até 64 MB", optimal.host_growth_reason)

    def test_host_plan_reserves_attention_checkpoint_state_when_not_hybrid(self):
        hardware = HardwareInfo()
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 20 * 1024
        hardware.disk_free_gb = "80"
        metadata = ModelMetadata()
        metadata.size_mb = 1024
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.layers = 2
        metadata.ctx_max = 4096
        metadata.kv_layers = 2
        metadata.heads_kv = 8
        metadata.key_len = 128
        metadata.val_len = 128

        optimal = OptimalParams(hardware, metadata)
        optimal.ctx = 4096
        optimal.parallel = 1
        optimal.cache_k = "bf16"
        optimal.cache_v = "q8_0"
        optimal.cache_ram = 0
        optimal.ctx_checkpoints = 2
        optimal._plan_host_memory()

        self.assertGreater(optimal.checkpoint_snapshot_mb, 0)
        self.assertEqual(
            optimal.checkpoint_peak_mb,
            optimal.checkpoint_snapshot_mb * optimal.ctx_checkpoints,
        )

    def test_active_nvme_swap_is_reported_as_satisfying_dynamic_plan(self):
        hardware = HardwareInfo()
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 20 * 1024
        hardware.disk_free_gb = "64"
        hardware.swap_nvme_total_mb = 24 * 1024
        hardware.swap_nvme_active = True
        hardware.swap_zram_priority = 100
        hardware.swap_nvme_priority = 110
        hardware.swap_nvme_preferred = True
        metadata = ModelMetadata()
        metadata.size_mb = 30 * 1024
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.layers = 40

        optimal = OptimalParams(hardware, metadata)
        optimal.ngl = 0
        optimal._plan_host_memory()

        self.assertLessEqual(optimal.swap_recommended_gib, 24)
        self.assertIn("atende a recomendação", optimal.swap_plan_reason)

    def test_lower_priority_nvme_swap_is_not_reported_as_safe(self):
        hardware = HardwareInfo()
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 20 * 1024
        hardware.disk_free_gb = "64"
        hardware.swap_zram_total_mb = 32 * 1024
        hardware.swap_zram_priority = 100
        hardware.swap_nvme_total_mb = 24 * 1024
        hardware.swap_nvme_priority = 10
        hardware.swap_nvme_active = True
        hardware.swap_nvme_preferred = False
        metadata = ModelMetadata()
        metadata.size_mb = 30 * 1024
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.layers = 40

        optimal = OptimalParams(hardware, metadata)
        optimal.ngl = 0
        optimal._plan_host_memory()

        self.assertEqual(optimal.load_mode, "mmap")
        self.assertIn("prioridade 10", optimal.swap_plan_reason)
        self.assertIn("ZRAM 100", optimal.swap_plan_reason)

    def test_full_form_kv_recalculation_preserves_kv_and_refreshes_moe_plan(self):
        hardware, metadata = self._qwen35_fixture()
        state = LauncherWebState()
        state.hardware = hardware
        state.hardware_ready = True
        state.meta = metadata
        state.model_path = metadata.path

        def make_optimal(meta):
            optimal = OptimalParams(hardware, meta)

            def estimate(_ctx, n_cpu_moe):
                # F16/F16 consome KV suficiente para mover mais duas camadas
                # de experts à CPU que o perfil BF16/Q8_0.
                base = 33255 if optimal.cache_v == "f16" else 31995
                return base - 630 * n_cpu_moe

            optimal._estimate_qwen_moe_gpu_mb = estimate
            return optimal

        state._new_optimal_params = make_optimal
        initial = make_optimal(metadata)
        initial.calculate()
        state.opt = initial
        state.params = state._optimal_values(initial, metadata)
        self.assertEqual(state.params["n_cpu_moe"], 33)

        full_form = dict(state.params)
        full_form.update({
            "cache_k": "f16", "cache_v": "f16",
            "recalculate_field": "cache_k",
        })
        values = state.recalculate_memory(full_form)["values"]

        self.assertEqual(values["cache_k"], "f16")
        self.assertEqual(values["cache_v"], "f16")
        self.assertEqual(values["ctx"], 262144)
        self.assertEqual(values["n_cpu_moe"], 35)

    def test_form_cache_and_checkpoint_values_reach_host_swap_plan(self):
        hardware = HardwareInfo()
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 20 * 1024
        hardware.disk_free_gb = "80"
        metadata = ModelMetadata()
        metadata.arch = "LLAMA"
        metadata.size_mb = 1024
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.layers = 2
        metadata.ctx_max = 4096
        metadata.kv_layers = 2
        metadata.heads_kv = 8
        metadata.key_len = 128
        metadata.val_len = 128

        state = LauncherWebState()
        state.hardware = hardware
        state.hardware_ready = True
        state.meta = metadata
        state.model_path = "/models/large.gguf"
        initial = OptimalParams(hardware, metadata)
        initial.calculate()
        state.opt = initial
        state.params = state._optimal_values(initial, metadata)

        full_form = dict(state.params)
        full_form.update({
            "cache_ram": "4096",
            "ctx_checkpoints": "0",
            "recalculate_field": "ctx_checkpoints",
        })
        state.recalculate_memory(full_form)

        self.assertEqual(state.opt.prompt_cache_peak_mb, 4096)
        self.assertEqual(state.opt.ctx_checkpoints, 0)
        self.assertEqual(state.opt.checkpoint_peak_mb, 0)

    def test_form_context_change_replans_host_state_without_overwriting_input(self):
        hardware = HardwareInfo()
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 20 * 1024
        hardware.disk_free_gb = "80"
        metadata = ModelMetadata()
        metadata.arch = "LLAMA"
        metadata.size_mb = 1024
        metadata.size_bytes = metadata.size_mb * 1048576
        metadata.layers = 2
        metadata.ctx_max = 4096
        metadata.kv_layers = 2
        metadata.heads_kv = 8
        metadata.key_len = 128
        metadata.val_len = 128

        state = LauncherWebState()
        state.hardware = hardware
        state.hardware_ready = True
        state.meta = metadata
        state.model_path = "/models/large.gguf"
        initial = OptimalParams(hardware, metadata)
        initial.calculate()
        state.opt = initial
        state.params = state._optimal_values(initial, metadata)

        full_form = dict(state.params)
        full_form.update({"ctx": "2048", "recalculate_field": "ctx"})
        values = state.recalculate_memory(full_form)["values"]

        self.assertEqual(values["ctx"], 2048)
        self.assertEqual(state.opt.ctx, 2048)


if __name__ == "__main__":
    unittest.main()
