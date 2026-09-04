import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launch_model_core import HardwareInfo, ModelMetadata, OptimalParams
from launch_model_ui import LauncherApp, LOCAL_MCP_DIR, LOCAL_MCP_ENTRY


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return str(self.value)


class _BoolVar:
    def __init__(self, value):
        self.value = bool(value)

    def get(self):
        return self.value


class DesktopGuiTests(unittest.TestCase):
    @staticmethod
    def _app(model_path: str, mcp=False):
        hardware = HardwareInfo()
        hardware.cpu_cores = 16
        hardware.cpu_threads = 32
        hardware.ram_total_gb = 31.2
        hardware.ram_avail_mb = 27673
        hardware.gpu_detected = True
        hardware.gpu_model = "NVIDIA GeForce RTX 3060"
        hardware.gpu_vram_mb = 12288
        hardware.gpu_vram_free_mb = 11480

        metadata = ModelMetadata()
        metadata.path = model_path
        metadata.meta_ok = True
        metadata.arch = "QWEN35MOE"
        metadata.ctx_max = 262144
        metadata.layers = 40
        metadata.moe_layers = 40
        metadata.size_mb = 27193

        optimal = OptimalParams(hardware, metadata)
        app = LauncherApp.__new__(LauncherApp)
        app.model_path = model_path
        app.meta = metadata
        app.opt = optimal
        app.mcp_enabled = _BoolVar(mcp)
        app.models_dir_var = _Var(str(Path(model_path).parent))
        app._pvars = {
            "ctx": _Var(262144),
            "ngl": _Var("all"),
            "parallel": _Var(1),
            "cache_k": _Var("bf16"),
            "cache_v": _Var("q8_0"),
            "mlock": _Var("n"),
            "no_mmap": _Var("n"),
            "n_cpu_moe": _Var(33),
            "fit": _Var("y"),
            "fit_target": _Var(256),
            "fit_ctx": _Var(16384),
            "load_mode": _Var("none"),
            "tensor_read_lazy": _Var("auto"),
            "n_cpu_ffn": _Var(0),
            "backend_sampling": _Var("y"),
            "cont_batching": _Var("y"),
            "cache_prompt": _Var("y"),
            "cache_idle_slots": _Var("y"),
            "offline": _Var("y"),
            "alias": _Var("Qwen3.6-35B-A3B-Q6_K"),
        }
        return app

    def test_offline_large_model_command_uses_current_core_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "Qwen3.6-35B-A3B-Q6_K.gguf"
            model.write_bytes(b"GGUF")
            app = self._app(str(model))

            final = app._get_final()
            command = app.opt.build_cmd(final)

            self.assertEqual(final["ctx"], 262144)
            self.assertEqual(final["n_cpu_moe"], 33)
            self.assertEqual(final["load_mode"], "none")
            self.assertEqual(final["offline"], "y")
            self.assertEqual(command[command.index("--ctx-size") + 1], "262144")
            self.assertEqual(command[command.index("--n-cpu-moe") + 1], "33")
            self.assertIn("--offline", command)
            self.assertEqual(command[command.index("--fit") + 1], "off")

    def test_qwen_cpu_mmproj_cannot_be_overridden_on_12gb_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "Qwen3.6-35B-A3B-Q6_K.gguf"
            mmproj = root / "mmproj-Qwen3.6-35B-A3B-BF16.gguf"
            model.write_bytes(b"GGUF")
            mmproj.write_bytes(b"GGUF")
            app = self._app(str(model))
            app.meta.mmproj_file = str(mmproj)
            app.meta.mmproj_valid = True
            app.opt.meta = app.meta
            app._pvars["omni"] = _Var("y")
            # Até uma tentativa manual de GPU deve ser recusada pelo comando
            # efetivo neste hardware; o projetor de ~860 MiB é carregado por
            # último e pressionaria a VRAM já ocupada pelo Qwen 35B.
            app._pvars["mmproj_offload"] = _Var("y")

            final = app._get_final()
            with mock.patch("launch_model_core._server_supports_flag", return_value=True):
                command = app.opt.build_cmd(final)

            self.assertIn("--no-mmproj-offload", command)
            self.assertEqual(
                command[command.index("--mmproj-device") + 1], "none"
            )

    def test_legacy_memory_recalculation_preserves_explicit_fit_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "Qwen3.6-35B-A3B-Q6_K.gguf"
            model.write_bytes(b"GGUF")
            app = self._app(str(model))
            app.hw = app.opt.hw
            app.model_path = str(model)
            app._recalc_request = 1
            app._post_ui = lambda callback, *args: callback(*args)
            app._update_cmd_preview = lambda: None
            app.status_var = _Var("")
            app.p_reason_var = _Var("")
            result = {}

            def capture(_request_id, _changed, _model_path, _snapshot, opt, error):
                result["fit_ctx"] = None if opt is None else opt.fit_ctx
                result["error"] = error

            app._memory_recalc_done = capture
            snapshot = {
                "ctx": 262144, "parallel": 1, "fit_target": 256,
                "fit_ctx": 16384, "fit": "y", "cache_k": "f16",
                "cache_v": "q8_0", "kv_offload": "y", "batch": 2048,
                "ubatch": 512, "cache_ram": 2048, "ctx_checkpoints": 32,
                "spec_type": "none", "swa_full": "n", "omni": "n",
            }
            # Keep the test focused on preservation. The fake GGUF does not
            # contain enough tensor information for the adaptive Qwen fit.
            with mock.patch(
                "launch_model_ui.OptimalParams._adapt_qwen35moe",
                return_value=True,
            ):
                app._memory_recalc_thread(
                    1, "cache_k", str(model), app.hw, app.meta, snapshot
                )

            self.assertEqual(result, {"fit_ctx": 16384, "error": ""})


if __name__ == "__main__":
    unittest.main()
