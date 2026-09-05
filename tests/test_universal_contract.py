"""Contract tests; synthetic metadata/fit output, not inference benchmarks."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launch_model_core import HardwareInfo, ModelMetadata, OptimalParams
from web.services import LauncherWebState


class UniversalContractTests(unittest.TestCase):
    def setUp(self):
        self.state = LauncherWebState.__new__(LauncherWebState)
        self.state.meta = ModelMetadata()
        self.state.meta.mmproj_file = "/models/projector.gguf"
        self.state.meta.mmproj_valid = True

    def capabilities(self, props=None, **final):
        return self.state._runtime_agent_capabilities(
            {"omni": "y", **final}, props or {}, {}, {}, False
        )

    def test_audio_projector_is_not_a_vision_encoder(self):
        self.state.meta.mmproj_has_audio = True
        capabilities, modalities, _ = self.capabilities()
        self.assertEqual(modalities, ["text", "audio"])
        self.assertEqual(capabilities["evidence"]["input_modalities"], "gguf_preview")
        self.assertFalse(capabilities["input"]["image"])
        self.assertFalse(capabilities["output"]["audio"])

    def test_combined_projector_preview_requires_enabled_media(self):
        self.state.meta.mmproj_has_audio = True
        self.state.meta.mmproj_has_vision = True
        self.assertEqual(self.capabilities()[1], ["text", "image", "audio"])
        self.assertEqual(self.capabilities(omni="n")[1], ["text"])
        self.state.meta.mmproj_valid = False
        self.assertEqual(self.capabilities()[1], ["text"])

    def test_runtime_denial_wins_over_projector_metadata(self):
        self.state.meta.mmproj_has_audio = True
        self.state.meta.mmproj_has_vision = True
        capabilities, modalities, _ = self.capabilities({
            "modalities": {"vision": False, "audio": False, "video": False}
        })
        self.assertEqual(modalities, ["text"])
        self.assertEqual(capabilities["evidence"]["input_modalities"], "runtime_props")

    def test_older_runtime_without_modality_flags_remains_unknown(self):
        self.state.meta.mmproj_has_audio = True
        self.state.meta.mmproj_has_vision = True
        capabilities, modalities, _ = self.capabilities({"model_alias": "older"})
        self.assertEqual(modalities, ["text"])
        self.assertEqual(capabilities["evidence"]["input_modalities"], "unknown")

    def test_runtime_omni_modalities_do_not_depend_on_filename(self):
        self.state.meta.mmproj_valid = False
        capabilities, modalities, _ = self.capabilities({
            "modalities": {"vision": True, "audio": True, "video": True}
        }, omni="n")
        self.assertEqual(modalities, ["text", "image", "video", "audio"])
        self.assertTrue(capabilities["attachment"])

    def test_native_context_is_not_inflated_by_requested_extension(self):
        self.state.meta.meta_ok = True
        self.state.meta.ctx_max = 32768
        capabilities, _, _ = self.capabilities(ctx=262144)
        self.assertEqual(capabilities["model"]["native_context_window"], 32768)
        self.assertEqual(capabilities["runtime"]["context_window"], 262144)
        self.state.meta.meta_ok = False
        self.assertEqual(self.state._agent_native_context(), 0)

    def test_zero_reasoning_budget_is_not_changed_to_unlimited(self):
        self.assertEqual(self.capabilities(reasoning_budget=0)[2]["reasoning_budget_tokens"], 0)

    def test_unknown_architecture_keeps_embedded_sampling_on_different_hardware(self):
        metadata = ModelMetadata()
        metadata.arch = "FUTURE_ARCHITECTURE"
        metadata.sampling_temp = 0.37
        metadata.sampling_top_k = 47
        metadata.sampling_top_p = 0.82
        metadata.sampling_min_p = 0.02
        for vram in (0, 12288, 98304):
            hardware = HardwareInfo()
            hardware.gpu_detected = bool(vram)
            hardware.gpu_vram_mb = vram
            optimal = OptimalParams(hardware, metadata)
            self.assertEqual((optimal.temp, optimal.top_k, optimal.top_p, optimal.min_p),
                             (0.37, 47, 0.82, 0.02))

    def test_native_fit_cache_tracks_hardware_snapshot_and_rejects_failure(self):
        hardware = HardwareInfo()
        hardware.gpu_detected = True
        hardware.gpu_vram_mb = 12288
        hardware.gpu_vram_free_mb = 11000
        hardware.ram_avail_mb = 24000
        metadata = ModelMetadata()
        metadata.arch = "FUTURE_ARCHITECTURE"
        metadata.ctx_max = 262144
        with tempfile.TemporaryDirectory() as directory:
            metadata.path = str(Path(directory) / "fixture.gguf")
            binary = Path(directory) / "llama-fit-params"
            Path(metadata.path).touch()
            binary.touch()
            optimal = OptimalParams(hardware, metadata, llama_fit_params=str(binary))
            result = mock.Mock(returncode=0, stdout="-c 262144 -ngl 40\n")
            with mock.patch("launch_model_core.subprocess.run", return_value=result) as run:
                self.assertIsNotNone(optimal._fit_plan())
                self.assertIsNotNone(optimal._fit_plan())
                self.assertEqual(run.call_count, 1)
                hardware.gpu_vram_free_mb = 8000
                self.assertIsNotNone(optimal._fit_plan())
                self.assertEqual(run.call_count, 2)
                hardware.ram_avail_mb = 20000
                self.assertIsNotNone(optimal._fit_plan())
                self.assertEqual(run.call_count, 3)
                hardware.gpu_model = "different accelerator"
                self.assertIsNotNone(optimal._fit_plan())
                self.assertEqual(run.call_count, 4)
                hardware.gpu_vram_free_mb = 7000
                result.returncode = 1
                self.assertIsNone(optimal._fit_plan())
                result.returncode = 0
                result.stdout = "-c 0 -ngl -1\n"
                self.assertIsNone(optimal._fit_plan())


if __name__ == "__main__":
    unittest.main()
