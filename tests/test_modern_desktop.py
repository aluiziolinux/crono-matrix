import unittest
from unittest import mock

from launch_model_core import SPECULATIVE_TYPES
from launch_model_ctk import (
    CronoDesktop, adaptive_window_metrics, decoration_geometry_pulse,
    format_bytes, hf_download_view, memory_guard_view, vram_plan_action,
)


class AdaptiveDesktopGeometryTests(unittest.TestCase):
    def test_vram_plan_is_rebuilt_after_large_release(self):
        self.assertEqual(vram_plan_action(820, 11200), "rebuild")

    def test_vram_plan_blocks_launch_after_large_consumption(self):
        self.assertEqual(vram_plan_action(11200, 820), "block")

    def test_small_vram_fluctuation_keeps_current_plan(self):
        self.assertEqual(vram_plan_action(11200, 10900), "keep")

    def test_memory_controls_that_change_fit_trigger_recalculation(self):
        self.assertTrue(
            {
                "ctx", "parallel", "fit_target", "fit", "kv_offload",
                "cache_ram", "ctx_checkpoints", "fit_ctx",
            }.issubset(CronoDesktop.MEMORY_FIELDS)
        )

    def test_speculative_selector_uses_only_llama_server_values(self):
        self.assertNotIn("draft", SPECULATIVE_TYPES)
        self.assertNotIn("ngram", SPECULATIVE_TYPES)
        self.assertIn("draft-mtp", SPECULATIVE_TYPES)
        self.assertIn("ngram-mod", SPECULATIVE_TYPES)

    def test_hd_notebook_uses_compact_scale_and_stays_inside_screen(self):
        metrics = adaptive_window_metrics(1366, 768)
        self.assertEqual(metrics["scale"], 0.78)
        self.assertLessEqual(metrics["width"], 1366 - 24)
        self.assertLessEqual(metrics["height"], 768 - 54)
        self.assertLessEqual(metrics["min_width"], metrics["width"])
        self.assertLessEqual(metrics["min_height"], metrics["height"])

    def test_1440_by_900_uses_balanced_scale(self):
        metrics = adaptive_window_metrics(1440, 900)
        self.assertEqual(metrics["scale"], 0.86)
        self.assertEqual(metrics["width"], 1310)
        self.assertEqual(metrics["height"], 756)

    def test_full_hd_preserves_more_density_without_maximizing(self):
        metrics = adaptive_window_metrics(1920, 1080)
        self.assertEqual(metrics["scale"], 0.94)
        self.assertEqual(metrics["width"], 1480)
        self.assertLess(metrics["height"], 1080)

    def test_minimum_supported_screen_never_forces_larger_minimum(self):
        metrics = adaptive_window_metrics(800, 600)
        self.assertLessEqual(metrics["min_width"], metrics["width"])
        self.assertLessEqual(metrics["min_height"], metrics["height"])

    def test_memory_guard_view_preserves_native_live_values(self):
        view = memory_guard_view({
            "available_mb": "1200", "current_mb": "27100",
            "trigger_mb": 1536, "memory_high_mb": 0,
            "pressure_count": 3, "scope_phase": "inference",
        })
        self.assertEqual(view["available_mb"], 1200)
        self.assertEqual(view["current_mb"], 27100)
        self.assertEqual(view["memory_high"], "max")
        self.assertTrue(view["pressure"])

    def test_memory_guard_view_handles_idle_snapshot(self):
        view = memory_guard_view(None)
        self.assertEqual(view["scope_phase"], "idle")
        self.assertEqual(view["scope_unit"], "—")
        self.assertFalse(view["pressure"])

    def test_kwin_decoration_pulse_preserves_final_geometry(self):
        self.assertEqual(
            decoration_geometry_pulse("1040x594+120+40"),
            ("1041x594+120+40", "1040x594+120+40"),
        )

    def test_invalid_decoration_geometry_is_ignored(self):
        self.assertIsNone(decoration_geometry_pulse("zoomed"))

    def test_hugging_face_download_progress_is_bounded_and_formatted(self):
        view = hf_download_view({
            "state": "running", "downloaded": 3 * 1024**3,
            "total": 6 * 1024**3, "speed": 24 * 1024**2,
            "filename": "model-Q6_K.gguf",
        })
        self.assertEqual(view["state"], "running")
        self.assertEqual(view["progress"], 0.5)
        self.assertEqual(view["percent"], 50.0)
        self.assertEqual(format_bytes(view["speed"]), "24.0 MiB")

    def test_hugging_face_download_handles_unknown_total(self):
        view = hf_download_view({"state": "running", "downloaded": 4096})
        self.assertEqual(view["progress"], 0.0)
        self.assertEqual(view["label"], "BAIXANDO")

    def test_hugging_face_download_never_exceeds_full_progress(self):
        view = hf_download_view({"state": "done", "downloaded": 12, "total": 10})
        self.assertEqual(view["progress"], 1.0)
        self.assertEqual(view["label"], "CONCLUÍDO")

    def test_radar_selection_requests_real_repository_details(self):
        desktop = CronoDesktop.__new__(CronoDesktop)
        desktop.backend = mock.Mock()
        desktop.backend.hf_details.return_value = {"repo_id": "owner/model", "files": []}
        desktop.hf_file_var = mock.Mock()
        desktop.hf_detail_title = mock.Mock()
        desktop.hf_detail_meta = mock.Mock()
        desktop.hf_download_button = mock.Mock()
        desktop._render_hf_detail = mock.Mock()
        desktop._set_status = mock.Mock()
        desktop._run_task = lambda action, done, _status: done(action())

        CronoDesktop._open_radar_model(desktop, "owner/model")

        desktop.backend.hf_details.assert_called_once_with("owner/model")
        desktop._render_hf_detail.assert_called_once_with(
            {"repo_id": "owner/model", "files": []}
        )

    def test_selected_gguf_starts_backend_download(self):
        desktop = CronoDesktop.__new__(CronoDesktop)
        desktop.backend = mock.Mock(models_dir="/models")
        desktop.backend.start_download.return_value = {
            "state": "running", "filename": "model-Q6_K.gguf",
            "downloaded": 0, "total": 1024,
        }
        desktop._radar_selected_repo = "owner/model-GGUF"
        desktop.hf_file_var = mock.Mock()
        desktop.hf_file_var.get.return_value = "model-Q6_K.gguf"
        desktop._render_hf_download = mock.Mock()
        desktop._set_status = mock.Mock()

        CronoDesktop._start_hf_download(desktop)

        desktop.backend.start_download.assert_called_once_with(
            "owner/model-GGUF", "model-Q6_K.gguf"
        )
        desktop._render_hf_download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
