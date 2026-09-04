import unittest

from launch_model_core import SPECULATIVE_TYPES
from launch_model_ctk import (
    CronoDesktop, adaptive_window_metrics, decoration_geometry_pulse,
    memory_guard_view, vram_plan_action,
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


if __name__ == "__main__":
    unittest.main()
