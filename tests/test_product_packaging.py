import unittest
from pathlib import Path

from launch_model_ctk import CronoDesktop, DONATION_QR_PATH


ROOT = Path(__file__).resolve().parent.parent


class ProductPackagingTests(unittest.TestCase):
    def test_distribution_uses_apache_2_license(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("Apache License\nVersion 2.0"))
        self.assertIn("END OF TERMS AND CONDITIONS", license_text)

    def test_donation_qr_is_packaged_as_local_jpeg(self):
        self.assertEqual(
            DONATION_QR_PATH,
            ROOT / "web" / "static" / "assets" / "donation-qrcode.jpeg",
        )
        payload = DONATION_QR_PATH.read_bytes()
        self.assertGreater(len(payload), 10_000)
        self.assertTrue(payload.startswith(b"\xff\xd8\xff"))
        self.assertTrue(payload.endswith(b"\xff\xd9"))

    def test_both_interfaces_reference_the_local_qr(self):
        template = (ROOT / "web" / "templates" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("assets/donation-qrcode.jpeg", template)
        self.assertIn("AJUDE O DESENVOLVEDOR", template)
        self.assertTrue(callable(getattr(CronoDesktop, "_show_donation_qr", None)))

    def test_core_only_web_stream_does_not_render_removed_snn_partial(self):
        web_app = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
        self.assertNotIn('render_string("partials/snn.html"', web_app)
        self.assertNotIn('name="partials/snn.html"', web_app)

    def test_support_documentation_is_shipped(self):
        support = (ROOT / "docs" / "SUPPORT.md").read_text(encoding="utf-8")
        self.assertIn("Doações não alteram", support)
        self.assertIn("Apache License 2.0", support)


if __name__ == "__main__":
    unittest.main()
