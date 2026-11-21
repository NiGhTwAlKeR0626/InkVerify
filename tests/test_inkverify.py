"""Run from the project root:  python -m unittest discover -s tests -v"""
import sys, tempfile, unittest
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from signature_utils import (load_signature, image_features, pair_features, WIDTH, HEIGHT,
                             FEATURE_NAMES, overlay_png_b64, indicators)
from document_utils import compare_documents


def _write(path, img):
    cv2.imwrite(str(path), img)
    return path


def _sig_image():
    img = np.full((200, 400), 255, np.uint8)
    cv2.polylines(img, [np.array([[30, 150], [120, 40], [220, 160], [370, 60]], np.int32)], False, 0, 3)
    return img


class SignatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_unreadable_file_rejected(self):
        bad = self.tmp / "bad.png"; bad.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            load_signature(bad)

    def test_blank_image_rejected(self):
        p = _write(self.tmp / "blank.png", np.full((100, 200), 255, np.uint8))
        with self.assertRaises(ValueError):
            load_signature(p)

    def test_normalised_size(self):
        img = load_signature(_write(self.tmp / "s.png", _sig_image()))
        self.assertEqual(img.shape, (HEIGHT, WIDTH))        # 128 x 256 (H x W)
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreater(int(img.max()), 200)             # ink present (bright on black)
        self.assertEqual(int(img[0, 0]), 0)                 # background stays black

    def test_pair_feature_vector_length(self):
        img = load_signature(_write(self.tmp / "s.png", _sig_image()))
        v = pair_features(img, img)
        self.assertEqual(len(v), len(FEATURE_NAMES))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertAlmostEqual(float(v[0]), 1.0, places=3)   # identical -> cosine 1

    def test_dark_background_is_handled(self):
        inv = 255 - _sig_image()
        a = load_signature(_write(self.tmp / "n.png", _sig_image()))
        b = load_signature(_write(self.tmp / "i.png", inv))
        self.assertGreater(float(pair_features(a, b)[0]), 0.9)


class EvidenceTests(unittest.TestCase):
    def test_overlay_and_indicators(self):
        tmp = Path(tempfile.mkdtemp())
        img = load_signature(_write(tmp / "s.png", _sig_image()))
        self.assertGreater(len(overlay_png_b64(img, img)), 100)
        feats = image_features(img)
        inds = indicators([feats, feats], feats)
        self.assertEqual(len(inds), 6)
        self.assertTrue(all(0.0 <= v <= 1.0 for _, v in inds))
        self.assertGreater(min(v for _, v in inds), 0.9)        # identical -> all indicators near 1


class AppTests(unittest.TestCase):
    def test_pages_and_password_gate(self):
        import app as iv
        c = iv.app.test_client()
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(c.get("/uploads/anything.py").status_code, 404)
        iv.PASSWORD = "secret"
        try:
            self.assertEqual(c.get("/").status_code, 401)
            import base64
            ok = {"Authorization": "Basic " + base64.b64encode(b"u:secret").decode()}
            self.assertEqual(c.get("/", headers=ok).status_code, 200)
        finally:
            iv.PASSWORD = None


class DocumentTests(unittest.TestCase):
    def test_same_document_no_changes_and_valid_output(self):
        src = ROOT / "sample_inputs" / "document_original_synthetic.png"
        out = Path(tempfile.mkdtemp()) / "t"
        r = compare_documents(src, src, out)
        self.assertEqual(r["region_count"], 0)
        self.assertTrue(Path(r["files"]["diff"]).exists() and Path(r["files"]["ref"]).exists())

    def test_changed_document_detected(self):
        a = ROOT / "sample_inputs" / "document_original_synthetic.png"
        b = ROOT / "sample_inputs" / "document_questioned_synthetic.png"
        r = compare_documents(a, b, Path(tempfile.mkdtemp()) / "h")
        self.assertGreaterEqual(r["region_count"], 1)

    def test_unreadable_document_rejected(self):
        bad = Path(tempfile.mkdtemp()) / "x.png"; bad.write_bytes(b"junk")
        with self.assertRaises(ValueError):
            compare_documents(bad, bad, bad.parent / "b")


if __name__ == "__main__":
    unittest.main()
