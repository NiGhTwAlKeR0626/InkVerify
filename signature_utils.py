"""Signature preprocessing and pair-wise feature extraction for InkVerify.

Normalised canvas: 256 x 128 pixels (width x height), ink bright (up to 255), background 0.
The aspect ratio of the writing is preserved (fit + centre), so the aspect and
centroid features stay meaningful.
"""
import cv2
import numpy as np

WIDTH, HEIGHT = 256, 128
SIZE = (WIDTH, HEIGHT)          # OpenCV order: (width, height)
PAD = 8
FEATURE_VERSION = 3             # bump when pair_features() changes -> old models are rejected
FEATURE_NAMES = [
    "hog_cosine", "hog_delta_mean", "hog_delta_std", "hog_delta_q90",
    "ink_ratio_diff", "aspect_diff", "centroid_dist",
    "hprofile_corr", "vprofile_corr", "thumb_corr",
]

HOG = cv2.HOGDescriptor(SIZE, (16, 16), (8, 8), (8, 8), 9)


def _normalise_background(gray):
    """Even out uneven lighting (typical for phone photos) by dividing by a paper estimate."""
    k = max(15, (min(gray.shape[:2]) // 8) | 1)
    paper = cv2.medianBlur(cv2.dilate(gray, np.ones((5, 5), np.uint8)), k)
    return cv2.divide(gray, paper, scale=255)


def _read_gray_on_white(path):
    """Read an image as grayscale. Transparent pixels count as white paper; JPEG/phone
    photos keep their EXIF rotation; very large photos are down-scaled for speed."""
    if str(path).lower().endswith((".jpg", ".jpeg")):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)        # honours EXIF orientation
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError("Cannot read image (corrupted or unsupported file).")
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        rgb = raw[:, :, :3].astype(np.float32)
        raw = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.dtype != np.uint8:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    side = max(raw.shape[:2])
    if side > 1600:
        f = 1600 / side
        raw = cv2.resize(raw, (int(raw.shape[1] * f), int(raw.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return raw


def load_signature(path):
    """Return a normalised signature image (HEIGHT x WIDTH, uint8, ink bright, background 0)."""
    gray = _read_gray_on_white(path)
    if min(gray.shape[:2]) < 16:
        raise ValueError("Image is too small to analyse.")
    gray = _normalise_background(cv2.GaussianBlur(gray, (3, 3), 0))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary > 0) > 0.5:            # dark background -> flip so ink is the minority
        binary = cv2.bitwise_not(binary)
    # remove isolated speckles (scan noise)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 30:
        raise ValueError("No signature ink found (image looks blank).")
    x, y, w, h = cv2.boundingRect(points)
    x0, y0 = max(0, x - PAD), max(0, y - PAD)
    x1, y1 = min(binary.shape[1], x + w + PAD), min(binary.shape[0], y + h + PAD)
    crop = binary[y0:y1, x0:x1]
    # fit into the canvas keeping aspect ratio, centred
    scale = min(WIDTH / crop.shape[1], HEIGHT / crop.shape[0])
    nw, nh = max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((HEIGHT, WIDTH), np.uint8)
    ox, oy = (WIDTH - nw) // 2, (HEIGHT - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def image_features(image):
    """Per-image descriptor bundle (compute once, reuse for every pair)."""
    hog = HOG.compute(image).reshape(-1).astype(np.float32)
    mask = image > 0
    ink_ratio = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs):
        aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
        centroid = (float(xs.mean() / WIDTH), float(ys.mean() / HEIGHT))
    else:
        aspect, centroid = 0.0, (0.5, 0.5)
    f = image.astype(np.float32) / 255.0
    hprof = cv2.resize(f.sum(axis=0).reshape(1, -1), (32, 1), interpolation=cv2.INTER_AREA).ravel()
    vprof = cv2.resize(f.sum(axis=1).reshape(1, -1), (32, 1), interpolation=cv2.INTER_AREA).ravel()
    thumb = cv2.resize(cv2.GaussianBlur(f, (5, 5), 0), (64, 32), interpolation=cv2.INTER_AREA).ravel()
    return {"hog": hog, "ink": ink_ratio, "aspect": aspect, "centroid": centroid,
            "hprof": hprof, "vprof": vprof, "thumb": thumb}


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def pair_features_from(fa, fb):
    """Pair feature vector from two precomputed descriptor bundles."""
    ha, hb = fa["hog"], fb["hog"]
    cosine = float(np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-8))
    delta = np.abs(ha - hb)
    cdist = float(np.hypot(fa["centroid"][0] - fb["centroid"][0], fa["centroid"][1] - fb["centroid"][1]))
    return np.array([
        cosine, delta.mean(), delta.std(), np.quantile(delta, .90),
        abs(fa["ink"] - fb["ink"]), abs(fa["aspect"] - fb["aspect"]), cdist,
        _corr(fa["hprof"], fb["hprof"]), _corr(fa["vprof"], fb["vprof"]),
        _corr(fa["thumb"], fb["thumb"]),
    ], dtype=np.float32)


def pair_features(image_a, image_b):
    """Convenience wrapper taking two normalised images."""
    return pair_features_from(image_features(image_a), image_features(image_b))


# ---------- evidence views for the result page ----------
def overlay_png_b64(known, questioned, scale=3):
    """Transparent PNG (base64) with the known signature in blue, the questioned one in
    vermilion and the overlap in grey. Works on light and dark page backgrounds."""
    import base64
    def mask(img):
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return big > 70                                    # smooth edges, keeps strokes thin

    k, q = mask(known), mask(questioned)
    out = np.zeros(k.shape + (4,), np.uint8)               # BGRA
    out[k] = (219, 111, 47, 255)
    out[q] = (45, 85, 226, 255)
    out[k & q] = (150, 150, 150, 255)
    ok, buf = cv2.imencode(".png", out)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def indicators(ref_feats, q_feat):
    """Plain similarity indicators (0-1) averaged over the known samples.
    They describe the pair; they are NOT the model's decision."""
    vecs = np.mean([pair_features_from(r, q_feat) for r in ref_feats], axis=0)
    v = dict(zip(FEATURE_NAMES, [float(x) for x in vecs]))
    clip = lambda x: max(0.0, min(1.0, x))
    return [
        ("Stroke structure", clip(v["hog_cosine"])),
        ("Horizontal layout", clip(v["hprofile_corr"])),
        ("Vertical layout", clip(v["vprofile_corr"])),
        ("Overall shape", clip(v["thumb_corr"])),
        ("Proportions", clip(1 - v["aspect_diff"])),
        ("Ink density", clip(1 - v["ink_ratio_diff"] * 8)),     # x8 = display scaling only
    ]
