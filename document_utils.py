"""Visual difference screening for two scans of the same document."""
import cv2
import numpy as np

MAX_SIDE = 1600          # larger scans are down-scaled for speed
MIN_INLIERS = 15         # fewer RANSAC inliers -> alignment is considered weak
DIFF_THRESHOLD = 40      # on illumination-normalised difference (0-255)
MAX_REGIONS = 50


def _read_color(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Both document images must be readable.")
    h, w = img.shape[:2]
    if min(h, w) < 64:
        raise ValueError("Document image is too small.")
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _normalise_illumination(gray):
    """Divide by a blurred background estimate to suppress lighting/shadow gradients."""
    k = max(21, (min(gray.shape) // 20) | 1)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), k | 1)
    return cv2.divide(gray, background, scale=255)


def _align(ref_gray, qry_gray, qry_color):
    """Return (aligned_color, valid_mask, status, inliers)."""
    h, w = ref_gray.shape
    full = np.full((h, w), 255, np.uint8)
    orb = cv2.ORB_create(nfeatures=3000)
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(qry_gray, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return cv2.resize(qry_color, (w, h)), full, "no_keypoints", 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return cv2.resize(qry_color, (w, h)), full, "few_matches", len(good)
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(inl.sum()) if inl is not None else 0
    if matrix is None or inliers < MIN_INLIERS:
        return cv2.resize(qry_color, (w, h)), full, "weak_alignment", inliers
    # reject absurd transforms (extreme scale / flip)
    det = np.linalg.det(matrix[:2, :2])
    if not (0.5 < det < 2.0):
        return cv2.resize(qry_color, (w, h)), full, "implausible_transform", inliers
    aligned = cv2.warpPerspective(qry_color, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    valid = cv2.warpPerspective(full, matrix, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))   # ignore the warped border
    return aligned, valid, "aligned", inliers


def compare_documents(reference_path, questioned_path, out_prefix):
    """Writes <prefix>_ref.jpg, <prefix>_qry.jpg (aligned) and <prefix>_diff.jpg (heatmap + boxes)."""
    out_prefix = str(out_prefix)
    ref = _read_color(reference_path)
    qry = _read_color(questioned_path)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    qry_gray = cv2.cvtColor(qry, cv2.COLOR_BGR2GRAY)

    aligned, valid, status, inliers = _align(ref_gray, qry_gray, qry)

    g1 = _normalise_illumination(ref_gray)
    g2 = _normalise_illumination(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
    diff = cv2.GaussianBlur(cv2.absdiff(g1, g2), (5, 5), 0)
    diff = cv2.bitwise_and(diff, diff, mask=valid)

    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))     # drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))    # join nearby marks
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(80, int(0.00005 * ref.shape[0] * ref.shape[1]))
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:MAX_REGIONS]

    heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(aligned, 0.65, heat, 0.35, 0)
    for x, y, w, h in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    q = [cv2.IMWRITE_JPEG_QUALITY, 88]
    files = {"ref": out_prefix + "_ref.jpg", "qry": out_prefix + "_qry.jpg", "diff": out_prefix + "_diff.jpg"}
    cv2.imwrite(files["ref"], ref, q)
    cv2.imwrite(files["qry"], aligned, q)
    cv2.imwrite(files["diff"], overlay, q)

    changed_pct = round(100.0 * float(np.count_nonzero(mask)) / mask.size, 3)
    warnings = {
        "aligned": None,
        "no_keypoints": "Too few keypoints found - images were only resized, not aligned.",
        "few_matches": "Too few keypoint matches - images were only resized, not aligned.",
        "weak_alignment": "Alignment is weak - highlighted regions may be caused by misalignment.",
        "implausible_transform": "Alignment looked implausible and was skipped - results are unreliable.",
    }[status]
    return {
        "region_count": len(boxes), "regions": boxes, "files": files,
        "width": int(ref.shape[1]), "height": int(ref.shape[0]),
        "alignment": status, "inliers": inliers, "changed_percent": changed_pct,
        "warning": warnings,
        "note": "Visual differences require human review; scan noise may also be highlighted.",
    }
