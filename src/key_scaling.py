import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k == 1:
        return x.copy()
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), ker, mode="same")


def find_peaks_1d(signal: np.ndarray, min_dist: int, thr: float) -> List[int]:
    """
    Very simple local maxima peak picker.
    - signal: 1D float array
    - thr: absolute threshold
    - min_dist: minimum distance between accepted peaks
    """
    s = signal
    peaks = []
    n = len(s)
    for i in range(1, n - 1):
        if s[i] > thr and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            peaks.append(i)

    # Non-max suppression by min_dist
    peaks = sorted(peaks, key=lambda i: s[i], reverse=True)
    kept = []
    for p in peaks:
        if all(abs(p - q) >= min_dist for q in kept):
            kept.append(p)
    kept.sort()
    return kept


@dataclass
class Box:
    y1: int
    x1: int
    y2: int
    x2: int

    def to_dict(self):
        return {"y1": int(self.y1), "x1": int(self.x1), "y2": int(self.y2), "x2": int(self.x2)}


# -----------------------------
# Step 1: Detect keyboard band (top/bottom edges)
# -----------------------------
def detect_keyboard_band(gray: np.ndarray, outdir: str) -> Tuple[int, int]:
    """
    Detect the horizontal band containing the keyboard using vertical gradient (dy)
    and row-wise energy.
    Returns (y_top, y_bottom) in FULL IMAGE coordinates.
    """
    h, w = gray.shape[:2]

    # Light smoothing for stability
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    # dy highlights horizontal edges (top/bottom of keyboard are strong horizontal edges)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    row_energy = np.sum(np.abs(dy), axis=1)

    # Smooth the profile
    row_energy_s = moving_average(row_energy, k=max(15, h // 120))

    # Search region: middle-ish (keyboard usually not at extreme top or bottom)
    y_lo = int(0.15 * h)
    y_hi = int(0.85 * h)
    region = row_energy_s[y_lo:y_hi].copy()

    # Find strongest peaks in region
    # threshold: a fraction of max energy
    thr = 0.35 * float(np.max(region))
    peaks = find_peaks_1d(region, min_dist=max(20, h // 30), thr=thr)

    if len(peaks) < 2:
        # fallback: take top-2 maxima
        idxs = np.argsort(region)[-2:]
        peaks = sorted(list(idxs))

    # Convert to full coordinates
    peaks_full = [p + y_lo for p in peaks]

    # Choose y_top as smaller peak, y_bottom as larger peak,
    # but enforce reasonable separation (keyboard height is not tiny).
    peaks_full.sort()
    y1 = peaks_full[0]
    y2 = peaks_full[-1]

    if y2 - y1 < int(0.15 * h):
        # If too close, expand around max area in the middle half
        mid = int(0.5 * (y1 + y2))
        y1 = clamp(mid - int(0.12 * h), 0, h - 1)
        y2 = clamp(mid + int(0.12 * h), 0, h - 1)

    # Add padding
    pad = int(0.01 * h)
    y_top = clamp(y1 - pad, 0, h - 1)
    y_bot = clamp(y2 + pad, 0, h - 1)

    # Debug plot image
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(debug, (0, y_top), (w - 1, y_top), (0, 255, 255), 2)
    cv2.line(debug, (0, y_bot), (w - 1, y_bot), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(outdir, "debug_band_lines.png"), debug)

    return y_top, y_bot


# -----------------------------
# Step 2: Find where black keys end (so we detect white borders below)
# -----------------------------
def estimate_black_bottom(gray_band: np.ndarray, outdir: str) -> int:
    """
    Estimate the y row (within band coordinates) where black keys typically end.
    We look for a row where "dark pixel fraction" drops sharply.
    Returns y_black_bottom (within band coordinates).
    """
    hb, wb = gray_band.shape[:2]

    # Normalize contrast a bit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gb = clahe.apply(gray_band)

    # For each row, compute fraction of pixels below a dark threshold
    # (adaptive threshold would be more complex; fixed works after CLAHE)
    dark_thr = int(np.percentile(gb, 25))
    dark_mask = (gb < dark_thr).astype(np.uint8)

    frac = np.mean(dark_mask, axis=1)
    frac_s = moving_average(frac, k=max(9, hb // 40))

    # Typically black keys occupy upper part of band; find the first significant drop
    # Search in upper 60% of band
    y_search_hi = int(0.65 * hb)
    f = frac_s[:y_search_hi]

    # We pick y where frac becomes small after being large
    # threshold chosen relative to its max
    fmax = float(np.max(f)) + 1e-6
    target = 0.35 * fmax  # when it drops below this, black keys likely ended

    y_black = None
    for i in range(5, len(f)):
        if f[i] < target and f[i - 1] >= target:
            y_black = i
            break

    if y_black is None:
        y_black = int(0.40 * hb)

    # Debug
    dbg = cv2.cvtColor(gray_band, cv2.COLOR_GRAY2BGR)
    cv2.line(dbg, (0, y_black), (wb - 1, y_black), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(outdir, "debug_black_bottom.png"), dbg)

    return int(y_black)


# -----------------------------
# Step 3: Detect white key borders (vertical lines)
# -----------------------------
def detect_white_borders(gray_band: np.ndarray, y_black_bottom: int, outdir: str) -> List[int]:
    """
    Detect vertical borders of white keys using vertical gradient (dx) summed over
    the lower part of the keyboard band.
    Returns list of x positions in band coordinates.
    """
    hb, wb = gray_band.shape[:2]

    # Focus on lower part where white key separations are clean
    y1 = clamp(y_black_bottom + int(0.05 * hb), 0, hb - 1)
    y2 = clamp(hb - int(0.05 * hb), 0, hb)
    roi = gray_band[y1:y2, :]

    # Contrast normalize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_n = clahe.apply(roi)

    roi_n = cv2.GaussianBlur(roi_n, (3, 3), 0)
    dx = cv2.Sobel(roi_n, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(dx)

    # Column energy
    col_energy = np.sum(mag, axis=0)
    col_energy_s = moving_average(col_energy, k=max(7, wb // 200))

    # Threshold peaks
    thr = 0.35 * float(np.max(col_energy_s))
    # Minimum distance between borders: depends on resolution.
    # Rough: width / ~88 keys ~ 14 px for 1280 wide -> min_dist ~ 6–10
    min_dist = max(6, wb // 180)

    peaks = find_peaks_1d(col_energy_s, min_dist=min_dist, thr=thr)

    # Add edges to make intervals
    borders = [0] + peaks + [wb - 1]
    borders = sorted(set(borders))

    # Filter weird borders by spacing consistency
    diffs = np.diff(borders)
    if len(diffs) > 5:
        med = float(np.median(diffs))
        # Keep borders that give plausible spacing
        good = [borders[0]]
        for i in range(1, len(borders)):
            if (borders[i] - good[-1]) >= max(4, 0.35 * med):
                good.append(borders[i])
        borders = good

    # Debug visualization
    dbg = cv2.cvtColor(gray_band, cv2.COLOR_GRAY2BGR)
    for x in borders:
        cv2.line(dbg, (int(x), 0), (int(x), hb - 1), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(outdir, "debug_white_borders.png"), dbg)

    return [int(x) for x in borders]


# -----------------------------
# Step 4: Build white key boxes
# -----------------------------
def build_white_keys_from_borders(borders_x: List[int], y_top: int, y_bot: int) -> List[Box]:
    boxes = []
    for i in range(len(borders_x) - 1):
        x1 = borders_x[i]
        x2 = borders_x[i + 1]
        if x2 - x1 < 4:
            continue
        boxes.append(Box(y1=y_top, x1=x1, y2=y_bot, x2=x2))
    return boxes


# -----------------------------
# Step 5: Detect black keys via connected components
# -----------------------------
def detect_black_keys(gray_band: np.ndarray, y_black_bottom: int, outdir: str) -> List[Box]:
    """
    Detect black keys as dark connected components in upper band.
    Returns boxes in band coordinates.
    """
    hb, wb = gray_band.shape[:2]

    y1 = 0
    y2 = clamp(y_black_bottom, 1, hb)
    roi = gray_band[y1:y2, :]

    # Normalize + threshold for dark regions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_n = clahe.apply(roi)

    # Invert threshold: black keys become white blobs
    # Use Otsu to adapt to lighting
    _, bw = cv2.threshold(roi_n, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean small noise
    bw = cv2.medianBlur(bw, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    # CC
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 150:
            continue
        # Black keys: tall-ish, not extremely wide
        if h < 0.25 * (y2 - y1):
            continue
        if w > 0.12 * wb:
            continue
        # Keep within ROI
        boxes.append(Box(y1=y, x1=x, y2=y + h, x2=x + w))

    # Debug
    dbg = cv2.cvtColor(gray_band, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(dbg, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(outdir, "debug_black_keys.png"), dbg)

    return boxes


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Full-image keyboard key detector (from scratch)")
    # ap.add_argument("--image", required=True, help="Path to full image (no-hands frame preferred)")
    ap.add_argument("--outdir", default="outputs_full", help="Output folder")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    IMAGE_PATH = "outputs/key_steps/01_full_input.png"
    full = cv2.imread(IMAGE_PATH)

    if full is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    
    h, w = full.shape[:2]
    gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)

    # 1) keyboard band in full coords
    y_top_full, y_bot_full = detect_keyboard_band(gray, args.outdir)

    # Crop band (full width)
    band = gray[y_top_full:y_bot_full, :]
    cv2.imwrite(os.path.join(args.outdir, "debug_band_crop.png"), band)

    # 2) estimate black-bottom within band coords
    y_black_bottom = estimate_black_bottom(band, args.outdir)

    # 3) white borders in band coords
    borders_x = detect_white_borders(band, y_black_bottom, args.outdir)

    # 4) build white keys in band coords
    white_boxes_band = build_white_keys_from_borders(borders_x, 0, band.shape[0] - 1)

    # 5) black keys in band coords
    black_boxes_band = detect_black_keys(band, y_black_bottom, args.outdir)

    # Convert band coords -> full image coords by adding y_top_full
    white_full = [Box(b.y1 + y_top_full, b.x1, b.y2 + y_top_full, b.x2) for b in white_boxes_band]
    black_full = [Box(b.y1 + y_top_full, b.x1, b.y2 + y_top_full, b.x2) for b in black_boxes_band]

    # Save json (lists)
    save_json(os.path.join(args.outdir, "white_keys.json"), [b.to_dict() for b in white_full])
    save_json(os.path.join(args.outdir, "black_keys.json"), [b.to_dict() for b in black_full])

    # Final overlay
    overlay = full.copy()
    for b in white_full:
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
    for b in black_full:
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(args.outdir, "overlay.png"), overlay)

    print("Done.")
    print(f"- white keys: {len(white_full)} -> {os.path.join(args.outdir, 'white_keys.json')}")
    print(f"- black keys: {len(black_full)} -> {os.path.join(args.outdir, 'black_keys.json')}")
    print(f"- overlay: {os.path.join(args.outdir, 'overlay.png')}")


if __name__ == "__main__":
    main()
