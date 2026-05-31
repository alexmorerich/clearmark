#!/usr/bin/env python3
"""
Standalone Sunsky watermark pipeline.

This script only reads source assets. Cleaned images, masks, manifests, and
review HTML are written to this project unless --out is supplied.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/Users/alexkou/Documents/github/b2bweb/content/products/assets")
SOURCE_REPO = Path("/Users/alexkou/Documents/github/b2bweb").resolve()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SKIP_IPHONE_MIN = 14
IPHONE_NUM_RE = re.compile(r"iphone-(\d+)")

PRESETS = {
    "fast": {
        "downscale_max": 360,
        "scales": [0.35, 0.55, 0.75, 1.0, 1.3],
        "threshold": 0.36,
        "verify_min": 0.46,
        "max_detections": 4,
        "peaks_per_scale": 8,
    },
    "review": {
        "downscale_max": 520,
        "scales": [0.3, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45],
        "threshold": 0.34,
        "verify_min": 0.44,
        "max_detections": 6,
        "peaks_per_scale": 10,
    },
    "high": {
        "downscale_max": 900,
        "scales": [0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75],
        "threshold": 0.32,
        "verify_min": 0.42,
        "max_detections": 8,
        "peaks_per_scale": 12,
    },
}

MAX_MASK_AREA = 0.03
PILOT_MASK_AREA = 0.022
INPAINT_RADIUS = 5
SHARPNESS_MIN_RATIO = 0.30
TEXT_LIKENESS_MIN = 0.12
CLEAN_VISIBLE_RESIDUAL_MAX = 0.48
CLEAN_STRICT_RESIDUAL_MAX = 0.25
CLEAN_STRICT_TEMPLATE_MAX = 0.30
CLEAN_FAIL_RESIDUAL_MIN = 0.45
CLEAN_FAIL_TEXT_COMPONENTS = 4
LAMA_ESCALATION_RESIDUAL_MIN = 0.40
ARTIFACT_SHARPNESS_REVIEW_RATIO = 0.18
COMBINED_MASK_REVIEW_AREA = 0.035
MIN_TEXT_COMPONENTS = 4
HIGH_CONTRAST_SPAN = 200.0
REVIEW_CONTRAST_SPAN = 170.0
FALLBACK_CONTRAST_SPAN = 150.0
LINE_DOMINANCE_MAX = 0.72
OCR_WATERMARK_RE = re.compile(r"(sunsky|sunsk|sursky|sky.*onlin|onlin.*com|onlinecom|olne.*com|alinec|sun.*com)")
ENABLE_BRIGHT_RECALL = True
ENABLE_HIGH_CONTRAST_BOX_MASK = True
ENABLE_LAMA_ESCALATION = True
_CANONICAL_INK_MASK: np.ndarray | None = None
_SIMPLE_LAMA = None


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    verify_score: float
    template: str
    scale: float
    mark_box: dict
    mask_area_pct: float
    text_score: float
    text_components: int
    contrast_span: float = 0.0
    line_dominance: float = 0.0
    confidence: float = 0.0

    def to_json(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "score": round(self.score, 4),
            "verify_score": round(self.verify_score, 4),
            "template": self.template,
            "scale": round(self.scale, 3),
            "mark_box": self.mark_box,
            "mask_area_pct": round(self.mask_area_pct, 3),
            "text_score": round(self.text_score, 4),
            "text_components": self.text_components,
            "contrast_span": round(self.contrast_span, 3),
            "line_dominance": round(self.line_dominance, 3),
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    image: np.ndarray
    kind: str
    start: float = 0.0
    end: float = 1.0


def now_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def prepare_out_dir(out: Path | None, prefix: str) -> Path:
    out_dir = out.expanduser().resolve() if out else (OUTPUT_ROOT / now_run_id(prefix)).resolve()
    if is_inside(out_dir, SOURCE_REPO):
        raise SystemExit(f"Refusing to write output inside source repo: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def require_rights(args: argparse.Namespace) -> None:
    if not getattr(args, "rights_confirmed", False):
        raise SystemExit(
            "Cleaning is blocked until you add --rights-confirmed. "
            "Use it only for images you own or are authorized to modify."
        )


def should_scan_file(filename: str) -> bool:
    nums = [int(m) for m in IPHONE_NUM_RE.findall(filename.lower())]
    if not nums:
        return True
    return min(nums) < SKIP_IPHONE_MIN


def iter_images(assets: Path) -> list[Path]:
    return sorted(
        p for p in assets.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def bucket_for(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    if 0.95 <= ratio <= 1.05:
        aspect = "square"
    elif ratio > 1.6:
        aspect = "wide"
    elif ratio > 1.05:
        aspect = "landscape"
    elif ratio < 0.62:
        aspect = "tall"
    else:
        aspect = "portrait"
    return f"{width}x{height}_{aspect}"


def phash_image(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    size = hash_size * highfreq_factor
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    small = np.float32(small)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size].copy()
    vals = low.flatten()[1:]
    med = float(np.median(vals))
    bits = (low.flatten() > med).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


class BKNode:
    def __init__(self, value: int, key: str):
        self.value = value
        self.keys = [key]
        self.children: dict[int, "BKNode"] = {}


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None

    def add_or_find(self, value: int, key: str, threshold: int) -> str:
        if self.root is None:
            self.root = BKNode(value, key)
            return key
        node = self.root
        while True:
            dist = hamming(value, node.value)
            if dist <= threshold:
                node.keys.append(key)
                return node.keys[0]
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = BKNode(value, key)
                return key
            node = child


def image_meta(path: Path) -> dict | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    return {
        "file": path.name,
        "width": w,
        "height": h,
        "bucket": bucket_for(w, h),
        "phash": f"{phash_image(img):016x}",
        "should_scan": should_scan_file(path.name),
    }


def template_signature(gray: np.ndarray) -> dict:
    """Return cheap text-shape features for rejecting non-text templates."""
    if gray.shape[0] < 8 or gray.shape[1] < 24:
        return {"text_like": False, "components": 0, "runs": 0, "density": 0.0}
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, foreground = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = float(np.count_nonzero(foreground)) / max(1, foreground.size)
    active_cols = (foreground > 0).sum(axis=0) >= max(1, int(gray.shape[0] * 0.10))
    runs = 0
    in_run = False
    for active in active_cols:
        if active and not in_run:
            runs += 1
            in_run = True
        elif not active:
            in_run = False

    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    components = 0
    for idx in range(1, count):
        _, _, ww, hh, area = stats[idx]
        if 2 <= area <= gray.size * 0.20 and hh <= gray.shape[0] * 0.95 and ww <= gray.shape[1] * 0.45:
            components += 1
    text_like = runs >= 5 and components >= 5 and 0.03 <= density <= 0.45
    return {"text_like": text_like, "components": components, "runs": runs, "density": density}


def crop_template_to_ink(gray: np.ndarray, margin: int = 3) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, foreground = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(foreground > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(gray.shape[1], int(xs.max()) + margin + 1)
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(gray.shape[0], int(ys.max()) + margin + 1)
    return gray[y1:y2, x1:x2]


def canonical_ink_mask() -> np.ndarray:
    global _CANONICAL_INK_MASK
    if _CANONICAL_INK_MASK is not None:
        return _CANONICAL_INK_MASK
    tpl = cv2.imread(str(TEMPLATE_DIR / "watermark-template.png"), cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise SystemExit(f"Missing canonical watermark template: {TEMPLATE_DIR / 'watermark-template.png'}")
    cropped = crop_template_to_ink(tpl, margin=1)
    blur = cv2.GaussianBlur(cropped, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _CANONICAL_INK_MASK = ink
    return ink


def derived_text_templates(name: str, gray: np.ndarray) -> list[TemplateSpec]:
    """Build a small set of exact-domain text crops from a full watermark template."""
    base = crop_template_to_ink(gray)
    variants = [TemplateSpec(name=name, image=base, kind="full", start=0.0, end=1.0)]
    width = base.shape[1]
    if width >= 120:
        for suffix, start, end in (
            ("left85", 0.00, 0.85),
            ("right85", 0.15, 1.00),
            ("right70", 0.30, 1.00),
        ):
            x1 = int(round(width * start))
            x2 = int(round(width * end))
            crop = base[:, x1:x2]
            if crop.shape[1] >= 60:
                variants.append(TemplateSpec(
                    name=f"{name}:{suffix}",
                    image=crop,
                    kind="crop",
                    start=start,
                    end=end,
                ))
    return variants


def load_templates() -> list[TemplateSpec]:
    templates: list[TemplateSpec] = []
    fallback_templates: list[tuple[str, np.ndarray]] = []
    for p in sorted(TEMPLATE_DIR.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size:
            signature = template_signature(img)
            if signature["text_like"]:
                templates.extend(derived_text_templates(p.name, img))
            else:
                fallback_templates.append((p.name, img))
    if not templates:
        raise SystemExit(f"No text-like watermark templates found in {TEMPLATE_DIR}")
    # Real-image strips are too generic for primary detection: in pilots they
    # matched screws, flex-cable traces, phone-frame edges, and shadows. Keep
    # the canonical text templates as the only signal that can trigger masking.
    if fallback_templates:
        print(
            f"  ignored {len(fallback_templates)} non-text fallback template(s) for detection",
            file=sys.stderr,
        )
    return templates


def clamp_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> dict:
    w = max(1, min(int(round(w)), img_w))
    h = max(1, min(int(round(h)), img_h))
    x = int(round(max(0, min(x, img_w - w))))
    y = int(round(max(0, min(y, img_h - h))))
    return {"x": x, "y": y, "w": w, "h": h}


def refined_mark_box(raw: dict, img_w: int, img_h: int) -> dict:
    cx = raw["x"] + raw["w"] / 2
    cy = raw["y"] + raw["h"] / 2
    mark_h = max(raw["h"] * 1.45, 14)
    mark_h = min(mark_h, img_h * 0.065)
    mark_w = max(raw["w"] * 1.18, mark_h * 6.8)
    mark_w = min(mark_w, img_w * 0.42)
    return clamp_box(cx - mark_w / 2, cy - mark_h / 2, mark_w, mark_h, img_w, img_h)


def project_mark_box(raw: dict, template: TemplateSpec, img_w: int, img_h: int) -> dict:
    """Project a template hit back to the full sunsky-online.com text band.

    Partial crops such as right70 match only one side of the text. Centering a
    mask on that partial hit misses the real watermark position. Projection uses
    the crop offset inside the canonical full text to recover the full band.
    """
    span = max(0.15, template.end - template.start)
    full_w = raw["w"] / span
    x = raw["x"] - full_w * template.start
    y = raw["y"]
    pad_x = max(3.0, full_w * 0.035)
    pad_y = max(3.0, raw["h"] * 0.24)
    mark_w = min(full_w + 2 * pad_x, img_w * 0.46)
    mark_h = min(raw["h"] + 2 * pad_y, img_h * 0.070)
    return clamp_box(x - pad_x, y - pad_y, mark_w, mark_h, img_w, img_h)


def text_likeness(gray: np.ndarray, box: dict) -> tuple[float, int]:
    """Score whether a box looks like many small text glyphs, not a frame edge.

    This is deliberately cheap and conservative. It helps demote false positives
    from phone-frame borders, screws, and hard product edges that can correlate
    with the watermark template but do not look like a text band.
    """
    img_h, img_w = gray.shape[:2]
    x = max(0, int(box["x"]))
    y = max(0, int(box["y"]))
    w = max(1, int(box["w"]))
    h = max(1, int(box["h"]))
    roi = gray[y:min(img_h, y + h), x:min(img_w, x + w)]
    if roi.shape[0] < 8 or roi.shape[1] < 30:
        return 0.0, 0

    kernel = max(5, min(31, (roi.shape[0] // 2) * 2 + 1))
    background = cv2.medianBlur(roi, kernel)
    dev = cv2.absdiff(roi, background)
    threshold = max(3, float(np.percentile(dev, 85)))
    glyphs = (dev >= threshold).astype(np.uint8) * 255

    count, _, stats, _ = cv2.connectedComponentsWithStats(glyphs, 8)
    components = []
    for idx in range(1, count):
        xx, yy, ww, hh, area = stats[idx]
        if area < 2 or area > roi.size * 0.18:
            continue
        if hh < 2 or hh > roi.shape[0] * 0.95:
            continue
        if ww > roi.shape[1] * 0.45:
            continue
        components.append((xx, yy, ww, hh, area))

    if components:
        xs = []
        for xx, _, ww, _, _ in components:
            xs.extend([xx, xx + ww])
        coverage = (max(xs) - min(xs)) / max(roi.shape[1], 1)
    else:
        coverage = 0.0
    density = sum(area for *_, area in components) / max(roi.size, 1)
    score = (
        min(1.0, len(components) / 18.0) * 0.45
        + min(1.0, coverage) * 0.35
        + min(1.0, density * 20.0) * 0.20
    )
    return float(score), len(components)


def band_features(gray: np.ndarray, box: dict) -> dict:
    img_h, img_w = gray.shape[:2]
    x = max(0, int(box["x"]))
    y = max(0, int(box["y"]))
    w = max(1, min(int(box["w"]), img_w - x))
    h = max(1, min(int(box["h"]), img_h - y))
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return {"contrast_span": 0.0, "line_dominance": 1.0}

    contrast_span = float(np.percentile(roi, 95) - np.percentile(roi, 5))
    kernel = max(5, min(31, (roi.shape[0] // 2) * 2 + 1))
    background = cv2.medianBlur(roi, kernel)
    dev = cv2.absdiff(roi, background)
    threshold = max(3, float(np.percentile(dev, 82)))
    foreground = dev >= threshold
    active_cols = foreground.sum(axis=0) >= max(1, int(roi.shape[0] * 0.10))
    max_run = 0
    run = 0
    for active in active_cols:
        if active:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    line_dominance = max_run / max(1, roi.shape[1])
    return {"contrast_span": contrast_span, "line_dominance": float(line_dominance)}


def is_fallback_template(template_name: str) -> bool:
    return ":" not in template_name and not template_name.startswith("watermark-template")


def candidate_confidence(
    score: float,
    verify: float,
    text_score: float,
    text_components: int,
    contrast_span: float,
    line_dominance: float,
    template_name: str,
) -> float:
    if template_name.startswith("ocr:"):
        return min(1.0, 0.82 + score * 0.18)
    confidence = (
        verify * 0.42
        + score * 0.22
        + text_score * 0.24
        + min(1.0, text_components / 18.0) * 0.12
    )
    if template_name.startswith("watermark-template"):
        confidence += 0.04
    if is_fallback_template(template_name):
        confidence -= 0.04
        if contrast_span > FALLBACK_CONTRAST_SPAN:
            confidence -= 0.24
    if contrast_span > HIGH_CONTRAST_SPAN:
        confidence -= 0.40
    elif contrast_span > REVIEW_CONTRAST_SPAN:
        confidence -= 0.18
    if line_dominance > LINE_DOMINANCE_MAX:
        confidence -= 0.35
    if text_components < MIN_TEXT_COMPONENTS:
        confidence -= 0.24
    return float(max(0.0, min(1.0, confidence)))


def plausible_candidate(
    template_name: str,
    verify: float,
    text_score: float,
    text_components: int,
    contrast_span: float,
    line_dominance: float,
    confidence: float,
) -> bool:
    if template_name.startswith("ocr:"):
        return True
    if text_components < MIN_TEXT_COMPONENTS and verify < 0.68:
        return False
    if line_dominance > LINE_DOMINANCE_MAX and text_components < 12:
        return False
    if contrast_span > HIGH_CONTRAST_SPAN:
        return False
    if contrast_span > REVIEW_CONTRAST_SPAN:
        return False
    if is_fallback_template(template_name) and confidence < 0.56:
        return False
    if is_fallback_template(template_name) and contrast_span > FALLBACK_CONTRAST_SPAN:
        return False
    if is_fallback_template(template_name) and line_dominance > 0.58 and text_components < 8:
        return False
    if ":" in template_name and confidence < 0.58:
        return False
    return confidence >= 0.52


def full_text_templates(templates: list[TemplateSpec]) -> list[TemplateSpec]:
    full = [tpl for tpl in templates if tpl.kind == "full"]
    return full or templates


def verify_candidate(gray: np.ndarray, box: dict, templates: list[TemplateSpec]) -> float:
    img_h, img_w = gray.shape[:2]
    pad_x = max(12, box["w"] // 8)
    pad_y = max(8, box["h"] // 2)
    x1 = max(0, box["x"] - pad_x)
    y1 = max(0, box["y"] - pad_y)
    x2 = min(img_w, box["x"] + box["w"] + pad_x)
    y2 = min(img_h, box["y"] + box["h"] + pad_y)
    roi = gray[y1:y2, x1:x2]
    if roi.shape[0] < 10 or roi.shape[1] < 40:
        return 0.0

    best = 0.0
    target_h = max(int(round(box["h"] / 1.48)), 6)
    for spec in full_text_templates(templates):
        tpl = spec.image
        th, tw = tpl.shape[:2]
        for factor in (0.82, 0.94, 1.0, 1.08, 1.20):
            scale = (target_h / max(th, 1)) * factor
            sw, sh = int(tw * scale), int(th * scale)
            if sw < 24 or sh < 6 or sw >= roi.shape[1] or sh >= roi.shape[0]:
                continue
            resized = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            best = max(best, score)
            if best >= 0.78:
                return best
    return best


def detection_rank(det: Detection, img_w: int, img_h: int) -> float:
    if det.template.startswith("ocr:"):
        return 10.0 + det.confidence + det.verify_score
    b = det.mark_box
    cx = b["x"] + b["w"] / 2
    cy = b["y"] + b["h"] / 2
    aspect = b["w"] / max(b["h"], 1)
    aspect_bonus = 0.25 if 5.0 <= aspect <= 10.0 else -0.2
    edge_penalty = 0.4 if (b["x"] <= 1 or b["y"] <= 1 or b["x"] + b["w"] >= img_w - 1 or b["y"] + b["h"] >= img_h - 1) else 0
    extreme_y_penalty = 0.85 if (cy < img_h * 0.10 or cy > img_h * 0.90) else 0.0
    center_penalty = 0.12 * abs(cx - img_w / 2) / img_w + 0.08 * abs(cy - img_h / 2) / img_h
    area_penalty = 0.25 if det.mask_area_pct > 4.0 else 0
    text_bonus = min(0.7, det.text_score * 0.7)
    contrast_penalty = 0.28 if det.contrast_span > REVIEW_CONTRAST_SPAN else 0.0
    line_penalty = 0.22 if det.line_dominance > LINE_DOMINANCE_MAX else 0.0
    rank = (
        det.score + det.verify_score + det.confidence + aspect_bonus + text_bonus
        - edge_penalty - extreme_y_penalty - center_penalty - area_penalty
        - contrast_penalty - line_penalty
    )
    if det.template.startswith("prior:"):
        rank -= 0.75
    return rank


def nms(detections: list[Detection], img_w: int, img_h: int, limit: int) -> list[Detection]:
    def iou(a: dict, b: dict) -> float:
        ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
        bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
        ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union else 0.0

    ordered = sorted(detections, key=lambda d: detection_rank(d, img_w, img_h), reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        if all(iou(det.mark_box, old.mark_box) < 0.25 for old in kept):
            kept.append(det)
        if len(kept) >= limit:
            break
    return kept


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ocr_text_matches_watermark(text: str) -> bool:
    norm = normalize_ocr_text(text)
    return bool(OCR_WATERMARK_RE.search(norm))


def load_ocr_reader(enabled: bool):
    if not enabled:
        return None
    try:
        import easyocr  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        raise SystemExit(f"--ocr requested but easyocr is unavailable: {exc}") from exc
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def ocr_watermark_detections(img: np.ndarray, gray: np.ndarray, reader) -> list[Detection]:
    if reader is None:
        return []
    img_h, img_w = gray.shape[:2]
    detections: list[Detection] = []
    try:
        results = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            text_threshold=0.20,
            low_text=0.05,
            link_threshold=0.10,
            canvas_size=1280,
            mag_ratio=2.0,
        )
    except Exception as exc:
        print(f"  warning: OCR failed: {type(exc).__name__}", file=sys.stderr)
        return []

    for box, text, conf in results:
        if not ocr_text_matches_watermark(str(text)):
            continue
        pts = np.array(box, dtype=np.float32)
        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))
        pad_x = max(4.0, (x2 - x1) * 0.03)
        pad_y = max(3.0, (y2 - y1) * 0.10)
        mark = clamp_box(x1 - pad_x, y1 - pad_y, (x2 - x1) + 2 * pad_x, (y2 - y1) + 2 * pad_y, img_w, img_h)
        area_pct = 100.0 * mark["w"] * mark["h"] / max(1, img_w * img_h)
        if area_pct > 100 * MAX_MASK_AREA:
            continue
        text_score, text_components = text_likeness(gray, mark)
        features = band_features(gray, mark)
        ocr_score = max(0.75, float(conf))
        confidence = candidate_confidence(
            ocr_score,
            1.0,
            max(text_score, 0.95),
            max(text_components, 12),
            features["contrast_span"],
            features["line_dominance"],
            f"ocr:{text}",
        )
        detections.append(Detection(
            x=mark["x"], y=mark["y"], w=mark["w"], h=mark["h"],
            score=ocr_score, verify_score=1.0,
            template=f"ocr:{str(text)[:48]}", scale=1.0,
            mark_box=mark, mask_area_pct=area_pct,
            text_score=max(text_score, 0.95), text_components=max(text_components, 12),
            contrast_span=features["contrast_span"],
            line_dominance=features["line_dominance"],
            confidence=confidence,
        ))
    return detections


def ocr_image_watermark_check(img: np.ndarray, reader, *, canvas_size: int = 960, mag_ratio: float = 2.5) -> dict:
    if reader is None or img.size == 0:
        return {"ocr_checked": False, "ocr_watermark": None, "ocr_text": []}
    try:
        results = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            text_threshold=0.16,
            low_text=0.04,
            link_threshold=0.08,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
        )
    except Exception as exc:
        return {
            "ocr_checked": False,
            "ocr_watermark": None,
            "ocr_text": [],
            "ocr_error": type(exc).__name__,
        }
    texts: list[str] = []
    watermark = False
    for _, text, conf in results:
        text = str(text)
        if not text.strip():
            continue
        texts.append(f"{text}:{float(conf):.2f}")
        if ocr_text_matches_watermark(text):
            watermark = True
    return {
        "ocr_checked": True,
        "ocr_watermark": watermark,
        "ocr_text": texts[:6],
    }


def bright_text_likeness(gray: np.ndarray, box: dict) -> tuple[float, int, float]:
    """Score faint grey text on white or bright backgrounds.

    The regular text_likeness path is tuned for ordinary contrast. This variant
    boosts only darker-than-background structure, which catches faint Sunsky text
    on white areas without treating bright product highlights as glyphs.
    """
    img_h, img_w = gray.shape[:2]
    x = max(0, int(box["x"]))
    y = max(0, int(box["y"]))
    w = max(1, min(int(box["w"]), img_w - x))
    h = max(1, min(int(box["h"]), img_h - y))
    roi = gray[y:y + h, x:x + w]
    if roi.shape[0] < 8 or roi.shape[1] < 40:
        return 0.0, 0, 0.0

    bright_ratio = float(np.mean(roi >= 190))
    if bright_ratio < 0.42 or float(np.mean(roi)) < 170.0:
        return 0.0, 0, bright_ratio

    kernel = max(5, min(35, (roi.shape[0] // 2) * 2 + 1))
    background = cv2.medianBlur(roi, kernel)
    dark_dev = cv2.subtract(background, roi)
    if float(np.percentile(dark_dev, 96)) < 2.0:
        return 0.0, 0, bright_ratio

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    boosted = clahe.apply(dark_dev)
    threshold = max(7.0, float(np.percentile(boosted, 86)))
    glyphs = (boosted >= threshold).astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(glyphs, 8)
    components = []
    for idx in range(1, count):
        xx, yy, ww, hh, area = stats[idx]
        if area < 2 or area > roi.size * 0.12:
            continue
        if hh < 2 or hh > roi.shape[0] * 0.90:
            continue
        if ww > roi.shape[1] * 0.38:
            continue
        components.append((xx, yy, ww, hh, area))

    if components:
        xs = []
        for xx, _, ww, _, _ in components:
            xs.extend([xx, xx + ww])
        coverage = (max(xs) - min(xs)) / max(roi.shape[1], 1)
    else:
        coverage = 0.0
    density = sum(area for *_, area in components) / max(roi.size, 1)
    darkness = min(1.0, float(np.percentile(dark_dev, 97)) / 18.0)
    score = (
        min(1.0, len(components) / 14.0) * 0.42
        + min(1.0, coverage) * 0.34
        + min(1.0, density * 28.0) * 0.14
        + darkness * 0.10
    )
    return float(score), len(components), bright_ratio


def bright_background_text_band_detections(
    gray: np.ndarray,
    img: np.ndarray | None = None,
    ocr_reader=None,
) -> list[Detection]:
    """Recall pass for faint Sunsky marks over bright/white backgrounds."""
    img_h, img_w = gray.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img is not None else None
    detections: list[Detection] = []
    width_fracs = (0.28, 0.32, 0.36, 0.40, 0.44)
    x_fracs = (0.18, 0.24, 0.30, 0.36, 0.42)
    y_fracs = (0.34, 0.42, 0.50, 0.58, 0.66)
    for wf in width_fracs:
        bw = int(round(img_w * wf))
        if bw < 85:
            continue
        for aspect in (7.0, 8.2, 9.4):
            bh = int(round(max(13, bw / aspect)))
            if bh < 11 or bh > img_h * 0.075:
                continue
            for xf in x_fracs:
                x = int(round(img_w * xf))
                if x + bw > img_w:
                    continue
                for yf in y_fracs:
                    y = int(round(img_h * yf - bh / 2))
                    box = clamp_box(x, y, bw, bh, img_w, img_h)
                    score, components, bright_ratio = bright_text_likeness(gray, box)
                    if score < 0.70 or components < 12:
                        continue
                    roi = gray[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
                    if roi.size and float(np.mean(roi < 120)) > 0.38:
                        continue
                    regular_score, regular_components = text_likeness(gray, box)
                    features = band_features(gray, box)
                    if features["contrast_span"] > 235.0 or features["line_dominance"] > 0.55:
                        continue
                    if features["contrast_span"] > 165.0 and (score < 0.92 or features["line_dominance"] > 0.32):
                        continue
                    if features["contrast_span"] > 165.0 and (regular_score < 0.72 or regular_components < 10):
                        continue
                    if img is not None and ocr_reader is not None:
                        pad_x = max(8, int(round(box["w"] * 0.12)))
                        pad_y = max(6, int(round(box["h"] * 0.75)))
                        x1 = max(0, box["x"] - pad_x)
                        y1 = max(0, box["y"] - pad_y)
                        x2 = min(img_w, box["x"] + box["w"] + pad_x)
                        y2 = min(img_h, box["y"] + box["h"] + pad_y)
                        ocr_meta = ocr_image_watermark_check(img[y1:y2, x1:x2], ocr_reader)
                        if (
                            ocr_meta.get("ocr_checked")
                            and ocr_meta.get("ocr_text")
                            and not ocr_meta.get("ocr_watermark")
                        ):
                            continue
                    elif features["contrast_span"] > 165.0 and regular_components < 22:
                        continue
                    if hsv is not None:
                        sat_roi = hsv[
                            box["y"]:box["y"] + box["h"],
                            box["x"]:box["x"] + box["w"],
                            1,
                        ]
                        if float(np.mean(sat_roi > 70)) > 0.18:
                            continue
                    area_pct = 100.0 * box["w"] * box["h"] / max(1, img_w * img_h)
                    if area_pct > 100 * MAX_MASK_AREA:
                        continue
                    cx = box["x"] + box["w"] / 2
                    cy = box["y"] + box["h"] / 2
                    center_bias = 1.0 - min(1.0, (
                        abs(cx - img_w * 0.50) / max(img_w * 0.50, 1)
                        + abs(cy - img_h * 0.54) / max(img_h * 0.54, 1)
                    ) / 2)
                    confidence = (
                        score * 0.58
                        + min(1.0, components / 16.0) * 0.16
                        + bright_ratio * 0.10
                        + center_bias * 0.16
                    )
                    detections.append(Detection(
                        x=box["x"], y=box["y"], w=box["w"], h=box["h"],
                        score=float(confidence),
                        verify_score=max(0.42, score * 0.38 + center_bias * 0.22),
                        template="prior:bright_text_band",
                        scale=1.0,
                        mark_box=box,
                        mask_area_pct=area_pct,
                        text_score=max(score, regular_score),
                        text_components=max(components, regular_components),
                        contrast_span=features["contrast_span"],
                        line_dominance=features["line_dominance"],
                        confidence=float(min(0.86, confidence)),
                    ))
    return detections


def prior_text_band_detections(gray: np.ndarray, img: np.ndarray | None = None) -> list[Detection]:
    """Find faint center-body text bands when template correlation is weak.

    Sunsky marks are low-contrast horizontal text, usually placed in the image
    body. Product edges that fooled template matching tend to have much higher
    local contrast. This pass deliberately ignores high-contrast texture.
    """
    img_h, img_w = gray.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img is not None else None
    detections: list[Detection] = []
    width_fracs = (0.22, 0.28, 0.34, 0.40)
    x_fracs = (0.20, 0.28, 0.36, 0.44)
    y_fracs = (0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78)
    for wf in width_fracs:
        bw = int(round(img_w * wf))
        if bw < 90:
            continue
        for aspect in (7.2, 8.4, 9.4):
            bh = int(round(max(14, bw / aspect)))
            if bh < 12 or bh > img_h * 0.075:
                continue
            for xf in x_fracs:
                x = int(round(img_w * xf))
                if x + bw > img_w:
                    continue
                for yf in y_fracs:
                    y = int(round(img_h * yf - bh / 2))
                    box = clamp_box(x, y, bw, bh, img_w, img_h)
                    text_score, text_components = text_likeness(gray, box)
                    if text_score < 0.78 or text_components < 10:
                        continue
                    features = band_features(gray, box)
                    contrast_span = features["contrast_span"]
                    line_dominance = features["line_dominance"]
                    if contrast_span > 95.0 or line_dominance > 0.48:
                        continue
                    if hsv is not None:
                        sat_roi = hsv[
                            box["y"]:box["y"] + box["h"],
                            box["x"]:box["x"] + box["w"],
                            1,
                        ]
                        if float(np.mean(sat_roi > 58)) > 0.20:
                            continue
                    area_pct = 100.0 * box["w"] * box["h"] / max(1, img_w * img_h)
                    if area_pct > 100 * MAX_MASK_AREA:
                        continue
                    cx = box["x"] + box["w"] / 2
                    cy = box["y"] + box["h"] / 2
                    center_bias = 1.0 - min(1.0, (
                        abs(cx - img_w * 0.50) / max(img_w * 0.50, 1)
                        + abs(cy - img_h * 0.54) / max(img_h * 0.54, 1)
                    ) / 2)
                    contrast_bonus = max(0.0, 1.0 - contrast_span / 95.0)
                    confidence = (
                        text_score * 0.52
                        + min(1.0, text_components / 22.0) * 0.18
                        + center_bias * 0.18
                        + contrast_bonus * 0.12
                    )
                    detections.append(Detection(
                        x=box["x"], y=box["y"], w=box["w"], h=box["h"],
                        score=float(confidence),
                        verify_score=max(0.44, contrast_bonus * 0.36 + center_bias * 0.24),
                        template="prior:text_band",
                        scale=1.0,
                        mark_box=box,
                        mask_area_pct=area_pct,
                        text_score=text_score,
                        text_components=text_components,
                        contrast_span=contrast_span,
                        line_dominance=line_dominance,
                        confidence=float(min(0.88, confidence)),
                    ))
    return detections


def detect_watermark(
    gray: np.ndarray,
    templates: list[TemplateSpec],
    preset_name: str,
    img: np.ndarray | None = None,
    ocr_reader=None,
) -> list[Detection]:
    preset = PRESETS[preset_name]
    img_h, img_w = gray.shape[:2]
    down = 1.0
    scan = gray
    if max(img_w, img_h) > preset["downscale_max"]:
        down = preset["downscale_max"] / max(img_w, img_h)
        scan = cv2.resize(gray, (int(img_w * down), int(img_h * down)), interpolation=cv2.INTER_AREA)
    scale_back = 1.0 / down
    scan_h, scan_w = scan.shape[:2]

    found: list[Detection] = []
    if img is not None and ocr_reader is not None:
        found.extend(ocr_watermark_detections(img, gray, ocr_reader))
    found.extend(prior_text_band_detections(gray, img=img))
    for spec in templates:
        tpl_name = spec.name
        tpl = spec.image
        th, tw = tpl.shape[:2]
        for scale in preset["scales"]:
            sw, sh = int(tw * scale), int(th * scale)
            if sw < 24 or sh < 6 or sw >= scan_w or sh >= scan_h:
                continue
            resized = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(scan, resized, cv2.TM_CCOEFF_NORMED)
            maxima = cv2.dilate(res, np.ones((9, 9), np.uint8))
            ys, xs = np.where((res == maxima) & (res >= preset["threshold"]))
            if len(xs) == 0:
                continue
            values = res[ys, xs]
            order = np.argsort(values)[::-1][: int(preset["peaks_per_scale"])]
            for idx in order:
                x = int(xs[idx])
                y = int(ys[idx])
                score = float(values[idx])
                raw = {
                    "x": int(round(x * scale_back)),
                    "y": int(round(y * scale_back)),
                    "w": int(round(sw * scale_back)),
                    "h": int(round(sh * scale_back)),
                }
                mark = project_mark_box(raw, spec, img_w, img_h)
                area_pct = 100.0 * mark["w"] * mark["h"] / max(1, img_w * img_h)
                if area_pct > 100 * MAX_MASK_AREA:
                    continue
                verify = verify_candidate(gray, mark, templates)
                if verify < preset["verify_min"]:
                    continue
                text_score, text_components = text_likeness(gray, mark)
                features = band_features(gray, mark)
                confidence = candidate_confidence(
                    score,
                    verify,
                    text_score,
                    text_components,
                    features["contrast_span"],
                    features["line_dominance"],
                    tpl_name,
                )
                if text_score < TEXT_LIKENESS_MIN and verify < preset["verify_min"] + 0.18:
                    continue
                if not plausible_candidate(
                    tpl_name,
                    verify,
                    text_score,
                    text_components,
                    features["contrast_span"],
                    features["line_dominance"],
                    confidence,
                ):
                    continue
                found.append(Detection(
                    x=raw["x"], y=raw["y"], w=raw["w"], h=raw["h"],
                    score=score, verify_score=verify,
                    template=tpl_name, scale=scale, mark_box=mark,
                    mask_area_pct=area_pct, text_score=text_score,
                    text_components=text_components,
                    contrast_span=features["contrast_span"],
                    line_dominance=features["line_dominance"],
                    confidence=confidence,
                ))
    if ENABLE_BRIGHT_RECALL and not found and ocr_reader is not None:
        found.extend(bright_background_text_band_detections(gray, img=img, ocr_reader=ocr_reader))
    return nms(found, img_w, img_h, int(preset["max_detections"]))


def create_mask(
    gray: np.ndarray,
    det: Detection,
    img: np.ndarray | None = None,
    pad_x: int = 8,
    pad_y: int = 5,
    dilate_px: int = 4,
    glyph: bool = False,
) -> tuple[np.ndarray | None, float]:
    img_h, img_w = gray.shape[:2]
    b = det.mark_box
    x1 = max(0, b["x"] - pad_x)
    y1 = max(0, b["y"] - pad_y)
    x2 = min(img_w, b["x"] + b["w"] + pad_x)
    y2 = min(img_h, b["y"] + b["h"] + pad_y)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if glyph:
        if det.template.startswith("ocr:"):
            ink = canonical_ink_mask()
            ih, iw = ink.shape[:2]
            target_w = max(24, int(round(b["w"] * 0.94)))
            target_h = max(8, int(round(target_w * ih / max(iw, 1))))
            if target_h > b["h"] * 0.72:
                target_h = max(8, int(round(b["h"] * 0.72)))
                target_w = max(24, int(round(target_h * iw / max(ih, 1))))
            target_w = min(target_w, img_w)
            target_h = min(target_h, img_h)
            resized = cv2.resize(ink, (target_w, target_h), interpolation=cv2.INTER_AREA)
            _, resized = cv2.threshold(resized, 20, 255, cv2.THRESH_BINARY)
            tx = int(round(b["x"] + (b["w"] - target_w) / 2))
            ty = int(round(b["y"] + (b["h"] - target_h) / 2))
            tx = max(0, min(tx, img_w - target_w))
            ty = max(0, min(ty, img_h - target_h))
            mask[ty:ty + target_h, tx:tx + target_w] = resized
        else:
            roi = gray[y1:y2, x1:x2]
            kernel = max(5, min(35, (max(3, b["h"]) // 2) * 2 + 1))
            background = cv2.medianBlur(roi, kernel)
            dev = cv2.absdiff(roi, background)
            threshold = max(3, float(np.percentile(dev, 76)))
            glyph_mask = (dev >= threshold).astype(np.uint8) * 255
            count, labels, stats, _ = cv2.connectedComponentsWithStats(glyph_mask, 8)
            filtered = np.zeros_like(glyph_mask)
            for idx in range(1, count):
                _, _, ww, hh, area = stats[idx]
                if area < 2 or area > roi.size * 0.20:
                    continue
                if hh < 2 or hh > roi.shape[0] * 0.95:
                    continue
                if ww > roi.shape[1] * 0.55:
                    continue
                filtered[labels == idx] = 255
            if np.count_nonzero(filtered) < max(20, int(roi.size * 0.01)):
                return None, 0.0
            mask[y1:y2, x1:x2] = filtered
    else:
        mask[y1:y2, x1:x2] = 255
    if dilate_px:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    area = float(np.count_nonzero(mask)) / mask.size
    if area > MAX_MASK_AREA:
        return None, area
    return mask, area


def residual_score(gray: np.ndarray, det: Detection, templates: list[TemplateSpec] | None = None) -> float:
    templates = templates or load_templates()
    return verify_candidate(gray, det.mark_box, templates)


def residual_visibility_score(
    template_residual: float,
    post_text_score: float,
    post_text_components: int,
) -> float:
    """Estimate visible watermark residue after cleaning.

    Template correlation alone is noisy after inpaint because product edges can
    still resemble the watermark strip. The approval gate therefore requires
    both a local template match and remaining text-like structure.
    """
    component_score = min(1.0, post_text_components / 8.0)
    if post_text_components < 3:
        text_score = post_text_score * (post_text_components / 3.0)
    else:
        text_score = post_text_score
    visible = (
        min(1.0, template_residual) * 0.45
        + min(1.0, text_score) * 0.35
        + component_score * 0.20
    )
    return float(max(0.0, min(1.0, visible)))


def residual_quality_metrics(
    gray: np.ndarray,
    det: Detection,
    templates: list[TemplateSpec] | None = None,
) -> dict:
    template_residual = residual_score(gray, det, templates)
    post_text_score, post_text_components = text_likeness(gray, det.mark_box)
    features = band_features(gray, det.mark_box)
    visible_residual = residual_visibility_score(
        template_residual,
        post_text_score,
        post_text_components,
    )
    return {
        "template_residual_score": template_residual,
        "residual_score": visible_residual,
        "post_text_score": post_text_score,
        "post_text_components": post_text_components,
        "post_contrast_span": features["contrast_span"],
        "post_line_dominance": features["line_dominance"],
    }


def laplacian_var(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None and np.count_nonzero(mask):
        ys, xs = np.where(mask > 0)
        y1, y2 = max(0, ys.min() - 10), min(gray.shape[0], ys.max() + 11)
        x1, x2 = max(0, xs.min() - 10), min(gray.shape[1], xs.max() + 11)
        gray = gray[y1:y2, x1:x2]
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def padded_crop(img: np.ndarray, box: dict, pad_x: int, pad_y: int) -> np.ndarray:
    img_h, img_w = img.shape[:2]
    x1 = max(0, int(box["x"]) - pad_x)
    y1 = max(0, int(box["y"]) - pad_y)
    x2 = min(img_w, int(box["x"] + box["w"]) + pad_x)
    y2 = min(img_h, int(box["y"] + box["h"]) + pad_y)
    return img[y1:y2, x1:x2]


def cleaned_crop_ocr_check(img: np.ndarray, det: Detection, reader) -> dict:
    """Run OCR on the cleaned mark-box crop only."""
    pad_x = max(8, int(round(det.mark_box["w"] * 0.12)))
    pad_y = max(6, int(round(det.mark_box["h"] * 0.75)))
    crop = padded_crop(img, det.mark_box, pad_x, pad_y)
    return ocr_image_watermark_check(crop, reader)


def post_clean_detection_count(
    img: np.ndarray,
    gray: np.ndarray,
    templates: list[TemplateSpec] | None,
) -> int | None:
    if templates is None:
        return None
    try:
        return len(detect_watermark(gray, templates, "review", img=img, ocr_reader=None))
    except Exception:
        return None


def run_lama_escalation(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Optional LaMa/IOPaint backend. It is used only when installed locally."""
    global _SIMPLE_LAMA
    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore
    except Exception:
        SimpleLama = None  # type: ignore
    if SimpleLama is not None:
        try:
            if _SIMPLE_LAMA is None:
                _SIMPLE_LAMA = SimpleLama()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = _SIMPLE_LAMA(rgb, mask)
            arr = np.array(result)
            if arr.shape[:2] != img.shape[:2]:
                arr = arr[:img.shape[0], :img.shape[1]]
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), "simple_lama"
        except Exception as exc:
            return None, f"simple_lama:{type(exc).__name__}"

    command = os.environ.get("CLEARMARK_IOPAINT_CMD") or shutil.which("iopaint")
    if not command:
        return None, "unavailable"
    with tempfile.TemporaryDirectory(prefix="clearmark-iopaint-") as tmp:
        tmp_dir = Path(tmp)
        image_path = tmp_dir / "image.png"
        mask_path = tmp_dir / "mask.png"
        output_path = tmp_dir / "output.png"
        cv2.imwrite(str(image_path), img)
        cv2.imwrite(str(mask_path), mask)
        cmd = [
            command,
            "run",
            "--model",
            "lama",
            "--device",
            "cpu",
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        except Exception as exc:
            return None, type(exc).__name__
        result = cv2.imread(str(output_path))
        if result is None or result.shape[:2] != img.shape[:2]:
            return None, "invalid_output"
        return result, "ok"


def clean_image(
    img: np.ndarray,
    gray: np.ndarray,
    det: Detection,
    templates: list[TemplateSpec] | None = None,
    ocr_reader=None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    high_contrast_mask = ENABLE_HIGH_CONTRAST_BOX_MASK and det.contrast_span >= REVIEW_CONTRAST_SPAN
    if high_contrast_mask:
        variants = [
            ("edge_box_tight", 8, 5, 5, 5, False),
            ("edge_box_medium", 12, 7, 7, 7, False),
            ("glyph_medium", 12, 7, 4, 4, True),
            ("glyph_strong", 14, 8, 6, 7, True),
        ]
    else:
        variants = [
            ("glyph_tight", 8, 5, 3, 3, True),
            ("glyph_medium", 12, 7, 4, 4, True),
            ("glyph_strong", 14, 8, 6, 7, True),
            ("tight", 8, 5, 4, 4, False),
            ("medium", 12, 7, 5, 5, False),
        ]
    candidates = []
    first_mask = None
    first_area = 0.0

    for variant, pad_x, pad_y, dilate_px, radius, glyph in variants:
        mask, area = create_mask(gray, det, img=img, pad_x=pad_x, pad_y=pad_y, dilate_px=dilate_px, glyph=glyph)
        if mask is None:
            continue
        if first_mask is None:
            first_mask = mask
            first_area = area
        area_limit = MAX_MASK_AREA if high_contrast_mask and not glyph else PILOT_MASK_AREA
        if area > area_limit:
            continue
        for method_name, method in (("telea", cv2.INPAINT_TELEA), ("ns", cv2.INPAINT_NS)):
            candidate = cv2.inpaint(img, mask, radius, method)
            cgray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            sharpness_ratio = laplacian_var(cgray, mask) / max(laplacian_var(gray, mask), 1e-6)
            metrics = residual_quality_metrics(cgray, det, templates)
            residual = metrics["residual_score"]
            template_residual = metrics["template_residual_score"]
            artifact_penalty = max(0.0, 0.18 - sharpness_ratio) * 0.55
            box_penalty = area * 2.0 if high_contrast_mask and not glyph else 0.0
            cost = (
                min(1.0, residual)
                + min(1.0, template_residual) * 0.18
                + area * 10.0
                + artifact_penalty
                + box_penalty
            )
            score = -cost
            candidates.append((
                score,
                residual,
                template_residual,
                sharpness_ratio,
                area,
                f"{variant}_{method_name}",
                candidate,
                mask,
                metrics,
            ))

    if not candidates:
        return None, first_mask, {
            "status": "needs_manual",
            "reason": "mask_too_large" if first_mask is None else "mask_area_review",
            "mask_area_pct": round(first_area * 100, 3),
        }

    candidates.sort(reverse=True, key=lambda item: item[0])
    _, residual, template_residual, ratio, area, name, best, mask, metrics = candidates[0]

    lama_reason = ""
    if ENABLE_LAMA_ESCALATION and residual >= LAMA_ESCALATION_RESIDUAL_MIN:
        lama, lama_reason = run_lama_escalation(img, mask)
        if lama is not None:
            lgray = cv2.cvtColor(lama, cv2.COLOR_BGR2GRAY)
            lratio = laplacian_var(lgray, mask) / max(laplacian_var(gray, mask), 1e-6)
            lmetrics = residual_quality_metrics(lgray, det, templates)
            lresidual = lmetrics["residual_score"]
            ltemplate = lmetrics["template_residual_score"]
            lcost = (
                min(1.0, lresidual)
                + min(1.0, ltemplate) * 0.18
                + area * 10.0
                + max(0.0, 0.18 - lratio) * 0.55
            )
            current_cost = -candidates[0][0]
            if lcost < current_cost:
                residual = lresidual
                template_residual = ltemplate
                ratio = lratio
                metrics = lmetrics
                best = lama
                name = "lama"

    best_gray = cv2.cvtColor(best, cv2.COLOR_BGR2GRAY)
    post_count = post_clean_detection_count(best, best_gray, templates)
    broad_mask = area > 0.020 or (not name.startswith("glyph_") and area > 0.012)
    position_confident = not det.template.startswith("prior:")
    strict_pass = (
        residual < CLEAN_STRICT_RESIDUAL_MAX
        and template_residual < CLEAN_STRICT_TEMPLATE_MAX
        and metrics["post_text_components"] <= 1
    )
    hard_fail = residual >= CLEAN_FAIL_RESIDUAL_MIN or metrics["post_text_components"] >= CLEAN_FAIL_TEXT_COMPONENTS

    reason = ""
    gate = ""
    ocr_meta: dict = {"ocr_checked": False, "ocr_watermark": None, "ocr_text": []}
    if ratio < SHARPNESS_MIN_RATIO and residual >= CLEAN_FAIL_RESIDUAL_MIN:
        status = "needs_manual"
        reason = "blurry_and_residual"
        gate = "fail_blurry_and_residual"
    elif hard_fail:
        status = "needs_manual"
        reason = "residual_visible"
        gate = "fail_residual_or_text_components"
    elif strict_pass:
        status = "cleaned"
        gate = "pass_strict_post_clean_metrics"
    else:
        ocr_meta = cleaned_crop_ocr_check(best, det, ocr_reader)
        if ocr_meta.get("ocr_checked"):
            if ocr_meta.get("ocr_watermark"):
                status = "needs_manual"
                reason = "ocr_residual_text"
                gate = "gray_zone_ocr_watermark"
            else:
                status = "cleaned"
                gate = "gray_zone_ocr_clear"
        elif post_count == 0 and residual < 0.35 and template_residual < 0.35 and metrics["post_text_components"] <= 2:
            status = "cleaned"
            gate = "gray_zone_post_detection_clear"
        else:
            status = "needs_manual"
            reason = "gray_zone_needs_ocr"
            gate = "gray_zone_no_ocr"

    warnings = []
    if ratio < 0.55:
        warnings.append("low_sharpness_ratio")
    if broad_mask:
        warnings.append("broad_mask")
    if not position_confident:
        warnings.append("prior_detection_source")
    if high_contrast_mask:
        warnings.append("high_contrast_box_mask")

    meta = {
        "status": status,
        "strategy": name,
        "gate": gate,
        "sharpness_ratio": round(ratio, 4),
        "residual_score": round(residual, 4),
        "template_residual_score": round(template_residual, 4),
        "mask_area_pct": round(area * 100, 3),
        "post_text_score": round(metrics["post_text_score"], 4),
        "post_text_components": metrics["post_text_components"],
        "cleaned_detection_count": post_count,
        "post_clean_detection_count": post_count,
        "position_confident": position_confident,
        "broad_mask": broad_mask,
    }
    if ocr_meta.get("ocr_checked") or ocr_meta.get("ocr_error"):
        meta.update({
            "ocr_checked": ocr_meta.get("ocr_checked", False),
            "ocr_watermark": ocr_meta.get("ocr_watermark"),
            "ocr_text": ocr_meta.get("ocr_text", []),
        })
        if ocr_meta.get("ocr_error"):
            meta["ocr_error"] = ocr_meta["ocr_error"]
    if ENABLE_LAMA_ESCALATION and residual >= LAMA_ESCALATION_RESIDUAL_MIN:
        meta["lama_attempted"] = True
        meta["lama_result"] = lama_reason or "not_selected"
    elif ENABLE_LAMA_ESCALATION:
        meta["lama_attempted"] = False
    if reason:
        meta["reason"] = reason
    if warnings:
        meta["warning"] = ";".join(warnings)
    return best, mask, meta


def clean_all_detections(
    img: np.ndarray,
    gray: np.ndarray,
    detections: list[Detection],
    templates: list[TemplateSpec] | None = None,
    ocr_reader=None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    current = img.copy()
    combined_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cleaned_any = False
    details = []
    statuses = []

    for det in detections:
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        cleaned, mask, meta = clean_image(current, current_gray, det, templates, ocr_reader=ocr_reader)
        detail = {**meta, "detection": det.to_json()}
        details.append(detail)
        statuses.append(meta["status"])
        if mask is not None and np.count_nonzero(mask):
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        if cleaned is not None:
            current = cleaned
            cleaned_any = True

    residuals = [float(item["residual_score"]) for item in details if isinstance(item.get("residual_score"), (int, float))]
    template_residuals = [
        float(item["template_residual_score"])
        for item in details
        if isinstance(item.get("template_residual_score"), (int, float))
    ]
    sharpness_ratios = [
        float(item["sharpness_ratio"])
        for item in details
        if isinstance(item.get("sharpness_ratio"), (int, float))
    ]
    text_scores = [float(item["post_text_score"]) for item in details if isinstance(item.get("post_text_score"), (int, float))]
    text_components = [
        int(item["post_text_components"])
        for item in details
        if isinstance(item.get("post_text_components"), int)
    ]
    post_detection_counts = [
        int(item["post_clean_detection_count"])
        for item in details
        if isinstance(item.get("post_clean_detection_count"), int)
    ]
    combined_area = float(np.count_nonzero(combined_mask)) / max(1, combined_mask.size)
    warnings = [str(item.get("warning")) for item in details if item.get("warning")]
    strategies = [str(item.get("strategy")) for item in details if item.get("strategy")]

    if "needs_manual" in statuses:
        status = "needs_manual"
    elif combined_area > COMBINED_MASK_REVIEW_AREA:
        status = "needs_manual"
    else:
        status = "cleaned"

    meta = {
        "status": status,
        "strategy": "multi:" + ",".join(strategies[:4]) if len(strategies) > 1 else (strategies[0] if strategies else ""),
        "detection_count": len(detections),
        "auto_cleaned_detection_count": sum(1 for item in details if item.get("status") != "needs_manual"),
        "cleaned_detection_count": max(post_detection_counts) if post_detection_counts else None,
        "post_clean_detection_count": max(post_detection_counts) if post_detection_counts else None,
        "sharpness_ratio": round(min(sharpness_ratios), 4) if sharpness_ratios else 0.0,
        "residual_score": round(max(residuals), 4) if residuals else 0.0,
        "template_residual_score": round(max(template_residuals), 4) if template_residuals else 0.0,
        "mask_area_pct": round(combined_area * 100, 3),
        "post_text_score": round(max(text_scores), 4) if text_scores else 0.0,
        "post_text_components": max(text_components) if text_components else 0,
        "detection_results": details,
    }
    if status == "needs_manual":
        reasons = [str(item.get("reason")) for item in details if item.get("reason")]
        if combined_area > COMBINED_MASK_REVIEW_AREA and "combined_mask_area_review" not in reasons:
            reasons.append("combined_mask_area_review")
        meta["reason"] = ";".join(dict.fromkeys(reasons)) or "detection_needs_manual"
    elif warnings:
        meta["warning"] = warnings[0]

    final_mask = combined_mask if np.count_nonzero(combined_mask) else None
    final_image = current if cleaned_any else None
    return final_image, final_mask, meta


def cleaning_targets(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []
    targets = [detections[0]]
    if not detections[0].template.startswith("ocr:"):
        return targets
    for det in detections[1:]:
        if det.template.startswith("ocr:") and det.confidence >= 0.60:
            targets.append(det)
        if len(targets) >= 4:
            break
    return targets


def copy_or_write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(path), img)


def make_review_html(out_dir: Path, rows: list[dict]) -> None:
    css = """
body{margin:0;background:#f5f7fb;color:#222;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.top{position:sticky;top:0;background:#111922;color:white;padding:16px 24px;z-index:2}
.top h1{font-size:24px;margin:0 0 6px}.top p{margin:0;color:#c9d1dc}
.wrap{padding:18px}.card{background:white;border:1px solid #d9e0e8;border-radius:8px;margin:0 0 18px;overflow:hidden}
.head{display:flex;justify-content:space-between;gap:16px;align-items:baseline;padding:12px 18px;border-bottom:1px solid #e3e8ef}
.head h2{font-size:20px;margin:0}.meta{color:#667085;font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr}.pane{border-right:1px solid #dfe5ed;text-align:center;background:#fff}
.pane:last-child{border-right:0}.pane img{max-width:100%;height:auto;display:block;margin:0 auto}
.label{border-top:1px solid #e8edf3;padding:9px 0;color:#667085;font-weight:600}
@media(max-width:900px){.grid{grid-template-columns:1fr}.pane{border-right:0;border-bottom:1px solid #dfe5ed}}
"""
    body = []
    for idx, row in enumerate(rows, 1):
        original = html.escape(row.get("review_original", ""))
        mask = html.escape(row.get("review_mask", ""))
        cleaned = html.escape(row.get("review_cleaned", ""))
        diff = html.escape(row.get("review_diff", ""))
        fname = html.escape(row["file"])
        meta_parts = [
            str(row.get("status")),
            str(row.get("strategy", row.get("reason", ""))),
            f"mask {row.get('mask_area_pct', '')}%",
        ]
        if "residual_score" in row:
            meta_parts.append(f"visible {row.get('residual_score')}")
        if "template_residual_score" in row:
            meta_parts.append(f"template {row.get('template_residual_score')}")
        meta = html.escape(" | ".join(part for part in meta_parts if part))
        body.append(f"""
<section class="card">
  <div class="head"><h2>#{idx} {fname}</h2><div class="meta">{meta}</div></div>
  <div class="grid">
    <div class="pane"><img src="{original}" alt="Original"><div class="label">Original</div></div>
    <div class="pane"><img src="{mask}" alt="Mask overlay"><div class="label">Mask overlay</div></div>
    <div class="pane"><img src="{cleaned}" alt="Result"><div class="label">Result</div></div>
    <div class="pane"><img src="{diff}" alt="Difference"><div class="label">Diff x3</div></div>
  </div>
</section>""")
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Watermark Pilot Review</title><style>{css}</style></head>
<body><div class="top"><h1>Watermark pilot review</h1><p>Source images were not modified. Output: {html.escape(str(out_dir))}</p></div>
<main class="wrap">{''.join(body)}</main></body></html>"""
    (out_dir / "review.html").write_text(doc)


def write_blank_mask(path: Path, source: np.ndarray) -> None:
    blank = np.zeros(source.shape[:2], dtype=np.uint8)
    copy_or_write_image(path, blank)


def write_mask_overlay(path: Path, source: np.ndarray, mask: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = source.copy()
    if mask is not None and np.count_nonzero(mask):
        red = np.zeros_like(source)
        red[:, :, 2] = 255
        alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.42
        overlay = np.uint8(np.clip(source.astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha, 0, 255))
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    copy_or_write_image(path, overlay)


def write_diff_image(path: Path, original: np.ndarray, cleaned: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cleaned is None:
        diff = np.zeros_like(original)
    else:
        diff = cv2.absdiff(original, cleaned)
        diff = np.uint8(np.clip(diff.astype(np.float32) * 3.0, 0, 255))
    copy_or_write_image(path, diff)


def manifest_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def build_inventory(assets: Path, out_dir: Path, phash_threshold: int) -> tuple[list[dict], dict]:
    files = iter_images(assets)
    tree = BKTree()
    rows = []
    counts = {"total": 0, "scan": 0, "skipped_iphone14_plus": 0, "duplicates": 0}
    bucket_counts: dict[str, int] = {}
    started = time.time()
    for idx, path in enumerate(files, 1):
        meta = image_meta(path)
        if meta is None:
            continue
        counts["total"] += 1
        if meta["should_scan"]:
            counts["scan"] += 1
        else:
            counts["skipped_iphone14_plus"] += 1
        bucket_counts[meta["bucket"]] = bucket_counts.get(meta["bucket"], 0) + 1
        master = tree.add_or_find(int(meta["phash"], 16), meta["file"], phash_threshold)
        meta["master"] = master
        meta["is_master"] = master == meta["file"]
        if not meta["is_master"]:
            counts["duplicates"] += 1
        rows.append(meta)
        if idx % 2000 == 0:
            rate = idx / max(time.time() - started, 1e-6)
            print(f"  inventory {idx}/{len(files)} ({rate:.0f} img/s)")
    (out_dir / "inventory.json").write_text(json.dumps(rows, indent=2))
    summary = {
        "assets": str(assets),
        "phash_threshold": phash_threshold,
        "counts": counts,
        "buckets": dict(sorted(bucket_counts.items(), key=lambda item: item[1], reverse=True)),
    }
    (out_dir / "inventory-summary.json").write_text(json.dumps(summary, indent=2))
    return rows, summary


def choose_pilot_files(inventory: list[dict], max_total: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = {}
    for row in inventory:
        if row["should_scan"] and row["is_master"]:
            by_bucket.setdefault(row["bucket"], []).append(row)
    chosen: list[str] = []
    buckets = list(by_bucket)
    rng.shuffle(buckets)
    while len(chosen) < max_total and buckets:
        progressed = False
        for bucket in list(buckets):
            rows = by_bucket[bucket]
            if not rows:
                buckets.remove(bucket)
                continue
            row = rows.pop(rng.randrange(len(rows)))
            chosen.append(row["file"])
            progressed = True
            if len(chosen) >= max_total:
                break
        if not progressed:
            break
    return chosen


def process_file(
    path: Path,
    templates: list[TemplateSpec],
    preset: str,
    out_dir: Path,
    review: bool,
    ocr_reader=None,
) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"file": path.name, "status": "needs_manual", "reason": "read_failed"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detect_watermark(gray, templates, preset, img=img, ocr_reader=ocr_reader)
    if not detections:
        entry = {"file": path.name, "status": "no_watermark"}
        if review:
            original_path = out_dir / "originals" / path.name
            original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, original_path)
            mask_path = out_dir / "masks" / path.name
            write_blank_mask(mask_path, img)
            overlay_path = out_dir / "overlays" / path.name
            write_mask_overlay(overlay_path, img, None)
            diff_path = out_dir / "diffs" / path.name
            write_diff_image(diff_path, img, None)
            entry["review_original"] = f"originals/{path.name}"
            entry["review_mask"] = f"overlays/{path.name}"
            entry["mask_binary"] = str(mask_path)
            entry["review_cleaned"] = f"originals/{path.name}"
            entry["review_diff"] = f"diffs/{path.name}"
        return entry
    targets = cleaning_targets(detections)
    cleaned, mask, meta = clean_all_detections(img, gray, targets, templates, ocr_reader=ocr_reader)
    meta["detected_count"] = len(detections)
    entry = {
        "file": path.name,
        **meta,
        "detection": detections[0].to_json(),
        "detections": [det.to_json() for det in detections],
    }
    if mask is not None:
        mask_path = out_dir / "masks" / path.name
        copy_or_write_image(mask_path, mask)
        entry["mask"] = str(mask_path)
    if cleaned is not None:
        clean_path = out_dir / "cleaned" / path.name
        copy_or_write_image(clean_path, cleaned)
        entry["cleaned"] = str(clean_path)
    if review:
        original_path = out_dir / "originals" / path.name
        original_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, original_path)
        entry["review_original"] = f"originals/{path.name}"
        if mask is not None:
            overlay_path = out_dir / "overlays" / path.name
            write_mask_overlay(overlay_path, img, mask)
            entry["review_mask"] = f"overlays/{path.name}"
            entry["mask_binary"] = str(out_dir / "masks" / path.name)
        else:
            overlay_path = out_dir / "overlays" / path.name
            write_mask_overlay(overlay_path, img, None)
            entry["review_mask"] = f"overlays/{path.name}"
        if cleaned is not None:
            entry["review_cleaned"] = f"cleaned/{path.name}"
        elif mask is not None:
            # Show source in result column for manual rows, so review layout stays intact.
            entry["review_cleaned"] = f"originals/{path.name}"
        else:
            entry["review_cleaned"] = f"originals/{path.name}"
        diff_path = out_dir / "diffs" / path.name
        write_diff_image(diff_path, img, cleaned)
        entry["review_diff"] = f"diffs/{path.name}"
    return entry


def cmd_inventory(args: argparse.Namespace) -> None:
    assets = args.assets.expanduser().resolve()
    out_dir = prepare_out_dir(args.out, "inventory")
    rows, summary = build_inventory(assets, out_dir, args.phash_threshold)
    print(f"Inventory written: {out_dir}")
    print(json.dumps(summary["counts"], indent=2))
    print(f"Rows: {len(rows)}")


def cmd_pilot(args: argparse.Namespace) -> None:
    require_rights(args)
    assets = args.assets.expanduser().resolve()
    out_dir = prepare_out_dir(args.out, "pilot")
    inventory, summary = build_inventory(assets, out_dir, args.phash_threshold)
    sample = choose_pilot_files(inventory, args.max_total, args.seed)
    templates = load_templates()
    ocr_reader = load_ocr_reader(args.ocr)
    rows = []
    started = time.time()
    for idx, fname in enumerate(sample, 1):
        row = process_file(assets / fname, templates, args.preset, out_dir, review=True, ocr_reader=ocr_reader)
        rows.append(row)
        print(f"  pilot {idx}/{len(sample)} {row['status']} {fname}")
    make_review_html(out_dir, rows)
    with manifest_writer(out_dir / "manifest.jsonl") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    final = {
        "mode": "pilot",
        "assets": str(assets),
        "out": str(out_dir),
        "preset": args.preset,
        "ocr": bool(args.ocr),
        "seed": args.seed,
        "seconds": round(time.time() - started, 2),
        "inventory": summary["counts"],
        "counts": counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2))
    print(f"Pilot complete: {out_dir}")
    print(f"Review HTML: {out_dir / 'review.html'}")
    print(json.dumps(counts, indent=2))


def cmd_process(args: argparse.Namespace) -> None:
    require_rights(args)
    assets = args.assets.expanduser().resolve()
    out_dir = prepare_out_dir(args.out, "process")
    inventory, summary = build_inventory(assets, out_dir, args.phash_threshold)
    templates = load_templates()
    ocr_reader = load_ocr_reader(args.ocr)

    master_results: dict[str, dict] = {}
    counts: dict[str, int] = {}
    started = time.time()
    limit = args.limit or math.inf
    masters = [
        row for row in inventory
        if row["should_scan"] and row["is_master"]
    ][:int(limit) if limit != math.inf else None]

    def run_master(row: dict) -> dict:
        return process_file(assets / row["file"], templates, args.preset, out_dir, review=False, ocr_reader=ocr_reader)

    done = 0
    if args.workers <= 1:
        for row in masters:
            entry = run_master(row)
            master_results[row["file"]] = entry
            done += 1
            if done % args.progress_every == 0:
                rate = done / max(time.time() - started, 1e-6)
                print(f"  masters {done}/{len(masters)} ({rate:.1f} img/s)")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(run_master, row): row for row in masters}
            for future in as_completed(future_map):
                row = future_map[future]
                try:
                    entry = future.result()
                except Exception as exc:
                    entry = {"file": row["file"], "status": "needs_manual", "reason": f"worker_error:{type(exc).__name__}"}
                master_results[row["file"]] = entry
                done += 1
                if done % args.progress_every == 0:
                    rate = done / max(time.time() - started, 1e-6)
                    print(f"  masters {done}/{len(masters)} ({rate:.1f} img/s)")

    with manifest_writer(out_dir / "manifest.jsonl") as fh:
        for row in inventory:
            fname = row["file"]
            if not row["should_scan"]:
                entry = {"file": fname, "status": "skipped", "reason": "iphone14_plus_known_clean"}
            elif not row["is_master"]:
                master = row["master"]
                master_entry = master_results.get(master)
                if master_entry and master_entry.get("status") == "cleaned":
                    src = Path(master_entry["cleaned"])
                    dst = out_dir / "cleaned" / fname
                    shutil.copy2(src, dst)
                    entry = {"file": fname, "status": "cleaned_duplicate", "master": master, "cleaned": str(dst)}
                else:
                    entry = {"file": fname, "status": "duplicate_no_action", "master": master}
            else:
                entry = master_results.get(fname, {"file": fname, "status": "not_processed", "reason": "limit"})

            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

            total_seen = sum(counts.values())
            if total_seen % args.progress_every == 0:
                rate = total_seen / max(time.time() - started, 1e-6)
                print(f"  process {total_seen}/{len(inventory)} ({rate:.1f} img/s) {counts}")

    final = {
        "mode": "process",
        "assets": str(assets),
        "out": str(out_dir),
        "preset": args.preset,
        "ocr": bool(args.ocr),
        "seconds": round(time.time() - started, 2),
        "inventory": summary["counts"],
        "counts": counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2))
    print(f"Process complete: {out_dir}")
    print(json.dumps(counts, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Sunsky watermark removal pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
        p.add_argument("--out", type=Path)
        p.add_argument("--phash-threshold", type=int, default=0)

    p = sub.add_parser("inventory", help="Build inventory, buckets, and pHash duplicate map")
    common(p)
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("pilot", help="Run bucket-balanced QA pilot and review HTML")
    common(p)
    p.add_argument("--preset", choices=sorted(PRESETS), default="review")
    p.add_argument("--max-total", type=int, default=50)
    p.add_argument("--seed", type=int, default=1779606245)
    p.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EasyOCR text confirmation during the pilot. Enabled by default; pass --no-ocr for faster heuristic-only QA.",
    )
    p.add_argument("--rights-confirmed", action="store_true")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("process", help="One-pass detect-clean-manifest workflow")
    common(p)
    p.add_argument("--preset", choices=sorted(PRESETS), default="review")
    p.add_argument("--workers", type=int, default=1, help="Reserved for future parallel mode; current pipeline is sequential for stable manifests.")
    p.add_argument("--limit", type=int)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--ocr", action="store_true", help="Use optional EasyOCR text confirmation. This is slower and best for QA runs.")
    p.add_argument("--rights-confirmed", action="store_true")
    p.set_defaults(func=cmd_process)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
