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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from sunsky_alpha_engine import (
    ALPHA_FLOOR,
    FINAL_ALPHA_TEMPLATE_MAX,
    MIN_ALPHA_RESIDUAL_REDUCTION,
    SunskyAlphaEngine,
    score_alpha_residual,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/Users/alexkou/Documents/github/b2bweb/content/products/assets")
SOURCE_REPO = Path("/Users/alexkou/Documents/github/b2bweb").resolve()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
SUNSKY_ALPHA_PATH = TEMPLATE_DIR / "sunsky-alpha.png"
SUNSKY_ALPHA_META_PATH = TEMPLATE_DIR / "sunsky-alpha-meta.json"
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
OCR_DETECTION_MAX_AREA = 0.045
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
FINAL_RESIDUAL_MAX = 0.18
FINAL_TEMPLATE_MAX = 0.20
FINAL_TEXT_COMPONENTS_MAX = 1
DOT_CHAIN_SCORE_MAX = 0.28
DOT_CHAIN_COMPONENT_COUNT = 4
DOT_CHAIN_SPAN_MIN = 0.22
DOT_CHAIN_AREA_RATIO_MIN = 0.035
VISIBLE_BAND_SCORE_MAX = 0.18
VISIBLE_BAND_LUMA_DELTA_MAX = 8.0
RESIDUAL_CLEANUP_AREA_MAX = 0.016
RESIDUAL_CLEANUP_RISKY_AREA_MAX = 0.006
RESIDUAL_CLEANUP_DILATE_X = 5
RESIDUAL_CLEANUP_DILATE_Y = 2
RESIDUAL_CLEANUP_MAX_AREA_MULTIPLIER = 2.2
RESIDUAL_CLEANUP_MAX_AREA_PCT = 0.012
RISKY_RESIDUAL_CLEANUP_MAX_AREA_MULTIPLIER = 1.45
RISKY_RESIDUAL_CLEANUP_DILATE_X = 3
RISKY_RESIDUAL_CLEANUP_DILATE_Y = 1
TAIL_EXPAND_RATIO_X = 0.08
TAIL_EXPAND_MIN_PX = 4
TAIL_EXPAND_MAX_PX = 18
TOP_K_CANDIDATES = 5
MAX_SECOND_PASS_ATTEMPTS = 2
MAX_TOTAL_REPAIR_CANDIDATES = 24
COMBINED_MASK_REVIEW_AREA = 0.035
MIN_TEXT_COMPONENTS = 4
HIGH_CONTRAST_SPAN = 200.0
REVIEW_CONTRAST_SPAN = 170.0
FALLBACK_CONTRAST_SPAN = 150.0
LINE_DOMINANCE_MAX = 0.72
OCR_CANONICAL = "sunskyonlinecom"
OCR_MATCH_MIN = 0.74
OCR_DIRECT_MIN = 0.78
OCR_CROP_MIN = 0.76
OCR_LOW_CONF_DIRECT_MIN = 0.88
OCR_POST_CLEAN_SUSPECT_MIN = 0.62
OCR_CROP_LOCALIZE_MIN = 0.62
WATERMARK_CANONICAL_ASPECT = 7.86
ENABLE_BRIGHT_RECALL = True
ENABLE_HIGH_CONTRAST_BOX_MASK = True
ENABLE_LAMA_ESCALATION = True
_CANONICAL_INK_MASK: np.ndarray | None = None
_SIMPLE_LAMA = None
_SUNSKY_ALPHA_ENGINE: SunskyAlphaEngine | None = None
_SUNSKY_ALPHA_META: dict | None = None


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
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    ocr_watermark_score: float = 0.0
    roi_class: str = ""
    product_overlap: float = 0.0
    layout_risk: str = ""

    def to_json(self) -> dict:
        payload = {
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
        if self.ocr_text:
            payload["ocr_text"] = self.ocr_text
            payload["ocr_confidence"] = round(self.ocr_confidence, 4)
            payload["ocr_watermark_score"] = round(self.ocr_watermark_score, 4)
        if self.roi_class:
            payload["roi_class"] = self.roi_class
            payload["product_overlap"] = round(self.product_overlap, 4)
        if self.layout_risk:
            payload["layout_risk"] = self.layout_risk
        return payload


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    image: np.ndarray
    kind: str
    start: float = 0.0
    end: float = 1.0


@dataclass
class RepairCandidate:
    id: str
    rank_score: float
    residual: float
    template_residual: float
    sharpness_ratio: float
    mask_area: float
    strategy: str
    image: np.ndarray
    mask: np.ndarray
    metrics: dict
    category: str = "candidate_failed_metrics_invalid"
    gray: np.ndarray | None = None
    post_count: int | None = None
    ocr_meta: dict | None = None
    dot_metrics: dict | None = None
    band_metrics: dict | None = None
    product_metrics: dict | None = None
    gate_meta: dict | None = None
    second_pass: bool = False
    second_pass_strategy: str = ""
    second_pass_mask_area: float = 0.0


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


def sunsky_alpha_engine() -> SunskyAlphaEngine | None:
    global _SUNSKY_ALPHA_ENGINE
    if _SUNSKY_ALPHA_ENGINE is not None:
        return _SUNSKY_ALPHA_ENGINE
    engine = SunskyAlphaEngine(
        SUNSKY_ALPHA_PATH,
        TEMPLATE_DIR / "watermark-template.png",
        min_alignment_score=0.35,
    )
    if not engine.alpha_available():
        return None
    _SUNSKY_ALPHA_ENGINE = engine
    return _SUNSKY_ALPHA_ENGINE


def sunsky_alpha_meta() -> dict:
    global _SUNSKY_ALPHA_META
    if _SUNSKY_ALPHA_META is not None:
        return _SUNSKY_ALPHA_META
    if not SUNSKY_ALPHA_META_PATH.exists():
        _SUNSKY_ALPHA_META = {}
        return _SUNSKY_ALPHA_META
    try:
        _SUNSKY_ALPHA_META = json.loads(SUNSKY_ALPHA_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        _SUNSKY_ALPHA_META = {}
    return _SUNSKY_ALPHA_META


def mark_box_tuple(det: Detection) -> tuple[int, int, int, int]:
    b = det.mark_box
    return int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])


def create_glyph_halo_mask(
    template_alpha: np.ndarray,
    mark_box: dict,
    halo_px_x: int,
    halo_px_y: int,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Scale canonical template alpha into mark_box, then add a controlled
    low-alpha halo around likely watermark pixels.
    """
    b = mark_box
    ih, iw = template_alpha.shape[:2]
    target_w = max(24, int(round(b["w"] * 0.94)))
    target_h = max(8, int(round(target_w * ih / max(iw, 1))))
    if target_h > b["h"] * 0.95:
        target_h = max(8, int(round(b["h"] * 0.95)))
        target_w = max(24, int(round(target_h * iw / max(ih, 1))))

    if shape is None:
        out_h = max(target_h, int(b["h"]))
        out_w = max(target_w, int(b["w"]))
        tx = max(0, int(round((out_w - target_w) / 2)))
        ty = max(0, int(round((out_h - target_h) / 2)))
        mask = np.zeros((out_h, out_w), dtype=np.uint8)
    else:
        out_h, out_w = shape
        target_w = min(target_w, out_w)
        target_h = min(target_h, out_h)
        tx = int(round(b["x"] + (b["w"] - target_w) / 2))
        ty = int(round(b["y"] + (b["h"] - target_h) / 2))
        tx = max(0, min(tx, out_w - target_w))
        ty = max(0, min(ty, out_h - target_h))
        mask = np.zeros((out_h, out_w), dtype=np.uint8)

    resized = cv2.resize(template_alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)
    _, glyph = cv2.threshold(resized, 12, 255, cv2.THRESH_BINARY)
    mask[ty:ty + target_h, tx:tx + target_w] = glyph
    if halo_px_x > 0 or halo_px_y > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(1, halo_px_x * 2 + 1), max(1, halo_px_y * 2 + 1)),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


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


def image_layout_features(gray: np.ndarray, img: np.ndarray | None = None) -> dict:
    """Detect dense instruction-sheet layouts that should not use weak priors."""
    h, w = gray.shape[:2]
    scale = min(1.0, 720.0 / max(h, w, 1))
    small = gray
    if scale < 1.0:
        small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    edges = cv2.Canny(small, 60, 150)
    edge_density = float(np.mean(edges > 0))

    binary = (small < 210).astype(np.uint8) * 255
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    small_components = 0
    long_horizontal = 0
    long_vertical = 0
    for idx in range(1, count):
        _, _, ww, hh, area = stats[idx]
        if area < 3:
            continue
        if area <= max(18, small.size * 0.0025) and ww <= sw * 0.24 and hh <= sh * 0.10:
            small_components += 1
        if ww >= sw * 0.28 and hh <= max(5, sh * 0.015):
            long_horizontal += 1
        if hh >= sh * 0.22 and ww <= max(5, sw * 0.015):
            long_vertical += 1

    row_edge_ratio = np.mean(edges > 0, axis=1) if edges.size else np.array([], dtype=np.float32)
    col_edge_ratio = np.mean(edges > 0, axis=0) if edges.size else np.array([], dtype=np.float32)
    strong_edge_rows = int(np.count_nonzero(row_edge_ratio > 0.42))
    strong_edge_cols = int(np.count_nonzero(col_edge_ratio > 0.42))
    panel_line_score = min(1.0, (long_horizontal + long_vertical + strong_edge_rows + strong_edge_cols) / 16.0)
    text_dense_score = min(1.0, small_components / 170.0) * 0.62 + min(1.0, edge_density / 0.085) * 0.22 + panel_line_score * 0.16
    text_dense = text_dense_score >= 0.62 or (small_components >= 120 and panel_line_score >= 0.25)
    step_layout = panel_line_score >= 0.42 and small_components >= 75
    return {
        "edge_density": edge_density,
        "small_component_count": int(small_components),
        "panel_line_score": float(panel_line_score),
        "text_dense_score": float(text_dense_score),
        "text_dense_layout": bool(text_dense),
        "step_layout": bool(step_layout),
    }


def estimate_product_overlap_v13(gray: np.ndarray, box: dict) -> dict:
    """Classify the watermark footprint using interior product signals.

    This mirrors the useful part of the V13 design: routing should come from
    what is under the watermark, not from detector provenance. High product
    overlap means broad white/box fills are unsafe.
    """
    img_h, img_w = gray.shape[:2]
    x = max(0, int(box["x"]))
    y = max(0, int(box["y"]))
    w = max(1, min(int(box["w"]), img_w - x))
    h = max(1, min(int(box["h"]), img_h - y))
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return {
            "roi_class": "unknown",
            "product_overlap": 0.0,
            "roi_edge_density": 0.0,
            "roi_dark_ratio": 0.0,
            "roi_nonwhite_ratio": 0.0,
            "protected_text_risk": False,
        }

    mean_luma = float(np.mean(roi))
    std_luma = float(np.std(roi))
    dark_ratio = float(np.mean(roi < 95))
    nonwhite_ratio = float(np.mean(roi < 238))
    white_ratio = float(np.mean(roi > 245))
    edges = cv2.Canny(roi, 55, 140)
    edge_density = float(np.mean(edges > 0))
    features = band_features(gray, box)
    text_score, text_components = text_likeness(gray, box)

    product_overlap = min(1.0, nonwhite_ratio * 0.48 + dark_ratio * 0.28 + min(1.0, edge_density / 0.16) * 0.18 + min(1.0, std_luma / 80.0) * 0.06)
    protected_text_risk = bool(text_components >= 12 and features["contrast_span"] >= 120 and edge_density >= 0.065)

    if white_ratio >= 0.86 and edge_density < 0.030 and std_luma < 22:
        roi_class = "plain_white"
    elif mean_luma > 218 and edge_density < 0.065 and dark_ratio < 0.08:
        roi_class = "near_white"
    elif dark_ratio >= 0.38 or mean_luma < 120:
        roi_class = "dark_product_surface"
    elif features["line_dominance"] >= 0.46 and (dark_ratio > 0.20 or edge_density > 0.08):
        roi_class = "thin_flex_cable"
    elif protected_text_risk:
        roi_class = "text_or_label_area"
    elif edge_density >= 0.135 or features["contrast_span"] >= REVIEW_CONTRAST_SPAN:
        roi_class = "complex_product_detail"
    elif edge_density < 0.055 and std_luma < 34:
        roi_class = "low_texture_background"
    else:
        roi_class = "simple_product_surface" if product_overlap >= 0.28 else "unknown"

    return {
        "roi_class": roi_class,
        "product_overlap": float(product_overlap),
        "roi_edge_density": edge_density,
        "roi_dark_ratio": dark_ratio,
        "roi_nonwhite_ratio": nonwhite_ratio,
        "protected_text_risk": protected_text_risk,
    }


def annotate_detection_context(det: Detection, gray: np.ndarray, layout: dict | None = None) -> Detection:
    roi_meta = estimate_product_overlap_v13(gray, det.mark_box)
    det.roi_class = str(roi_meta["roi_class"])
    det.product_overlap = float(roi_meta["product_overlap"])
    if layout and layout.get("text_dense_layout"):
        det.layout_risk = "step_layout" if layout.get("step_layout") else "text_dense_layout"
    return det


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
        if det.roi_class in {"dark_product_surface", "thin_flex_cable", "complex_product_detail", "text_or_label_area"}:
            rank -= 0.85
        if det.product_overlap >= 0.55:
            rank -= 0.40
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
    norm = re.sub(r"[^a-z0-9]", "", text.lower())
    return norm.translate(str.maketrans({
        "0": "o",
        "1": "l",
        "3": "e",
        "5": "s",
        "7": "t",
    }))


def _window_similarity(norm: str, token: str) -> float:
    if not norm or not token:
        return 0.0
    if token in norm:
        return 1.0
    tlen = len(token)
    best = 0.0
    min_len = max(2, tlen - 1)
    max_len = min(len(norm), tlen + 2)
    for size in range(min_len, max_len + 1):
        for start in range(0, len(norm) - size + 1):
            best = max(best, SequenceMatcher(None, norm[start:start + size], token).ratio())
            if best >= 0.98:
                return best
    return best


def watermark_text_score(text: str) -> float:
    """Score whether OCR text is specifically the Sunsky watermark domain.

    The previous matcher accepted broad fragments such as "online...com" or
    "alinec", which pulled clean product-detail images into watermarked-only
    pilots. This score requires the domain structure: a Sunsky/sky-like left
    token plus an online/com-like right token. Fuzzy matching remains because
    low-opacity watermark OCR often reads "sunsky" as "sumsky", "sursky", or
    drops the leading "sun".
    """
    norm = normalize_ocr_text(text)
    if len(norm) < 7:
        return 0.0

    full = SequenceMatcher(None, norm, OCR_CANONICAL).ratio()
    sunsky = max(
        _window_similarity(norm, "sunsky"),
        _window_similarity(norm, "sursky"),
        _window_similarity(norm, "sumsky"),
        _window_similarity(norm, "sunsk"),
    )
    sky = 1.0 if "sky" in norm else _window_similarity(norm, "sky")
    online = max(
        _window_similarity(norm, "online"),
        _window_similarity(norm, "onlne"),
        _window_similarity(norm, "oniine"),
        _window_similarity(norm, "onlin"),
    )
    com = max(_window_similarity(norm, "com"), 0.72 if norm.endswith("co") or "co" in norm[-4:] else 0.0)

    if sunsky >= 0.68 and online >= 0.70:
        return float(min(1.0, sunsky * 0.42 + online * 0.40 + com * 0.12 + full * 0.06))
    if sky >= 0.95 and online >= 0.76 and com >= 0.60:
        return float(min(0.86, 0.30 * sky + 0.48 * online + 0.16 * com + 0.06 * full))
    if full >= 0.80 and (sunsky >= 0.58 or sky >= 0.90) and online >= 0.62:
        return float(min(0.90, full))
    return 0.0


def ocr_text_matches_watermark(text: str, min_score: float = OCR_MATCH_MIN) -> bool:
    return watermark_text_score(text) >= min_score


def ocr_confidence_pass(score: float, conf: float, *, crop: bool = False) -> bool:
    if crop:
        return score >= OCR_CROP_MIN and (conf >= 0.14 or score >= OCR_LOW_CONF_DIRECT_MIN)
    return score >= OCR_DIRECT_MIN and (conf >= 0.18 or score >= OCR_LOW_CONF_DIRECT_MIN)


def ocr_crop_localization_pass(score: float, conf: float) -> bool:
    return score >= OCR_CROP_LOCALIZE_MIN and (conf >= 0.08 or score >= 0.86)


def load_ocr_reader(enabled: bool):
    if not enabled:
        return None
    try:
        import easyocr  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        raise SystemExit(f"--ocr requested but easyocr is unavailable: {exc}") from exc
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def ocr_mark_box_from_points(
    pts: np.ndarray,
    text: str,
    img_w: int,
    img_h: int,
) -> dict:
    """Normalize an OCR text bbox to the actual Sunsky domain footprint.

    EasyOCR sometimes merges nearby product labels into the same line, e.g.
    "sunsky-online 12mini". The repair mask must stay on the domain text, so
    very wide OCR lines are clamped back toward the canonical watermark aspect.
    """
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    raw_w = max(1.0, x2 - x1)
    raw_h = max(1.0, y2 - y1)
    cx = x1 + raw_w / 2.0
    cy = y1 + raw_h / 2.0

    mark_h = max(10.0, raw_h * 1.18)
    mark_h = min(mark_h, img_h * 0.072)
    min_w = max(45.0, raw_h * 5.4)
    max_w = min(img_w * 0.46, max(raw_w * 1.22, raw_h * 11.2))
    raw_aspect = raw_w / raw_h
    norm = normalize_ocr_text(text)

    domain_end = -1
    for token in ("online", "onlne", "oniine", "onlin", "onling", "oniin", "onlinl"):
        pos = norm.find(token)
        if pos >= 0:
            domain_end = max(domain_end, pos + len(token))
    com_pos = norm.find("com", max(0, domain_end))
    if com_pos >= 0:
        domain_end = max(domain_end, com_pos + 3)
    trailing = norm[domain_end:] if domain_end >= 0 else ""
    has_trailing_model = bool(trailing) and (
        any(ch.isdigit() for ch in trailing)
        or any(token in trailing for token in ("mini", "pro", "max", "plus"))
    )
    if has_trailing_model and norm.startswith(("sun", "sur", "sum", "unsky", "sky")):
        domain_fraction = max(0.48, min(0.92, (domain_end + 1) / max(len(norm), 1)))
        trailing_min_w = max(45.0, raw_h * 4.6)
        mark_w = max(trailing_min_w, raw_w * domain_fraction * 0.92)
        mark_w = min(max_w, mark_h * 5.0, mark_w)
        x = x1 - max(3.0, mark_w * 0.025)
        y = cy - mark_h / 2.0
        return clamp_box(x, y, mark_w, mark_h, img_w, img_h)

    if raw_aspect > 11.2:
        mark_w = min(max_w, max(min_w, raw_h * 9.2))
        # Most merged OCR lines start with the watermark and append a product
        # token. Anchor left so the extra trailing label is not masked.
        if norm.startswith(("sun", "sur", "sum", "unsky", "sky")):
            x = x1 - max(3.0, mark_w * 0.03)
        else:
            x = cx - mark_w / 2.0
    elif raw_aspect < 5.2:
        mark_w = min(max_w, max(raw_w * 1.08, WATERMARK_CANONICAL_ASPECT * raw_h))
        x = cx - mark_w / 2.0
    else:
        mark_w = min(max_w, max(min_w, raw_w * 1.08))
        x = cx - mark_w / 2.0

    y = cy - mark_h / 2.0
    return clamp_box(x, y, mark_w, mark_h, img_w, img_h)


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
        raw_text = str(text)
        text_score_match = watermark_text_score(raw_text)
        conf = float(conf)
        if not ocr_confidence_pass(text_score_match, conf, crop=False):
            continue
        pts = np.array(box, dtype=np.float32)
        mark = ocr_mark_box_from_points(pts, raw_text, img_w, img_h)
        area_pct = 100.0 * mark["w"] * mark["h"] / max(1, img_w * img_h)
        if area_pct > 100 * OCR_DETECTION_MAX_AREA:
            continue
        text_score, text_components = text_likeness(gray, mark)
        features = band_features(gray, mark)
        ocr_score = max(0.75, text_score_match * 0.74 + min(1.0, conf) * 0.26)
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
            template=f"ocr:{raw_text[:48]}", scale=1.0,
            mark_box=mark, mask_area_pct=area_pct,
            text_score=max(text_score, 0.95), text_components=max(text_components, 12),
            contrast_span=features["contrast_span"],
            line_dominance=features["line_dominance"],
            confidence=confidence,
            ocr_text=raw_text,
            ocr_confidence=conf,
            ocr_watermark_score=text_score_match,
        ))
    return detections


def ocr_crop_watermark_detections(
    img: np.ndarray,
    gray: np.ndarray,
    source_detections: list[Detection],
    reader,
    *,
    max_sources: int = 12,
) -> list[Detection]:
    """Localize faint watermarks inside candidate crops.

    The template/prior detector is allowed to propose a broad neighborhood for
    recall, but it is not allowed to decide the repair position when crop OCR
    can read the actual Sunsky string. This fixes the common failure where a
    low-contrast watermark above a dark flex cable is confirmed by OCR, while
    the mask itself lands on product printing below it.
    """
    if reader is None or img.size == 0 or not source_detections:
        return []
    img_h, img_w = gray.shape[:2]
    localized: list[Detection] = []
    # Use a wider source beam than the final repair beam. Some wrong prior
    # boxes score higher than the true faint watermark on product surfaces; the
    # OCR crop pass is specifically meant to rescue those lower-ranked but
    # nearby candidates.
    sources = sorted(source_detections, key=lambda d: detection_rank(d, img_w, img_h), reverse=True)[:max_sources]
    for src in sources:
        b = src.mark_box
        pad_x = max(18, int(round(b["w"] * 0.28)))
        pad_y = max(18, int(round(b["h"] * 2.40)))
        x1 = max(0, b["x"] - pad_x)
        y1 = max(0, b["y"] - pad_y)
        x2 = min(img_w, b["x"] + b["w"] + pad_x)
        y2 = min(img_h, b["y"] + b["h"] + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 12 or crop.shape[1] < 45:
            continue
        try:
            results = reader.readtext(
                crop,
                detail=1,
                paragraph=False,
                text_threshold=0.14,
                low_text=0.035,
                link_threshold=0.08,
                canvas_size=760,
                mag_ratio=2.2,
            )
        except Exception:
            continue
        for box, text, conf in results:
            raw_text = str(text)
            text_score_match = watermark_text_score(raw_text)
            conf = float(conf)
            if not ocr_crop_localization_pass(text_score_match, conf):
                continue
            pts = np.array(box, dtype=np.float32)
            pts[:, 0] += x1
            pts[:, 1] += y1
            mark = ocr_mark_box_from_points(pts, raw_text, img_w, img_h)
            area_pct = 100.0 * mark["w"] * mark["h"] / max(1, img_w * img_h)
            if area_pct > 100 * OCR_DETECTION_MAX_AREA:
                continue
            text_score, text_components = text_likeness(gray, mark)
            features = band_features(gray, mark)
            confidence = min(1.0, 0.86 + text_score_match * 0.10 + min(1.0, conf) * 0.04)
            localized.append(Detection(
                x=mark["x"], y=mark["y"], w=mark["w"], h=mark["h"],
                score=max(0.76, text_score_match),
                verify_score=max(0.78, src.verify_score),
                template=f"ocr:crop:{raw_text[:42]}",
                scale=1.0,
                mark_box=mark,
                mask_area_pct=area_pct,
                text_score=max(text_score, 0.92),
                text_components=max(text_components, 12),
                contrast_span=features["contrast_span"],
                line_dominance=features["line_dominance"],
                confidence=confidence,
                ocr_text=raw_text,
                ocr_confidence=conf,
                ocr_watermark_score=text_score_match,
            ))
    return localized


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
    best_score = 0.0
    best_text = ""
    best_conf = 0.0
    for _, text, conf in results:
        text = str(text)
        if not text.strip():
            continue
        conf = float(conf)
        score = watermark_text_score(text)
        texts.append(f"{text}:{conf:.2f}:{score:.2f}")
        if score > best_score:
            best_score = score
            best_text = text
            best_conf = conf
        if ocr_confidence_pass(score, conf, crop=True):
            watermark = True
    return {
        "ocr_checked": True,
        "ocr_watermark": watermark,
        "ocr_text": texts[:6],
        "ocr_best_text": best_text,
        "ocr_best_confidence": best_conf,
        "ocr_watermark_score": best_score,
    }


def confirm_watermark_presence(
    img: np.ndarray,
    gray: np.ndarray,
    detections: list[Detection],
    ocr_reader=None,
    layout: dict | None = None,
) -> dict:
    """Confirm that a detection is likely the Sunsky watermark, not a product detail.

    The permissive prior text-band detector is useful for recall, but it must
    not decide watermarked-only sampling by itself. OCR is authoritative when
    available; without OCR, only strong canonical-template detections pass.
    """
    if not detections:
        return {"presence_confirmed": False, "presence_reason": "no_detection"}

    layout = layout or image_layout_features(gray, img)
    ocr_detections = [det for det in detections if det.template.startswith("ocr:")]
    if ocr_detections:
        best_ocr = max(ocr_detections, key=lambda det: (det.ocr_watermark_score, det.ocr_confidence, det.confidence))
        if ocr_confidence_pass(best_ocr.ocr_watermark_score, best_ocr.ocr_confidence, crop=False):
            return {
                "presence_confirmed": True,
                "presence_reason": "ocr_detection",
                "presence_score": round(best_ocr.ocr_watermark_score, 4),
                "presence_ocr_text": [f"{best_ocr.ocr_text}:{best_ocr.ocr_confidence:.2f}:{best_ocr.ocr_watermark_score:.2f}"],
            }

    if ocr_reader is not None:
        checked = []
        best_crop_score = 0.0
        best_crop_text = ""
        best_crop_conf = 0.0
        for det in detections[:3]:
            pad_x = max(10, int(round(det.mark_box["w"] * 0.16)))
            pad_y = max(8, int(round(det.mark_box["h"] * 0.90)))
            crop = padded_crop(img, det.mark_box, pad_x, pad_y)
            ocr_meta = ocr_image_watermark_check(crop, ocr_reader, canvas_size=760, mag_ratio=2.0)
            checked.extend(ocr_meta.get("ocr_text", []))
            if float(ocr_meta.get("ocr_watermark_score") or 0.0) > best_crop_score:
                best_crop_score = float(ocr_meta.get("ocr_watermark_score") or 0.0)
                best_crop_text = str(ocr_meta.get("ocr_best_text") or "")
                best_crop_conf = float(ocr_meta.get("ocr_best_confidence") or 0.0)
            if ocr_meta.get("ocr_watermark"):
                return {
                    "presence_confirmed": True,
                    "presence_reason": "ocr_crop_confirmed",
                    "presence_score": round(best_crop_score, 4),
                    "presence_best_text": best_crop_text,
                    "presence_best_confidence": round(best_crop_conf, 4),
                    "presence_ocr_text": checked[:8],
                }
        return {
            "presence_confirmed": False,
            "presence_reason": "ocr_crop_not_confirmed",
            "presence_score": round(best_crop_score, 4),
            "presence_best_text": best_crop_text,
            "presence_best_confidence": round(best_crop_conf, 4),
            "presence_ocr_text": checked[:8],
        }

    best = detections[0]
    canonical = best.template.startswith("watermark-template")
    strong_template = (
        canonical
        and best.confidence >= 0.68
        and best.verify_score >= 0.56
        and best.text_score >= 0.42
        and best.text_components >= 10
        and best.contrast_span <= REVIEW_CONTRAST_SPAN
        and best.line_dominance <= 0.45
    )
    if layout.get("text_dense_layout") and not best.template.startswith("watermark-template"):
        return {
            "presence_confirmed": False,
            "presence_reason": "text_dense_layout_requires_direct_evidence",
            "presence_score": 0.0,
        }
    if strong_template:
        return {"presence_confirmed": True, "presence_reason": "strong_template_no_ocr", "presence_score": round(best.confidence, 4)}
    return {
        "presence_confirmed": False,
        "presence_reason": "weak_or_prior_detection_without_ocr",
        "presence_score": round(best.confidence, 4),
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
    layout: dict | None = None,
) -> list[Detection]:
    preset = PRESETS[preset_name]
    img_h, img_w = gray.shape[:2]
    layout = layout or image_layout_features(gray, img)
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
    if not layout.get("text_dense_layout"):
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
    if ENABLE_BRIGHT_RECALL and not found and ocr_reader is not None and not layout.get("text_dense_layout"):
        found.extend(bright_background_text_band_detections(gray, img=img, ocr_reader=ocr_reader))
    if img is not None and ocr_reader is not None and found:
        found.extend(ocr_crop_watermark_detections(img, gray, found, ocr_reader))
    for det in found:
        annotate_detection_context(det, gray, layout)
    return nms(found, img_w, img_h, int(preset["max_detections"]))


def create_mask(
    gray: np.ndarray,
    det: Detection,
    img: np.ndarray | None = None,
    pad_x: int = 8,
    pad_y: int = 5,
    dilate_px: int = 4,
    glyph: bool = False,
    halo: bool = False,
    tail_guard: bool = False,
) -> tuple[np.ndarray | None, float]:
    img_h, img_w = gray.shape[:2]
    b = det.mark_box
    x1 = max(0, b["x"] - pad_x)
    y1 = max(0, b["y"] - pad_y)
    x2 = min(img_w, b["x"] + b["w"] + pad_x)
    y2 = min(img_h, b["y"] + b["h"] + pad_y)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if glyph:
        if det.template.startswith(("ocr:", "prior:", "watermark-template")):
            ink = canonical_ink_mask()
            if halo:
                mask = create_glyph_halo_mask(
                    ink,
                    b,
                    max(1, dilate_px),
                    max(1, int(round(dilate_px * 0.45))),
                    gray.shape[:2],
                )
                dilate_px = 0
            else:
                ih, iw = ink.shape[:2]
                target_w = max(24, int(round(b["w"] * 0.94)))
                target_h = max(8, int(round(target_w * ih / max(iw, 1))))
                if target_h > b["h"] * 0.95:
                    target_h = max(8, int(round(b["h"] * 0.95)))
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
            if tail_guard and det.roi_class != "text_or_label_area":
                tail_px = int(round(b["w"] * TAIL_EXPAND_RATIO_X))
                tail_px = max(TAIL_EXPAND_MIN_PX, min(TAIL_EXPAND_MAX_PX, tail_px))
                tail_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tail_px * 2 + 1, 3))
                tail_mask = cv2.dilate(mask, tail_kernel, iterations=1)
                text_line = np.zeros_like(mask)
                y_line1 = max(0, b["y"] - max(1, b["h"] // 8))
                y_line2 = min(img_h, b["y"] + b["h"] + max(1, b["h"] // 8))
                x_line1 = max(0, b["x"] - tail_px)
                x_line2 = min(img_w, b["x"] + b["w"] + tail_px)
                text_line[y_line1:y_line2, x_line1:x_line2] = 255
                mask = cv2.bitwise_or(mask, cv2.bitwise_and(tail_mask, text_line))
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


def residual_component_metrics(gray: np.ndarray, det: Detection) -> dict:
    """Detect dot-chain or broken-glyph residue in the known watermark footprint."""
    img_h, img_w = gray.shape[:2]
    b = det.mark_box
    pad_x = max(10, int(round(b["w"] * 0.25)))
    pad_y = max(6, int(round(b["h"] * 0.55)))
    x1 = max(0, b["x"] - pad_x)
    y1 = max(0, b["y"] - pad_y)
    x2 = min(img_w, b["x"] + b["w"] + pad_x)
    y2 = min(img_h, b["y"] + b["h"] + pad_y)
    roi = gray[y1:y2, x1:x2]
    full_mask = np.zeros_like(gray)
    if roi.shape[0] < 8 or roi.shape[1] < 40:
        return {
            "dot_chain_score": 0.0,
            "dot_component_count": 0,
            "dot_horizontal_span": 0.0,
            "dot_component_area_ratio": 0.0,
            "component_mask_area": 0.0,
            "dot_chain_fail": False,
            "component_mask": full_mask,
        }

    kernel = max(5, min(31, (roi.shape[0] // 2) * 2 + 1))
    background = cv2.medianBlur(roi, kernel)
    dev = cv2.absdiff(roi, background)
    percentile = 84 if det.product_overlap >= 0.42 else 80
    threshold = max(2.0, float(np.percentile(dev, percentile)))
    raw = (dev >= threshold).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    xs = []
    total_area = 0
    components = 0
    for idx in range(1, count):
        xx, yy, ww, hh, area = stats[idx]
        if area < 2 or area > roi.size * 0.055:
            continue
        if hh < 2 or hh > roi.shape[0] * 0.78:
            continue
        if ww > roi.shape[1] * 0.30:
            continue
        components += 1
        total_area += int(area)
        xs.extend([xx, xx + ww])
        kept[labels == idx] = 255

    horizontal_span = ((max(xs) - min(xs)) / max(roi.shape[1], 1)) if xs else 0.0
    area_ratio = total_area / max(roi.size, 1)
    score = (
        min(1.0, components / 8.0) * 0.38
        + min(1.0, horizontal_span) * 0.42
        + min(1.0, area_ratio / 0.08) * 0.20
    )
    fail = (
        score > DOT_CHAIN_SCORE_MAX
        or (
            components >= DOT_CHAIN_COMPONENT_COUNT
            and horizontal_span > DOT_CHAIN_SPAN_MIN
            and area_ratio > DOT_CHAIN_AREA_RATIO_MIN
        )
    )
    full_mask[y1:y2, x1:x2] = kept
    component_mask_area = float(np.count_nonzero(full_mask)) / max(1, full_mask.size)
    return {
        "dot_chain_score": float(score),
        "dot_component_count": int(components),
        "dot_horizontal_span": float(horizontal_span),
        "dot_component_area_ratio": float(area_ratio),
        "component_mask_area": component_mask_area,
        "dot_chain_fail": bool(fail),
        "component_mask": full_mask,
    }


def cleanup_residual_components_with_ring_fill(img: np.ndarray, component_mask: np.ndarray, det: Detection) -> np.ndarray | None:
    if component_mask is None or np.count_nonzero(component_mask) < 4:
        return None
    b = det.mark_box
    if det.contrast_span > REVIEW_CONTRAST_SPAN:
        return None
    img_h, img_w = img.shape[:2]
    x1 = max(0, b["x"] - max(10, b["w"] // 8))
    y1 = max(0, b["y"] - max(8, b["h"]))
    x2 = min(img_w, b["x"] + b["w"] + max(10, b["w"] // 8))
    y2 = min(img_h, b["y"] + b["h"] + max(8, b["h"]))
    inner = np.zeros(component_mask.shape, dtype=np.uint8)
    inner[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = 255
    window = np.zeros(component_mask.shape, dtype=np.uint8)
    window[y1:y2, x1:x2] = 255
    ring = cv2.bitwise_and(window, cv2.bitwise_not(inner))
    ring_pixels = img[ring > 0]
    if ring_pixels.size == 0:
        return None
    fill = np.median(ring_pixels.reshape(-1, 3), axis=0)
    mask = cv2.dilate((component_mask > 0).astype(np.uint8) * 255, np.ones((3, 5), np.uint8), iterations=1)
    if float(np.count_nonzero(mask)) / max(1, mask.size) > 0.012:
        return None
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32)[:, :, None] / 255.0
    cleaned = img.astype(np.float32) * (1.0 - alpha) + fill.reshape(1, 1, 3).astype(np.float32) * alpha
    return np.uint8(np.clip(cleaned, 0, 255))


def cleanup_residual_components_with_inpaint(
    img: np.ndarray,
    component_mask: np.ndarray,
    det: Detection,
    *,
    risky: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    if component_mask is None or np.count_nonzero(component_mask) < 4:
        return None, None, 0.0
    area_limit = RESIDUAL_CLEANUP_RISKY_AREA_MAX if risky else RESIDUAL_CLEANUP_AREA_MAX
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    mask = cv2.dilate((component_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
    if not risky:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8), iterations=1)
    area = float(np.count_nonzero(mask)) / max(1, mask.size)
    if area <= 0.0 or area > area_limit:
        return None, mask, area
    radius = 2 if risky else 3
    cleaned = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return cleaned, mask, area


def _mask_area(mask: np.ndarray | None) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / max(1, mask.size)


def _expanded_mark_bounds(shape: tuple[int, int], box: dict, pad_x: int, pad_y: int) -> tuple[int, int, int, int]:
    img_h, img_w = shape[:2]
    x1 = max(0, int(box["x"]) - pad_x)
    y1 = max(0, int(box["y"]) - pad_y)
    x2 = min(img_w, int(box["x"] + box["w"]) + pad_x)
    y2 = min(img_h, int(box["y"] + box["h"]) + pad_y)
    return x1, y1, x2, y2


def protected_product_edge_mask(original_bgr: np.ndarray, mark_box: dict) -> np.ndarray:
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    pad_x = max(8, int(round(mark_box["w"] * 0.16)))
    pad_y = max(6, int(round(mark_box["h"] * 0.65)))
    x1, y1, x2, y2 = _expanded_mark_bounds(gray.shape, mark_box, pad_x, pad_y)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return mask
    edges = cv2.Canny(roi, 55, 150)
    dark = (roi < 88).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    dark_lines = np.zeros_like(dark)
    for idx in range(1, count):
        _, _, ww, hh, area = stats[idx]
        if area < 4:
            continue
        line_like = ww >= max(8, roi.shape[1] * 0.08) or hh >= max(8, roi.shape[0] * 0.35)
        if line_like or area >= roi.size * 0.015:
            dark_lines[labels == idx] = 255
    protected = cv2.bitwise_or(edges, dark_lines)
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8), iterations=1)
    mask[y1:y2, x1:x2] = protected
    return mask


def _residual_text_component_mask(
    candidate_gray: np.ndarray,
    mark_box: dict,
    roi_class: str,
) -> tuple[np.ndarray, dict]:
    img_h, img_w = candidate_gray.shape[:2]
    pad_x = max(10, int(round(mark_box["w"] * 0.18)))
    pad_y = max(6, int(round(mark_box["h"] * 0.60)))
    x1, y1, x2, y2 = _expanded_mark_bounds(candidate_gray.shape, mark_box, pad_x, pad_y)
    roi = candidate_gray[y1:y2, x1:x2]
    full = np.zeros_like(candidate_gray)
    if roi.shape[0] < 8 or roi.shape[1] < 40:
        return full, {"component_count": 0, "component_area": 0.0, "component_span": 0.0}

    kernel = max(5, min(31, (roi.shape[0] // 2) * 2 + 1))
    background = cv2.medianBlur(roi, kernel)
    dev = cv2.absdiff(roi, background)
    percentile = 86 if roi_class in {"dark_product_surface", "thin_flex_cable", "complex_product_detail", "text_or_label_area"} else 80
    threshold = max(2.0, float(np.percentile(dev, percentile)))
    raw = (dev >= threshold).astype(np.uint8) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    xs: list[int] = []
    total_area = 0
    components = 0
    mark_center_y = (mark_box["y"] + mark_box["h"] / 2.0) - y1
    for idx in range(1, count):
        xx, yy, ww, hh, area = stats[idx]
        if area < 2 or area > roi.size * 0.060:
            continue
        if hh < 2 or hh > roi.shape[0] * 0.78:
            continue
        if ww > roi.shape[1] * 0.34 and area > roi.size * 0.012:
            continue
        _, cy = centroids[idx]
        if abs(float(cy) - mark_center_y) > max(roi.shape[0] * 0.42, mark_box["h"] * 0.90):
            continue
        kept[labels == idx] = 255
        xs.extend([xx, xx + ww])
        total_area += int(area)
        components += 1

    if components:
        kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8), iterations=1)
    full[y1:y2, x1:x2] = kept
    span = ((max(xs) - min(xs)) / max(1, roi.shape[1])) if xs else 0.0
    return full, {
        "component_count": int(components),
        "component_area": float(total_area / max(1, candidate_gray.size)),
        "component_span": float(span),
    }


def build_residual_cleanup_mask(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    initial_mask: np.ndarray,
    mark_box: dict,
    roi_class: str,
    detection: Detection,
    qa_metrics: dict,
) -> tuple[np.ndarray, dict]:
    """
    Build a small second-pass mask only around post-clean residual watermark
    components. Do not use this for product damage, visible bands, or unknown
    failure types.
    """
    shape = original_bgr.shape[:2]
    empty = np.zeros(shape, dtype=np.uint8)
    gate_meta = qa_metrics.get("gate_meta") or {}
    category = candidate_failure_category(gate_meta)
    residual_reasons = {"residual_visible", "dot_chain_residual", "post_clean_sunsky_detected", "alpha_template_residual"}
    reject_reasons = set(gate_meta.get("reject_reasons") or [])
    if category != "candidate_failed_residual_only" or not (reject_reasons & residual_reasons):
        return empty, {
            "eligible": False,
            "reason": "not_residual_only_failure",
            "category": category,
        }
    if detection.template.startswith("prior:") and detection.confidence < 0.72:
        return empty, {
            "eligible": False,
            "reason": "uncertain_detection",
            "category": category,
        }
    if initial_mask is None or np.count_nonzero(initial_mask) == 0:
        return empty, {
            "eligible": False,
            "reason": "missing_initial_mask",
            "category": category,
        }

    risky = roi_class in {"dark_product_surface", "thin_flex_cable", "complex_product_detail", "text_or_label_area"}
    dilate_x = RISKY_RESIDUAL_CLEANUP_DILATE_X if risky else RESIDUAL_CLEANUP_DILATE_X
    dilate_y = RISKY_RESIDUAL_CLEANUP_DILATE_Y if risky else RESIDUAL_CLEANUP_DILATE_Y
    multiplier = RISKY_RESIDUAL_CLEANUP_MAX_AREA_MULTIPLIER if risky else RESIDUAL_CLEANUP_MAX_AREA_MULTIPLIER
    max_pct = RESIDUAL_CLEANUP_RISKY_AREA_MAX if risky else RESIDUAL_CLEANUP_MAX_AREA_PCT
    initial_area = _mask_area(initial_mask)
    area_limit = min(max_pct, max(initial_area * multiplier, initial_area + 0.0012))

    b = mark_box
    pad_x = max(10, int(round(b["w"] * 0.18)))
    pad_y = max(6, int(round(b["h"] * 0.60)))
    x1, y1, x2, y2 = _expanded_mark_bounds(shape, b, pad_x, pad_y)
    window = np.zeros(shape, dtype=np.uint8)
    window[y1:y2, x1:x2] = 255

    candidate_gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    component_mask, component_meta = _residual_text_component_mask(candidate_gray, b, roi_class)
    diff_gray = cv2.absdiff(original_gray, candidate_gray)
    unchanged_residue = cv2.bitwise_and(component_mask, ((diff_gray < 28).astype(np.uint8) * 255))

    dot_metrics = qa_metrics.get("dot_metrics") or {}
    dot_mask = dot_metrics.get("component_mask")
    if isinstance(dot_mask, np.ndarray):
        dot_mask = cv2.bitwise_and(dot_mask, window)
    else:
        dot_mask = empty

    metrics = qa_metrics.get("metrics") or {}
    ocr_meta = qa_metrics.get("ocr_meta") or {}
    template_needed = (
        float(metrics.get("template_residual_score") or 0.0) > FINAL_TEMPLATE_MAX
        or float(metrics.get("residual_score") or 0.0) > FINAL_RESIDUAL_MAX
        or bool(ocr_meta.get("ocr_watermark"))
        or float(ocr_meta.get("ocr_watermark_score") or 0.0) >= OCR_POST_CLEAN_SUSPECT_MIN
    )
    template_halo = empty
    if template_needed:
        template_halo = create_glyph_halo_mask(
            canonical_ink_mask(),
            b,
            2 if risky else 3,
            1,
            shape,
        )
        template_halo = cv2.bitwise_and(template_halo, window)
        if (
            detection.template.startswith("ocr:")
            and detection.ocr_watermark_score >= OCR_CROP_LOCALIZE_MIN
            and roi_class != "text_or_label_area"
        ) or bool(ocr_meta.get("ocr_watermark")):
            tail_px = int(round(b["w"] * TAIL_EXPAND_RATIO_X))
            tail_px = max(TAIL_EXPAND_MIN_PX, min(TAIL_EXPAND_MAX_PX, tail_px))
            tail = cv2.dilate(template_halo, cv2.getStructuringElement(cv2.MORPH_RECT, (tail_px * 2 + 1, 3)), iterations=1)
            text_line = np.zeros(shape, dtype=np.uint8)
            y_line1 = max(0, b["y"] - max(2, b["h"] // 6))
            y_line2 = min(shape[0], b["y"] + b["h"] + max(2, b["h"] // 6))
            x_line1 = max(0, b["x"] - tail_px)
            x_line2 = min(shape[1], b["x"] + b["w"] + tail_px)
            text_line[y_line1:y_line2, x_line1:x_line2] = 255
            template_halo = cv2.bitwise_or(template_halo, cv2.bitwise_and(tail, text_line))

    initial_seed = cv2.bitwise_and((initial_mask > 0).astype(np.uint8) * 255, window)
    seed = cv2.bitwise_or(initial_seed, component_mask)
    seed = cv2.bitwise_or(seed, unchanged_residue)
    seed = cv2.bitwise_or(seed, dot_mask)
    seed = cv2.bitwise_or(seed, template_halo)
    seed = cv2.bitwise_and(seed, window)
    if np.count_nonzero(seed) < 4:
        return empty, {
            "eligible": False,
            "reason": "no_residual_components",
            "category": category,
            **component_meta,
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_x * 2 + 1, dilate_y * 2 + 1))
    cleanup = cv2.dilate(seed, kernel, iterations=1)
    cleanup = cv2.morphologyEx(cleanup, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8), iterations=1)
    cleanup = cv2.bitwise_and(cleanup, window)

    if risky:
        protected = protected_product_edge_mask(original_bgr, b)
        # Keep residual pixels on top of strong product edges out of the
        # second-pass mask. They remain visible in review instead of risking
        # cable or label destruction.
        cleanup = cv2.bitwise_and(cleanup, cv2.bitwise_not(protected))

    area = _mask_area(cleanup)
    if area > area_limit:
        seed = cv2.bitwise_or(component_mask, dot_mask)
        if bool(ocr_meta.get("ocr_watermark")):
            seed = cv2.bitwise_or(seed, template_halo)
        seed = cv2.bitwise_and(seed, window)
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate_x * 2 - 1), max(1, dilate_y * 2 - 1)))
        cleanup = cv2.dilate(seed, small_kernel, iterations=1)
        cleanup = cv2.bitwise_and(cleanup, window)
        area = _mask_area(cleanup)

    if area <= 0.0 or area > area_limit:
        return empty, {
            "eligible": False,
            "reason": "cleanup_mask_area_limit",
            "category": category,
            "cleanup_area_pct": area * 100,
            "area_limit_pct": area_limit * 100,
            **component_meta,
        }

    return cleanup, {
        "eligible": True,
        "reason": "residual_evidence_mask",
        "category": category,
        "cleanup_area_pct": area * 100,
        "area_limit_pct": area_limit * 100,
        "initial_mask_area_pct": initial_area * 100,
        "risky_roi": bool(risky),
        "dilate_x": dilate_x,
        "dilate_y": dilate_y,
        "template_halo_used": bool(template_needed),
        "dot_component_area_pct": _mask_area(dot_mask) * 100,
        **component_meta,
    }


def _context_ring(mask_shape: tuple[int, int], mark_box: dict, mask: np.ndarray, pad_x: int, pad_y: int) -> np.ndarray:
    x1, y1, x2, y2 = _expanded_mark_bounds(mask_shape, mark_box, pad_x, pad_y)
    window = np.zeros(mask_shape, dtype=np.uint8)
    window[y1:y2, x1:x2] = 255
    expanded = cv2.dilate((mask > 0).astype(np.uint8) * 255, np.ones((7, 15), np.uint8), iterations=1)
    return cv2.bitwise_and(window, cv2.bitwise_not(expanded))


def _blend_repair(base_bgr: np.ndarray, fill_bgr: np.ndarray, mask: np.ndarray, sigma: float = 0.85) -> np.ndarray:
    alpha = cv2.GaussianBlur((mask > 0).astype(np.uint8) * 255, (0, 0), sigma).astype(np.float32)[:, :, None] / 255.0
    repaired = base_bgr.astype(np.float32) * (1.0 - alpha) + fill_bgr.astype(np.float32) * alpha
    return np.uint8(np.clip(repaired, 0, 255))


def white_or_near_white_row_fill(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    cleanup_mask: np.ndarray,
    det: Detection,
) -> tuple[np.ndarray | None, dict]:
    if cleanup_mask is None or np.count_nonzero(cleanup_mask) < 4:
        return None, {"operator": "white_or_near_white_row_fill", "reason": "empty_mask"}
    gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    b = det.mark_box
    pad_x = max(18, int(round(b["w"] * 0.22)))
    pad_y = max(14, int(round(b["h"] * 1.20)))
    ring = _context_ring(gray.shape, b, cleanup_mask, pad_x, pad_y)
    bright_ring = cv2.bitwise_and(ring, ((gray >= 190).astype(np.uint8) * 255))
    if np.count_nonzero(bright_ring) < 24:
        return None, {"operator": "white_or_near_white_row_fill", "reason": "insufficient_bright_context"}

    fill = candidate_bgr.astype(np.float32).copy()
    ys, xs = np.where(cleanup_mask > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    ring_pixels = candidate_bgr[bright_ring > 0].reshape(-1, 3)
    global_fill = np.median(ring_pixels, axis=0).astype(np.float32)
    global_noise = np.clip(np.std(ring_pixels, axis=0), 0.0, 2.0)
    for y in range(y_min, y_max + 1):
        row1 = max(0, y - 2)
        row2 = min(gray.shape[0], y + 3)
        row_ctx = bright_ring[row1:row2, :]
        pixels = candidate_bgr[row1:row2, :][row_ctx > 0]
        row_fill = np.median(pixels.reshape(-1, 3), axis=0).astype(np.float32) if pixels.size >= 18 else global_fill
        cols = np.where(cleanup_mask[y, :] > 0)[0]
        if not len(cols):
            continue
        pseudo = (((cols * 17 + y * 31 + b["x"] * 7) % 23) - 11).astype(np.float32)[:, None] / 11.0
        fill[y, cols] = row_fill.reshape(1, 3) + pseudo * global_noise.reshape(1, 3)

    repaired = _blend_repair(candidate_bgr, np.uint8(np.clip(fill, 0, 255)), cleanup_mask, sigma=0.75)
    return repaired, {
        "operator": "white_or_near_white_row_fill",
        "repair_mask_area_pct": _mask_area(cleanup_mask) * 100,
        "context_pixels": int(np.count_nonzero(bright_ring)),
    }


def dark_surface_low_alpha_scrub(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    cleanup_mask: np.ndarray,
    det: Detection,
) -> tuple[np.ndarray | None, dict]:
    if cleanup_mask is None or np.count_nonzero(cleanup_mask) < 4:
        return None, {"operator": "dark_surface_low_alpha_scrub", "reason": "empty_mask"}
    gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    b = det.mark_box
    pad_x = max(12, int(round(b["w"] * 0.18)))
    pad_y = max(8, int(round(b["h"] * 0.85)))
    ring = _context_ring(gray.shape, b, cleanup_mask, pad_x, pad_y)
    dark_ring = cv2.bitwise_and(ring, ((gray <= 170).astype(np.uint8) * 255))
    protected = protected_product_edge_mask(original_bgr, b)
    allowed = cv2.bitwise_and(cleanup_mask, cv2.bitwise_not(protected))
    if np.count_nonzero(allowed) < 4:
        return None, {"operator": "dark_surface_low_alpha_scrub", "reason": "protected_all_residual_pixels"}
    if np.count_nonzero(dark_ring) < 18:
        dark_ring = ring
    pixels = candidate_bgr[dark_ring > 0].reshape(-1, 3)
    if pixels.size == 0:
        return None, {"operator": "dark_surface_low_alpha_scrub", "reason": "insufficient_context"}
    fill_color = np.median(pixels, axis=0).astype(np.float32)
    noise_sigma = np.clip(np.std(pixels, axis=0), 0.0, 3.5)
    yy, xx = np.indices(gray.shape)
    pseudo = (((xx * 19 + yy * 23 + b["y"] * 11) % 29) - 14).astype(np.float32) / 14.0
    fill = np.zeros_like(candidate_bgr, dtype=np.float32)
    fill[:, :] = fill_color.reshape(1, 1, 3) + pseudo[:, :, None] * noise_sigma.reshape(1, 1, 3)
    repaired = _blend_repair(candidate_bgr, np.uint8(np.clip(fill, 0, 255)), allowed, sigma=0.70)
    return repaired, {
        "operator": "dark_surface_low_alpha_scrub",
        "dark_surface_scrub_used": True,
        "repair_mask_area_pct": _mask_area(allowed) * 100,
        "protected_edge_area_pct": _mask_area(protected) * 100,
    }


def thin_flex_cable_protected_cleanup(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    cleanup_mask: np.ndarray,
    det: Detection,
) -> tuple[np.ndarray | None, dict]:
    if cleanup_mask is None or np.count_nonzero(cleanup_mask) < 4:
        return None, {"operator": "thin_flex_cable_protected_cleanup", "reason": "empty_mask"}
    b = det.mark_box
    protected = protected_product_edge_mask(original_bgr, b)
    allowed = cv2.bitwise_and(cleanup_mask, cv2.bitwise_not(cv2.dilate(protected, np.ones((3, 3), np.uint8), iterations=1)))
    if np.count_nonzero(allowed) < 4:
        return None, {"operator": "thin_flex_cable_protected_cleanup", "reason": "protected_all_residual_pixels"}
    allowed = cv2.morphologyEx(allowed, cv2.MORPH_OPEN, np.ones((2, 3), np.uint8), iterations=1)
    if np.count_nonzero(allowed) < 4:
        return None, {"operator": "thin_flex_cable_protected_cleanup", "reason": "no_low_contrast_residue"}
    repaired = cv2.inpaint(candidate_bgr, allowed, 2, cv2.INPAINT_NS)
    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    rep_gray = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY)
    orig_edges = cv2.bitwise_and(cv2.Canny(orig_gray, 55, 150), protected)
    rep_edges = cv2.bitwise_and(cv2.Canny(rep_gray, 55, 150), protected)
    protected_edge_loss = 1.0 - (
        np.count_nonzero(cv2.bitwise_and(orig_edges, rep_edges)) / max(1, np.count_nonzero(orig_edges))
    )
    x1, y1, x2, y2 = _expanded_mark_bounds(orig_gray.shape, b, max(8, b["w"] // 8), max(6, b["h"] // 2))
    orig_sil = orig_gray[y1:y2, x1:x2] < 95
    rep_sil = rep_gray[y1:y2, x1:x2] < 95
    cable_silhouette_delta = float(np.mean(orig_sil != rep_sil)) if orig_sil.size else 0.0
    return repaired, {
        "operator": "thin_flex_cable_protected_cleanup",
        "repair_mask_area_pct": _mask_area(allowed) * 100,
        "protected_edge_loss": float(protected_edge_loss),
        "cable_silhouette_delta": cable_silhouette_delta,
    }


def solid_color_surface_fill(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    cleanup_mask: np.ndarray,
    det: Detection,
) -> tuple[np.ndarray | None, dict]:
    if cleanup_mask is None or np.count_nonzero(cleanup_mask) < 4:
        return None, {"operator": "solid_color_surface_fill", "reason": "empty_mask"}
    gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    b = det.mark_box
    pad_x = max(16, int(round(b["w"] * 0.22)))
    pad_y = max(12, int(round(b["h"] * 1.10)))
    ring = _context_ring(gray.shape, b, cleanup_mask, pad_x, pad_y)
    if np.count_nonzero(ring) < 36:
        return None, {"operator": "solid_color_surface_fill", "reason": "insufficient_context"}
    ring_gray = gray[ring > 0]
    edge_density = float(np.mean(cv2.Canny(gray, 55, 150)[ring > 0] > 0)) if ring_gray.size else 1.0
    ring_pixels = candidate_bgr[ring > 0].reshape(-1, 3)
    color_std = float(np.mean(np.std(ring_pixels, axis=0))) if ring_pixels.size else 255.0
    if edge_density > 0.12 or color_std > 46.0:
        return None, {
            "operator": "solid_color_surface_fill",
            "reason": "not_solid_color_plane",
            "edge_density": edge_density,
            "color_std": color_std,
        }

    ys, xs = np.where(ring > 0)
    if len(xs) > 5000:
        step = max(1, len(xs) // 5000)
        xs = xs[::step]
        ys = ys[::step]
    design = np.stack([xs.astype(np.float32), ys.astype(np.float32), np.ones_like(xs, dtype=np.float32)], axis=1)
    fill = candidate_bgr.astype(np.float32).copy()
    mask_ys, mask_xs = np.where(cleanup_mask > 0)
    mask_design = np.stack(
        [mask_xs.astype(np.float32), mask_ys.astype(np.float32), np.ones_like(mask_xs, dtype=np.float32)],
        axis=1,
    )
    for channel in range(3):
        values = candidate_bgr[ys, xs, channel].astype(np.float32)
        coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
        predicted = mask_design @ coeff
        fill[mask_ys, mask_xs, channel] = predicted
    repaired = _blend_repair(candidate_bgr, np.uint8(np.clip(fill, 0, 255)), cleanup_mask, sigma=0.80)
    return repaired, {
        "operator": "solid_color_surface_fill",
        "repair_mask_area_pct": _mask_area(cleanup_mask) * 100,
        "edge_density": edge_density,
        "color_std": color_std,
    }


def repeated_object_neighbor_clone(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    cleanup_mask: np.ndarray,
    det: Detection,
) -> tuple[np.ndarray | None, dict]:
    if cleanup_mask is None or np.count_nonzero(cleanup_mask) < 4:
        return None, {"operator": "repeated_object_neighbor_clone", "reason": "empty_mask"}
    ys, xs = np.where(cleanup_mask > 0)
    x1, x2 = max(0, int(xs.min()) - 6), min(candidate_bgr.shape[1], int(xs.max()) + 7)
    y1, y2 = max(0, int(ys.min()) - 6), min(candidate_bgr.shape[0], int(ys.max()) + 7)
    patch = candidate_bgr[y1:y2, x1:x2]
    if patch.shape[0] < 10 or patch.shape[1] < 10 or patch.size > candidate_bgr.size * 0.08:
        return None, {"operator": "repeated_object_neighbor_clone", "reason": "patch_size_not_supported"}
    gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    tpl = cv2.Canny(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 45, 130)
    scene = cv2.Canny(gray, 45, 130)
    if tpl.shape[0] >= scene.shape[0] or tpl.shape[1] >= scene.shape[1]:
        return None, {"operator": "repeated_object_neighbor_clone", "reason": "patch_larger_than_scene"}
    result = cv2.matchTemplate(scene, tpl, cv2.TM_CCOEFF_NORMED)
    ignore_x1 = max(0, x1 - patch.shape[1])
    ignore_y1 = max(0, y1 - patch.shape[0])
    ignore_x2 = min(result.shape[1], x2 + patch.shape[1])
    ignore_y2 = min(result.shape[0], y2 + patch.shape[0])
    result[ignore_y1:ignore_y2, ignore_x1:ignore_x2] = -1.0
    _, score, _, loc = cv2.minMaxLoc(result)
    if score < 0.94:
        return None, {"operator": "repeated_object_neighbor_clone", "reason": "low_similarity", "similarity": float(score)}
    sx, sy = loc
    donor = candidate_bgr[sy:sy + patch.shape[0], sx:sx + patch.shape[1]]
    if donor.shape != patch.shape:
        return None, {"operator": "repeated_object_neighbor_clone", "reason": "invalid_donor"}
    fill = candidate_bgr.copy()
    local_mask = cleanup_mask[y1:y2, x1:x2]
    fill[y1:y2, x1:x2][local_mask > 0] = donor[local_mask > 0]
    repaired = _blend_repair(candidate_bgr, fill, cleanup_mask, sigma=0.65)
    return repaired, {
        "operator": "repeated_object_neighbor_clone",
        "repair_mask_area_pct": _mask_area(cleanup_mask) * 100,
        "similarity": float(score),
        "donor_box": {"x": int(sx), "y": int(sy), "w": int(patch.shape[1]), "h": int(patch.shape[0])},
    }


def roi_specific_repair_candidates(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    cleanup_mask: np.ndarray,
    det: Detection,
) -> list[tuple[str, np.ndarray, np.ndarray, dict]]:
    operators = []
    if det.roi_class in {"plain_white", "near_white", "low_texture_background"}:
        operators.append(white_or_near_white_row_fill)
    if det.roi_class in {"dark_product_surface"}:
        operators.append(dark_surface_low_alpha_scrub)
    if det.roi_class == "thin_flex_cable":
        operators.append(thin_flex_cable_protected_cleanup)
    if det.roi_class in {"low_texture_background", "simple_product_surface", "near_white", "plain_white", "dark_product_surface"}:
        operators.append(solid_color_surface_fill)
    operators.append(repeated_object_neighbor_clone)

    repaired: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    seen = set()
    for operator in operators:
        if operator.__name__ in seen:
            continue
        seen.add(operator.__name__)
        image, meta = operator(original_bgr, candidate_bgr, cleanup_mask, det)
        if image is None:
            continue
        repaired.append((operator.__name__, image, cleanup_mask, meta))
    return repaired


def near_area_background_fill_repair(
    img: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    det: Detection,
    *,
    risky: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """Repair watermark pixels on plain background by copying nearby background.

    Telea/NS often creates visible smears on pure white or low-texture product
    photos. For those pixels, the right operation is simpler: estimate the
    nearby background from the context ring and paste it back with a feather.
    Product-overlap pixels are left to a tiny inpaint pass, not a white fill.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return None, None, 0.0, 0.0
    img_h, img_w = gray.shape[:2]
    b = det.mark_box
    pad_x = max(18, int(round(b["w"] * 0.24)))
    pad_y = max(16, int(round(b["h"] * 1.80)))
    x1 = max(0, b["x"] - pad_x)
    y1 = max(0, b["y"] - pad_y)
    x2 = min(img_w, b["x"] + b["w"] + pad_x)
    y2 = min(img_h, b["y"] + b["h"] + pad_y)

    window = np.zeros(gray.shape, dtype=np.uint8)
    window[y1:y2, x1:x2] = 255
    expanded_mask = cv2.dilate((mask > 0).astype(np.uint8) * 255, np.ones((7, 15), np.uint8), iterations=1)
    ring = cv2.bitwise_and(window, cv2.bitwise_not(expanded_mask))
    if np.count_nonzero(ring) < 30:
        return None, None, 0.0, 0.0

    background = cv2.medianBlur(gray, max(9, min(41, (max(5, b["h"] * 2) // 2) * 2 + 1)))
    bg_threshold = 216 if risky else 208
    bg_target = cv2.bitwise_and(mask, ((background >= bg_threshold).astype(np.uint8) * 255))
    bg_target_area = int(np.count_nonzero(bg_target))
    if bg_target_area < max(8, int(np.count_nonzero(mask) * 0.18)):
        return None, None, 0.0, 0.0

    ring_gray = gray[ring > 0]
    bright_ring = cv2.bitwise_and(ring, ((gray >= bg_threshold).astype(np.uint8) * 255))
    if np.count_nonzero(bright_ring) >= 24:
        ring_pixels = img[bright_ring > 0]
    elif det.roi_class in {"low_texture_background", "near_white", "plain_white"}:
        ring_pixels = img[ring > 0]
    else:
        return None, None, 0.0, 0.0
    if ring_pixels.size == 0:
        return None, None, 0.0, 0.0

    fill = np.median(ring_pixels.reshape(-1, 3), axis=0).astype(np.float32)
    noise_sigma = np.clip(np.std(ring_pixels.reshape(-1, 3), axis=0), 0.0, 2.5)
    fill_img = np.zeros_like(img, dtype=np.float32)
    fill_img[:, :] = fill.reshape(1, 1, 3)
    if float(np.max(noise_sigma)) > 0.2:
        yy, xx = np.indices(gray.shape)
        pseudo = (((xx * 17 + yy * 31 + b["x"] * 7 + b["y"] * 13) % 23) - 11).astype(np.float32) / 11.0
        fill_img += pseudo[:, :, None] * noise_sigma.reshape(1, 1, 3)

    bg_target = cv2.dilate(bg_target, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=1)
    alpha = cv2.GaussianBlur(bg_target, (0, 0), 0.9).astype(np.float32)[:, :, None] / 255.0
    candidate = img.astype(np.float32) * (1.0 - alpha) + fill_img * alpha
    candidate = np.uint8(np.clip(candidate, 0, 255))

    foreground_mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg_target))
    foreground_area = float(np.count_nonzero(foreground_mask)) / max(1, mask.size)
    if np.count_nonzero(foreground_mask) >= 4 and foreground_area <= (0.0045 if risky else 0.010):
        candidate = cv2.inpaint(candidate, foreground_mask, 2 if risky else 3, cv2.INPAINT_TELEA)
        effective_mask = cv2.bitwise_or(bg_target, foreground_mask)
    else:
        effective_mask = bg_target

    area = float(np.count_nonzero(effective_mask)) / max(1, effective_mask.size)
    bg_fraction = bg_target_area / max(1, np.count_nonzero(mask))
    if area > (0.014 if risky else 0.026):
        return None, effective_mask, area, bg_fraction
    return candidate, effective_mask, area, float(bg_fraction)


def evaluate_cleaned_output(
    original: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    det: Detection,
    templates: list[TemplateSpec] | None,
    ocr_reader,
) -> tuple[np.ndarray, dict, int | None, dict, dict, dict, dict, dict, dict]:
    cgray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    metrics = residual_quality_metrics(cgray, det, templates)
    post_count = post_clean_detection_count(candidate, cgray, templates)
    ocr_meta = cleaned_crop_ocr_check(candidate, det, ocr_reader)
    dot_metrics = residual_component_metrics(cgray, det)
    band_metrics = detect_rectangular_band_visibility(original, candidate, mask)
    product_metrics = detect_product_damage_v13(original, candidate, mask, det)
    alpha_metrics = alpha_quality_metrics(original, candidate, det)
    gate_meta = final_publish_gate(metrics, post_count, ocr_meta, dot_metrics, band_metrics, product_metrics, alpha_metrics)
    return cgray, metrics, post_count, ocr_meta, dot_metrics, band_metrics, product_metrics, alpha_metrics, gate_meta


def detect_rectangular_band_visibility(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None) -> dict:
    if mask is None or np.count_nonzero(mask) == 0:
        return {
            "visible_band_score": 0.0,
            "band_luma_delta": 0.0,
            "band_edge_box": 0.0,
            "band_gate_fail": False,
        }
    diff = cv2.absdiff(original, candidate)
    luma = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(mask > 0)
    y1, y2 = max(0, ys.min() - 4), min(mask.shape[0], ys.max() + 5)
    x1, x2 = max(0, xs.min() - 4), min(mask.shape[1], xs.max() + 5)
    roi = luma[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "visible_band_score": 0.0,
            "band_luma_delta": 0.0,
            "band_edge_box": 0.0,
            "band_gate_fail": False,
        }
    band_luma_delta = float(np.mean(roi))
    top = float(np.mean(roi[:2, :])) if roi.shape[0] >= 4 else 0.0
    bottom = float(np.mean(roi[-2:, :])) if roi.shape[0] >= 4 else 0.0
    left = float(np.mean(roi[:, :2])) if roi.shape[1] >= 4 else 0.0
    right = float(np.mean(roi[:, -2:])) if roi.shape[1] >= 4 else 0.0
    center = float(np.mean(roi[2:-2, 2:-2])) if roi.shape[0] > 5 and roi.shape[1] > 5 else band_luma_delta
    edge_box = max(top, bottom, left, right) / max(center, 1.0)
    visible_score = min(1.0, band_luma_delta / 32.0) * 0.62 + min(1.0, edge_box / 2.5) * 0.38
    fail = visible_score > VISIBLE_BAND_SCORE_MAX and band_luma_delta > VISIBLE_BAND_LUMA_DELTA_MAX and edge_box > 1.20
    return {
        "visible_band_score": float(visible_score),
        "band_luma_delta": band_luma_delta,
        "band_edge_box": float(edge_box),
        "band_gate_fail": bool(fail),
    }


def detect_product_damage_v13(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None, det: Detection) -> dict:
    if mask is None or np.count_nonzero(mask) == 0:
        return {
            "product_gate_fail": False,
            "product_color_delta": 0.0,
            "product_edge_retention": 1.0,
            "product_blob_score": 0.0,
            "product_changed_area_ratio": 0.0,
        }
    if det.product_overlap < 0.30 and det.roi_class not in {"dark_product_surface", "thin_flex_cable", "complex_product_detail", "text_or_label_area"}:
        return {
            "product_gate_fail": False,
            "product_color_delta": 0.0,
            "product_edge_retention": 1.0,
            "product_blob_score": 0.0,
            "product_changed_area_ratio": 0.0,
        }

    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cand_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(orig_gray, cand_gray)
    product_mask = ((orig_gray < 235) | (cv2.Canny(orig_gray, 55, 150) > 0)).astype(np.uint8) * 255
    protect = cv2.bitwise_and(product_mask, cv2.dilate(mask, np.ones((5, 9), np.uint8), iterations=1))
    changed = cv2.bitwise_and(((diff_gray > 10).astype(np.uint8) * 255), protect)
    changed_count = int(np.count_nonzero(changed))
    if changed_count == 0:
        return {
            "product_gate_fail": False,
            "product_color_delta": 0.0,
            "product_edge_retention": 1.0,
            "product_blob_score": 0.0,
            "product_changed_area_ratio": 0.0,
        }

    orig_edges = cv2.bitwise_and(cv2.Canny(orig_gray, 55, 150), protect)
    cand_edges = cv2.bitwise_and(cv2.Canny(cand_gray, 55, 150), protect)
    edge_retention = float(np.count_nonzero(cv2.bitwise_and(orig_edges, cand_edges)) / max(1, np.count_nonzero(orig_edges)))
    product_color_delta = float(np.mean(diff_gray[changed > 0]))
    orig_i = orig_gray.astype(np.int16)
    cand_i = cand_gray.astype(np.int16)
    dark_surface = orig_i < 110
    bright_blob = np.logical_and.reduce((changed > 0, dark_surface, cand_i > orig_i + 42))
    dark_blob = np.logical_and(changed > 0, cand_i + 42 < orig_i)
    blob_score = float((np.count_nonzero(bright_blob) + np.count_nonzero(dark_blob)) / max(1, changed_count))
    changed_area_ratio = float(changed_count / max(1, np.count_nonzero(mask)))
    contour_break = max(0.0, 1.0 - edge_retention)
    fail = (
        (det.roi_class in {"dark_product_surface", "thin_flex_cable"} and blob_score > 0.10)
        or (det.product_overlap >= 0.45 and product_color_delta > 34.0 and edge_retention < 0.72)
        or (det.product_overlap >= 0.55 and contour_break > 0.42 and changed_area_ratio > 0.38)
    )
    return {
        "product_gate_fail": bool(fail),
        "product_color_delta": product_color_delta,
        "product_edge_retention": edge_retention,
        "product_blob_score": blob_score,
        "product_changed_area_ratio": changed_area_ratio,
    }


def alpha_quality_metrics(original: np.ndarray, candidate: np.ndarray, det: Detection) -> dict:
    engine = sunsky_alpha_engine()
    if engine is None or not engine.alpha_available() or engine.alpha is None:
        return {
            "alpha_checked": False,
            "alpha_template_residual_before": 0.0,
            "alpha_template_residual_after": 0.0,
            "alpha_residual_reduction": 0.0,
        }
    mark = mark_box_tuple(det)
    before = score_alpha_residual(original, mark, engine.alpha)
    after = score_alpha_residual(candidate, mark, engine.alpha)
    reduction = 0.0 if before <= 1e-6 else max(0.0, (before - after) / before)
    return {
        "alpha_checked": True,
        "alpha_template_residual_before": float(before),
        "alpha_template_residual_after": float(after),
        "alpha_residual_reduction": float(reduction),
    }


def final_publish_gate(
    metrics: dict,
    post_count: int | None,
    ocr_meta: dict,
    dot_metrics: dict,
    band_metrics: dict,
    product_metrics: dict | None = None,
    alpha_metrics: dict | None = None,
) -> dict:
    product_metrics = product_metrics or {"product_gate_fail": False}
    alpha_metrics = alpha_metrics or {"alpha_checked": False}
    required = ["residual_score", "template_residual_score", "post_text_score", "post_text_components"]
    metrics_valid = all(isinstance(metrics.get(key), (int, float)) for key in required)
    residual_pass = (
        metrics_valid
        and float(metrics["residual_score"]) <= FINAL_RESIDUAL_MAX
        and float(metrics["template_residual_score"]) <= FINAL_TEMPLATE_MAX
        and int(metrics["post_text_components"]) <= FINAL_TEXT_COMPONENTS_MAX
        and (post_count == 0)
    )
    post_ocr_score = float(ocr_meta.get("ocr_watermark_score") or 0.0)
    sunsky_check_pass = (
        not bool(ocr_meta.get("ocr_watermark"))
        and post_ocr_score < OCR_POST_CLEAN_SUSPECT_MIN
    )
    dot_pass = not bool(dot_metrics.get("dot_chain_fail"))
    band_pass = not bool(band_metrics.get("band_gate_fail"))
    product_pass = not bool(product_metrics.get("product_gate_fail"))
    alpha_checked = bool(alpha_metrics.get("alpha_checked"))
    alpha_pass = True
    if alpha_checked:
        alpha_before = float(alpha_metrics.get("alpha_template_residual_before") or 0.0)
        alpha_after = float(alpha_metrics.get("alpha_template_residual_after") or 0.0)
        alpha_reduction = float(alpha_metrics.get("alpha_residual_reduction") or 0.0)
        alpha_pass = (
            alpha_after <= FINAL_ALPHA_TEMPLATE_MAX
            and (
                alpha_before <= FINAL_ALPHA_TEMPLATE_MAX
                or alpha_reduction >= MIN_ALPHA_RESIDUAL_REDUCTION
            )
        )
    reject_reasons = []
    if not metrics_valid:
        reject_reasons.append("missing_required_qa_metric")
    if not residual_pass:
        reject_reasons.append("residual_visible")
    if not dot_pass:
        reject_reasons.append("dot_chain_residual")
    if not band_pass:
        reject_reasons.append("visible_rectangular_band")
    if not product_pass:
        reject_reasons.append("product_damage")
    if not sunsky_check_pass:
        reject_reasons.append("post_clean_sunsky_detected")
    if not alpha_pass:
        reject_reasons.append("alpha_template_residual")
    publish_ok = metrics_valid and residual_pass and sunsky_check_pass and dot_pass and band_pass and product_pass and alpha_pass
    return {
        "publish_ok": bool(publish_ok),
        "status": "cleaned" if publish_ok else "needs_manual",
        "reject_reasons": reject_reasons,
        "metrics_valid": bool(metrics_valid),
        "residual_pass": bool(residual_pass),
        "sunsky_check_pass": bool(sunsky_check_pass),
        "post_clean_ocr_score": post_ocr_score,
        "dot_chain_pass": bool(dot_pass),
        "band_pass": bool(band_pass),
        "product_gate_pass": bool(product_pass),
        "alpha_template_pass": bool(alpha_pass),
    }


def candidate_failure_category(gate_meta: dict | None) -> str:
    if not gate_meta:
        return "candidate_failed_metrics_invalid"
    if gate_meta.get("publish_ok"):
        return "candidate_passed"
    reasons = set(gate_meta.get("reject_reasons") or [])
    if "missing_required_qa_metric" in reasons:
        return "candidate_failed_metrics_invalid"
    if "product_damage" in reasons:
        return "candidate_failed_product_damage"
    if "visible_rectangular_band" in reasons:
        return "candidate_failed_band"
    residual_reasons = {"residual_visible", "dot_chain_residual", "post_clean_sunsky_detected"}
    if reasons and reasons.issubset(residual_reasons):
        return "candidate_failed_residual_only"
    if reasons & residual_reasons and not (reasons - residual_reasons):
        return "candidate_failed_residual_only"
    return "candidate_failed_detection"


def final_blocker_type(gate_meta: dict | None) -> str:
    category = candidate_failure_category(gate_meta)
    return {
        "candidate_passed": "none",
        "candidate_failed_residual_only": "residual_watermark",
        "candidate_failed_band": "visible_band",
        "candidate_failed_product_damage": "product_damage",
        "candidate_failed_detection": "detection_or_policy",
        "candidate_failed_metrics_invalid": "metrics_invalid",
    }.get(category, "unknown")


def candidate_trace(candidate: RepairCandidate) -> dict:
    gate_meta = candidate.gate_meta or {}
    return {
        "candidate_id": candidate.id,
        "strategy": candidate.strategy,
        "category": candidate.category,
        "rank_score": round(float(candidate.rank_score), 5),
        "residual_score": round(float(candidate.residual), 4),
        "template_residual_score": round(float(candidate.template_residual), 4),
        "sharpness_ratio": round(float(candidate.sharpness_ratio), 4),
        "mask_area_pct": round(float(candidate.mask_area) * 100, 3),
        "second_pass": bool(candidate.second_pass),
        "second_pass_strategy": candidate.second_pass_strategy,
        "second_pass_mask_area_pct": round(float(candidate.second_pass_mask_area) * 100, 3),
        "reject_reasons": list(gate_meta.get("reject_reasons") or []),
        "publish_ok": bool(gate_meta.get("publish_ok")),
        "band_fail": bool((candidate.band_metrics or {}).get("band_gate_fail")),
        "product_fail": bool((candidate.product_metrics or {}).get("product_gate_fail")),
        "post_clean_detection_count": candidate.post_count,
        "post_clean_ocr_score": round(float((candidate.gate_meta or {}).get("post_clean_ocr_score") or 0.0), 4),
        "post_text_components": int(candidate.metrics.get("post_text_components") or 0),
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
    pad_x = max(12, int(round(det.mark_box["w"] * 0.22)))
    pad_y = max(8, int(round(det.mark_box["h"] * 1.00)))
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
    risky_product_roi = (
        det.product_overlap >= 0.42
        or det.roi_class in {"dark_product_surface", "thin_flex_cable", "complex_product_detail", "text_or_label_area"}
        or det.layout_risk
    )
    if risky_product_roi:
        variants = [
            ("glyph_tight", 8, 5, 2, 3, True),
            ("glyph_medium", 11, 6, 3, 4, True),
            ("glyph_strong", 13, 7, 4, 5, True),
        ]
    elif high_contrast_mask:
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
    alpha_candidate_count = 0
    alpha_asset_used = False

    engine = sunsky_alpha_engine()
    alpha_allowed = (
        engine is not None
        and engine.alpha_available()
        and det.mark_box
        and (det.mask_area_pct / 100.0) <= MAX_MASK_AREA
        and not (det.template.startswith("prior:") and det.confidence < 0.72)
    )
    if alpha_allowed and engine is not None:
        alpha_asset_used = True
        for idx, alpha_candidate in enumerate(
            engine.reverse_alpha_candidates(img, mark_box_tuple(det), roi_class=det.roi_class)
        ):
            alpha_mask = (alpha_candidate.alpha_map >= ALPHA_FLOOR).astype(np.uint8) * 255
            alpha_area = float(np.count_nonzero(alpha_mask)) / max(1, alpha_mask.size)
            area_limit = RESIDUAL_CLEANUP_RISKY_AREA_MAX if risky_product_roi else PILOT_MASK_AREA
            if alpha_area <= 0.0 or alpha_area > area_limit:
                continue
            cgray = cv2.cvtColor(alpha_candidate.image, cv2.COLOR_BGR2GRAY)
            sharpness_ratio = laplacian_var(cgray, alpha_mask) / max(laplacian_var(gray, alpha_mask), 1e-6)
            metrics = residual_quality_metrics(cgray, det, templates)
            residual = metrics["residual_score"]
            template_residual = metrics["template_residual_score"]
            cost = (
                min(1.0, residual) * 3.0
                + min(1.0, template_residual) * 3.0
                + float(alpha_candidate.residual_probe_score) * 2.0
                + metrics["post_text_components"] * 0.20
                + alpha_area * 5.0
                - min(0.35, float(alpha_candidate.alignment_score) * 0.20)
            )
            extra = {
                "alpha_engine_used": True,
                "alpha_candidate_index": idx,
                "alpha_alignment_score": float(alpha_candidate.alignment_score),
                "alpha_best_gain": float(alpha_candidate.alpha_gain),
                "alpha_best_logo_bgr": [float(v) for v in alpha_candidate.logo_bgr],
                "alpha_residual_probe_score": float(alpha_candidate.residual_probe_score),
                "thin_residual_inpaint": alpha_candidate.name.endswith("_thin_ns"),
                "alpha_bbox": {
                    "x": int(alpha_candidate.glyph_bbox[0]),
                    "y": int(alpha_candidate.glyph_bbox[1]),
                    "w": int(alpha_candidate.glyph_bbox[2]),
                    "h": int(alpha_candidate.glyph_bbox[3]),
                },
            }
            candidates.append((
                -cost,
                residual,
                template_residual,
                sharpness_ratio,
                alpha_area,
                alpha_candidate.name,
                alpha_candidate.image,
                alpha_mask,
                metrics,
                extra,
            ))
            alpha_candidate_count += 1
            if alpha_candidate_count >= 12:
                break

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
        near_candidate, near_mask, near_area, bg_fraction = near_area_background_fill_repair(
            img,
            gray,
            mask,
            det,
            risky=risky_product_roi,
        )
        if near_candidate is not None and near_mask is not None:
            cgray = cv2.cvtColor(near_candidate, cv2.COLOR_BGR2GRAY)
            sharpness_ratio = laplacian_var(cgray, near_mask) / max(laplacian_var(gray, near_mask), 1e-6)
            metrics = residual_quality_metrics(cgray, det, templates)
            residual = metrics["residual_score"]
            template_residual = metrics["template_residual_score"]
            cost = (
                min(1.0, residual)
                + min(1.0, template_residual) * 0.18
                + near_area * 6.0
                + max(0.0, 0.12 - sharpness_ratio) * 0.30
                - min(0.18, bg_fraction * 0.10)
            )
            candidates.append((
                -cost,
                residual,
                template_residual,
                sharpness_ratio,
                near_area,
                f"{variant}_near_area_fill",
                near_candidate,
                near_mask,
                metrics,
                {},
            ))
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
                {},
            ))

    if not candidates:
        return None, first_mask, {
            "status": "needs_manual",
            "reason": "mask_too_large" if first_mask is None else "mask_area_review",
            "mask_area_pct": round(first_area * 100, 3),
        }

    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = candidates[0]
    selected_eval: dict | None = None
    candidate_traces = []
    selected_candidate_id = "cand_00"
    for cand in candidates[:MAX_TOTAL_REPAIR_CANDIDATES]:
        _, cresidual, ctemplate_residual, cratio, carea, cname, cbest, cmask, cmetrics, cextra = cand
        (
            cgray,
            cmetrics,
            cpost_count,
            cocr_meta,
            cdot_metrics,
            cband_metrics,
            cproduct_metrics,
            calpha_metrics,
            cgate_meta,
        ) = evaluate_cleaned_output(img, cbest, cmask, det, templates, ocr_reader)
        category = candidate_failure_category(cgate_meta)
        trace_id = f"cand_{len(candidate_traces):02d}"
        candidate_traces.append({
            "candidate_id": trace_id,
            "strategy": cname,
            "category": category,
            "rank_score": round(float(cand[0]), 5),
            "residual_score": round(float(cmetrics["residual_score"]), 4),
            "template_residual_score": round(float(cmetrics["template_residual_score"]), 4),
            "post_clean_ocr_score": round(float(cgate_meta.get("post_clean_ocr_score") or 0.0), 4),
            "post_text_components": int(cmetrics.get("post_text_components") or 0),
            "dot_chain_score": round(float(cdot_metrics.get("dot_chain_score") or 0.0), 4),
            "visible_band_score": round(float(cband_metrics.get("visible_band_score") or 0.0), 4),
            "product_blob_score": round(float(cproduct_metrics.get("product_blob_score") or 0.0), 4),
            "alpha_alignment_score": round(float(cextra.get("alpha_alignment_score") or 0.0), 4),
            "alpha_template_residual_after": round(float(calpha_metrics.get("alpha_template_residual_after") or 0.0), 4),
            "reject_reasons": list(cgate_meta.get("reject_reasons") or []),
        })
        if cgate_meta["publish_ok"]:
            selected = cand
            selected_candidate_id = trace_id
            selected_eval = {
                "gray": cgray,
                "metrics": cmetrics,
                "post_count": cpost_count,
                "ocr_meta": cocr_meta,
                "dot_metrics": cdot_metrics,
                "band_metrics": cband_metrics,
                "product_metrics": cproduct_metrics,
                "alpha_metrics": calpha_metrics,
                "gate_meta": cgate_meta,
            }
            break

    _, residual, template_residual, ratio, area, name, best, mask, metrics, selected_extra = selected

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
                selected_extra = {}
                selected_eval = None

    if selected_eval is not None:
        best_gray = selected_eval["gray"]
        metrics = selected_eval["metrics"]
        residual = metrics["residual_score"]
        template_residual = metrics["template_residual_score"]
        post_count = selected_eval["post_count"]
        ocr_meta = selected_eval["ocr_meta"]
        dot_metrics = selected_eval["dot_metrics"]
        band_metrics = selected_eval["band_metrics"]
        product_metrics = selected_eval["product_metrics"]
        alpha_metrics = selected_eval["alpha_metrics"]
        gate_meta = selected_eval["gate_meta"]
    else:
        (
            best_gray,
            metrics,
            post_count,
            ocr_meta,
            dot_metrics,
            band_metrics,
            product_metrics,
            alpha_metrics,
            gate_meta,
        ) = evaluate_cleaned_output(img, best, mask, det, templates, ocr_reader)
    broad_mask = area > 0.020 or (not name.startswith("glyph_") and area > 0.012)
    position_confident = not det.template.startswith("prior:")

    cleanup_attempted = False
    cleanup_strategy = ""
    cleanup_mask_area = 0.0
    cleanup_reasons = set(gate_meta["reject_reasons"])
    residual_cleanup_needed = (
        not gate_meta["publish_ok"]
        and (
            "dot_chain_residual" in cleanup_reasons
            or "residual_visible" in cleanup_reasons
            or "alpha_template_residual" in cleanup_reasons
            or bool(ocr_meta.get("ocr_watermark"))
        )
        and "product_damage" not in cleanup_reasons
        and "visible_rectangular_band" not in cleanup_reasons
    )
    if residual_cleanup_needed:
        cleanup_attempted = True
        cleanup_candidates: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        ring_cleanup = cleanup_residual_components_with_ring_fill(best, dot_metrics["component_mask"], det)
        if ring_cleanup is not None:
            ring_mask = cv2.dilate((dot_metrics["component_mask"] > 0).astype(np.uint8) * 255, np.ones((3, 5), np.uint8), iterations=1)
            ring_area = float(np.count_nonzero(ring_mask)) / max(1, ring_mask.size)
            cleanup_candidates.append(("residual_ring_fill", ring_cleanup, ring_mask, ring_area))
        inpaint_cleanup, inpaint_mask, inpaint_area = cleanup_residual_components_with_inpaint(
            best,
            dot_metrics["component_mask"],
            det,
            risky=risky_product_roi,
        )
        if inpaint_cleanup is not None and inpaint_mask is not None:
            cleanup_candidates.append(("residual_component_inpaint", inpaint_cleanup, inpaint_mask, inpaint_area))

        best_cleanup_score = (
            float(metrics["residual_score"])
            + float(metrics["template_residual_score"]) * 0.35
            + (1.0 if ocr_meta.get("ocr_watermark") else 0.0)
        )
        for cname, cleanup, cleanup_mask, carea in cleanup_candidates:
            combined_cleanup_mask = cv2.bitwise_or(mask, cleanup_mask)
            (
                cgray,
                cleanup_metrics,
                cleanup_post_count,
                cleanup_ocr_meta,
                cleanup_dot_metrics,
                cleanup_band_metrics,
                cleanup_product_metrics,
                cleanup_alpha_metrics,
                cleanup_gate,
            ) = evaluate_cleaned_output(img, cleanup, combined_cleanup_mask, det, templates, ocr_reader)
            cleanup_score = (
                float(cleanup_metrics["residual_score"])
                + float(cleanup_metrics["template_residual_score"]) * 0.35
                + (1.0 if cleanup_ocr_meta.get("ocr_watermark") else 0.0)
                + (0.50 if cleanup_product_metrics.get("product_gate_fail") else 0.0)
                + (0.35 if cleanup_band_metrics.get("band_gate_fail") else 0.0)
            )
            meaningful_improvement = cleanup_score <= best_cleanup_score - 0.08
            if cleanup_gate["publish_ok"] or meaningful_improvement:
                best_cleanup_score = cleanup_score
                best = cleanup
                best_gray = cgray
                metrics = cleanup_metrics
                residual = metrics["residual_score"]
                template_residual = metrics["template_residual_score"]
                post_count = cleanup_post_count
                ocr_meta = cleanup_ocr_meta
                dot_metrics = cleanup_dot_metrics
                band_metrics = cleanup_band_metrics
                product_metrics = cleanup_product_metrics
                alpha_metrics = cleanup_alpha_metrics
                gate_meta = cleanup_gate
                mask = combined_cleanup_mask
                area = float(np.count_nonzero(mask)) / max(1, mask.size)
                cleanup_strategy = cname
                cleanup_mask_area = carea
                name = f"{name}_{cname}"

    status = gate_meta["status"]
    reject_reasons = list(gate_meta["reject_reasons"])
    if ratio < SHARPNESS_MIN_RATIO and residual >= CLEAN_FAIL_RESIDUAL_MIN:
        reject_reasons.append("blurry_and_residual")
    reason = ";".join(dict.fromkeys(reject_reasons))
    gate = "final_publish_gate_pass" if gate_meta["publish_ok"] else "final_publish_gate_reject"

    warnings = []
    if ratio < 0.55:
        warnings.append("low_sharpness_ratio")
    if broad_mask:
        warnings.append("broad_mask")
    if not position_confident:
        warnings.append("prior_detection_source")
    if high_contrast_mask:
        warnings.append("high_contrast_box_mask")
    if risky_product_roi:
        warnings.append(f"risky_roi:{det.roi_class or 'unknown'}")

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
        "roi_class": det.roi_class,
        "product_overlap": round(det.product_overlap, 4),
        "layout_risk": det.layout_risk,
        "metrics_valid": gate_meta["metrics_valid"],
        "residual_pass": gate_meta["residual_pass"],
        "sunsky_check_pass": gate_meta["sunsky_check_pass"],
        "post_clean_ocr_score": round(float(gate_meta["post_clean_ocr_score"]), 4),
        "dot_chain_pass": gate_meta["dot_chain_pass"],
        "band_pass": gate_meta["band_pass"],
        "product_gate_pass": gate_meta["product_gate_pass"],
        "alpha_template_pass": gate_meta.get("alpha_template_pass"),
        "dot_chain_score": round(float(dot_metrics["dot_chain_score"]), 4),
        "dot_component_count": dot_metrics["dot_component_count"],
        "dot_horizontal_span": round(float(dot_metrics["dot_horizontal_span"]), 4),
        "dot_component_area_ratio": round(float(dot_metrics["dot_component_area_ratio"]), 4),
        "visible_band_score": round(float(band_metrics["visible_band_score"]), 4),
        "band_luma_delta": round(float(band_metrics["band_luma_delta"]), 4),
        "band_edge_box": round(float(band_metrics["band_edge_box"]), 4),
        "product_color_delta": round(float(product_metrics["product_color_delta"]), 4),
        "product_edge_retention": round(float(product_metrics["product_edge_retention"]), 4),
        "product_blob_score": round(float(product_metrics["product_blob_score"]), 4),
        "product_changed_area_ratio": round(float(product_metrics["product_changed_area_ratio"]), 4),
        "cleanup_attempted": cleanup_attempted,
        "cleanup_strategy": cleanup_strategy,
        "cleanup_mask_area_pct": round(cleanup_mask_area * 100, 3),
        "candidate_count": len(candidates),
        "best_candidate_id": selected_candidate_id,
        "candidate_gate_trace": candidate_traces[:TOP_K_CANDIDATES],
        "final_blocker_type": final_blocker_type(gate_meta),
        "alpha_engine_used": bool(selected_extra.get("alpha_engine_used")),
        "alpha_asset": str(SUNSKY_ALPHA_PATH.relative_to(PROJECT_ROOT)) if SUNSKY_ALPHA_PATH.exists() else "",
        "alpha_asset_version": str(sunsky_alpha_meta().get("method") or ""),
        "alpha_alignment_score": round(float(selected_extra.get("alpha_alignment_score") or 0.0), 4),
        "alpha_candidate_count": alpha_candidate_count,
        "alpha_best_gain": round(float(selected_extra.get("alpha_best_gain") or 0.0), 4),
        "alpha_best_logo_bgr": selected_extra.get("alpha_best_logo_bgr") or [],
        "alpha_template_residual_before": round(float(alpha_metrics.get("alpha_template_residual_before") or 0.0), 4),
        "alpha_template_residual_after": round(float(alpha_metrics.get("alpha_template_residual_after") or 0.0), 4),
        "alpha_residual_reduction": round(float(alpha_metrics.get("alpha_residual_reduction") or 0.0), 4),
        "thin_residual_inpaint": bool(selected_extra.get("thin_residual_inpaint")),
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
    post_clean_ocr_scores = [
        float(item["post_clean_ocr_score"])
        for item in details
        if isinstance(item.get("post_clean_ocr_score"), (int, float))
    ]
    alpha_alignment_scores = [
        float(item["alpha_alignment_score"])
        for item in details
        if isinstance(item.get("alpha_alignment_score"), (int, float))
    ]
    alpha_before_scores = [
        float(item["alpha_template_residual_before"])
        for item in details
        if isinstance(item.get("alpha_template_residual_before"), (int, float))
    ]
    alpha_after_scores = [
        float(item["alpha_template_residual_after"])
        for item in details
        if isinstance(item.get("alpha_template_residual_after"), (int, float))
    ]
    alpha_reductions = [
        float(item["alpha_residual_reduction"])
        for item in details
        if isinstance(item.get("alpha_residual_reduction"), (int, float))
    ]
    alpha_candidate_counts = [
        int(item["alpha_candidate_count"])
        for item in details
        if isinstance(item.get("alpha_candidate_count"), int)
    ]
    repair_candidate_counts = [
        int(item["candidate_count"])
        for item in details
        if isinstance(item.get("candidate_count"), int)
    ]
    sunsky_check_passes = [
        bool(item["sunsky_check_pass"])
        for item in details
        if "sunsky_check_pass" in item
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
        "post_clean_ocr_score": round(max(post_clean_ocr_scores), 4) if post_clean_ocr_scores else None,
        "sunsky_check_pass": all(sunsky_check_passes) if sunsky_check_passes else None,
        "alpha_engine_used": any(bool(item.get("alpha_engine_used")) for item in details),
        "alpha_asset": str(SUNSKY_ALPHA_PATH.relative_to(PROJECT_ROOT)) if SUNSKY_ALPHA_PATH.exists() else "",
        "alpha_asset_version": str(sunsky_alpha_meta().get("method") or ""),
        "alpha_alignment_score": round(max(alpha_alignment_scores), 4) if alpha_alignment_scores else 0.0,
        "alpha_candidate_count": sum(alpha_candidate_counts) if alpha_candidate_counts else 0,
        "alpha_best_gain": next((item.get("alpha_best_gain") for item in details if item.get("alpha_best_gain")), 0.0),
        "alpha_best_logo_bgr": next((item.get("alpha_best_logo_bgr") for item in details if item.get("alpha_best_logo_bgr")), []),
        "alpha_template_residual_before": round(max(alpha_before_scores), 4) if alpha_before_scores else 0.0,
        "alpha_template_residual_after": round(max(alpha_after_scores), 4) if alpha_after_scores else 0.0,
        "alpha_residual_reduction": round(max(alpha_reductions), 4) if alpha_reductions else 0.0,
        "thin_residual_inpaint": any(bool(item.get("thin_residual_inpaint")) for item in details),
        "candidate_count": sum(repair_candidate_counts) if repair_candidate_counts else 0,
        "best_candidate_id": next((str(item.get("best_candidate_id")) for item in details if item.get("best_candidate_id")), ""),
        "final_blocker_type": next((str(item.get("final_blocker_type")) for item in details if item.get("final_blocker_type") != "none"), "none"),
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
        if row.get("alpha_alignment_score"):
            meta_parts.append(f"alpha_align {row.get('alpha_alignment_score')}")
        if row.get("alpha_template_residual_before") is not None and row.get("alpha_template_residual_after") is not None:
            meta_parts.append(
                f"alpha {row.get('alpha_template_residual_before')}->{row.get('alpha_template_residual_after')}"
            )
        if row.get("post_clean_ocr_score") is not None:
            meta_parts.append(f"postOCR {row.get('post_clean_ocr_score')}")
        if row.get("post_text_components") is not None:
            meta_parts.append(f"components {row.get('post_text_components')}")
        if row.get("presence_reason"):
            meta_parts.append(str(row.get("presence_reason")))
        if row.get("detection", {}).get("roi_class"):
            meta_parts.append(f"roi {row['detection']['roi_class']}")
        if row.get("reason"):
            meta_parts.append(str(row.get("reason")))
        meta = html.escape(" | ".join(part for part in meta_parts if part))
        result_label = "Cleaned" if row.get("status") == "cleaned" else "Attempt (failed QA)"
        body.append(f"""
<section class="card">
  <div class="head"><h2>#{idx} {fname}</h2><div class="meta">{meta}</div></div>
  <div class="grid">
    <div class="pane"><img src="{original}" alt="Original"><div class="label">Original</div></div>
    <div class="pane"><img src="{mask}" alt="Mask overlay"><div class="label">Mask overlay</div></div>
    <div class="pane"><img src="{cleaned}" alt="Result"><div class="label">{html.escape(result_label)}</div></div>
    <div class="pane"><img src="{diff}" alt="Difference"><div class="label">Diff x3</div></div>
  </div>
</section>""")
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Watermark Pilot Review</title><style>{css}</style></head>
<body><div class="top"><h1>Watermark pilot review</h1><p>Source images were not modified. Output: {html.escape(str(out_dir))}</p></div>
<main class="wrap">{''.join(body)}</main></body></html>"""
    (out_dir / "review.html").write_text(doc)


def _load_pdf_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _pdf_safe_image(path: Path, max_size: tuple[int, int]):
    from PIL import Image, ImageDraw

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", max_size, "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, max_size[0] - 1, max_size[1] - 1], outline="#cbd5e1")
        draw.text((18, 18), "missing image", fill="#64748b", font=_load_pdf_font(18))
        return img
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def make_compare_pdf(out_dir: Path, rows: list[dict]) -> Path:
    """Create a compact visual review PDF from the same assets as review.html."""
    from PIL import Image, ImageDraw

    pdf_path = out_dir / "compare.pdf"
    title_font = _load_pdf_font(26, bold=True)
    meta_font = _load_pdf_font(18)
    label_font = _load_pdf_font(20, bold=True)
    small_font = _load_pdf_font(15)
    page_size = (1680, 1080)
    margin = 36
    gap = 18
    header_h = 100
    label_h = 34
    col_w = (page_size[0] - margin * 2 - gap * 3) // 4
    img_h = page_size[1] - margin * 2 - header_h - label_h
    panes = [
        ("Original", "review_original"),
        ("Mask overlay", "review_mask"),
        ("__RESULT__", "review_cleaned"),
        ("Diff x3", "review_diff"),
    ]
    pages = []
    pdf_rows = rows or [{"file": "no rows", "status": "empty"}]
    for idx, row in enumerate(pdf_rows, 1):
        page = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(page)
        filename = str(row.get("file", ""))
        status = str(row.get("status", ""))
        strategy = str(row.get("strategy") or row.get("reason") or "")
        metrics = []
        for key, label in [
            ("mask_area_pct", "mask"),
            ("residual_score", "visible"),
            ("template_residual_score", "template"),
            ("post_clean_ocr_score", "postOCR"),
            ("post_text_components", "components"),
            ("presence_score", "presence"),
            ("alpha_alignment_score", "alpha_align"),
            ("alpha_template_residual_after", "alpha_after"),
        ]:
            if key in row:
                metrics.append(f"{label} {row.get(key)}")
        if row.get("alpha_template_residual_before") is not None and row.get("alpha_template_residual_after") is not None:
            metrics.append(f"alpha {row.get('alpha_template_residual_before')}->{row.get('alpha_template_residual_after')}")
        if row.get("detection", {}).get("roi_class"):
            metrics.append(f"roi {row['detection']['roi_class']}")
        if row.get("reason"):
            metrics.append(f"reason {row.get('reason')}")
        draw.text((margin, 24), f"#{idx} {filename}", fill="#111827", font=title_font)
        draw.text(
            (margin, 60),
            " | ".join(part for part in [status, strategy, *metrics] if part),
            fill="#475569",
            font=meta_font,
        )

        for col, (label, rel_key) in enumerate(panes):
            pane_label = label
            if label == "__RESULT__":
                pane_label = "Cleaned" if row.get("status") == "cleaned" else "Attempt (failed QA)"
            x = margin + col * (col_w + gap)
            y = margin + header_h
            rel_path = row.get(rel_key)
            img_path = out_dir / rel_path if rel_path else Path()
            img = _pdf_safe_image(img_path, (col_w, img_h))
            box = [x, y, x + col_w, y + img_h]
            draw.rectangle(box, outline="#d8dee8", width=2)
            px = x + (col_w - img.width) // 2
            py = y + (img_h - img.height) // 2
            page.paste(img, (px, py))
            draw.rectangle([x, y + img_h, x + col_w, y + img_h + label_h], fill="#f8fafc", outline="#d8dee8")
            draw.text((x + 12, y + img_h + 7), pane_label, fill="#334155", font=label_font)
        draw.text(
            (margin, page_size[1] - 24),
            f"Clearmark visual review PDF: {out_dir}",
            fill="#64748b",
            font=small_font,
        )
        pages.append(page)

    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:], resolution=120.0, quality=90)
    return pdf_path


def send_telegram_document(
    document_path: Path,
    caption: str,
    token_env: str,
    chat_id_env: str,
    chat_id_arg: str | None = None,
) -> dict:
    token = os.environ.get(token_env)
    chat_id = chat_id_arg or os.environ.get(chat_id_env)
    if not token or not chat_id:
        return {
            "sent": False,
            "reason": "missing_telegram_env",
            "token_env": token_env,
            "chat_id_env": chat_id_env,
        }

    import urllib.error
    import urllib.request

    boundary = f"----clearmark-{uuid.uuid4().hex}"
    body = bytearray()

    def add_field(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    def add_file(name: str, path: Path, content_type: str) -> None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    add_field("chat_id", chat_id)
    add_field("caption", caption[:1024])
    add_file("document", document_path, "application/pdf")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendDocument",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        return {"sent": False, "reason": f"http_{exc.code}", "response": payload[:600]}
    except Exception as exc:
        return {"sent": False, "reason": type(exc).__name__}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"sent": False, "reason": "invalid_json", "response": payload[:600]}
    if not parsed.get("ok"):
        return {"sent": False, "reason": "telegram_rejected", "response": parsed}
    result = parsed.get("result", {})
    return {"sent": True, "message_id": result.get("message_id"), "chat_id": chat_id}


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
    require_presence_confirmed: bool = False,
) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"file": path.name, "status": "needs_manual", "reason": "read_failed"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    layout = image_layout_features(gray, img)
    detections = detect_watermark(gray, templates, preset, img=img, ocr_reader=ocr_reader, layout=layout)
    if not detections:
        entry = {"file": path.name, "status": "no_watermark", "layout": layout}
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
    presence = confirm_watermark_presence(img, gray, detections, ocr_reader, layout=layout)
    if require_presence_confirmed and not presence.get("presence_confirmed"):
        entry = {"file": path.name, "status": "no_watermark", **presence, "layout": layout}
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
        **presence,
        "layout": layout,
        "detection": detections[0].to_json(),
        "detections": [det.to_json() for det in detections],
    }
    if mask is not None:
        mask_path = out_dir / "masks" / path.name
        copy_or_write_image(mask_path, mask)
        entry["mask"] = str(mask_path)
    if cleaned is not None and meta.get("status") == "cleaned":
        clean_path = out_dir / "cleaned" / path.name
        copy_or_write_image(clean_path, cleaned)
        entry["cleaned"] = str(clean_path)
    elif cleaned is not None:
        attempt_path = out_dir / "attempts" / path.name
        copy_or_write_image(attempt_path, cleaned)
        entry["attempt"] = str(attempt_path)
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
        if cleaned is not None and meta.get("status") == "cleaned":
            entry["review_cleaned"] = f"cleaned/{path.name}"
        elif cleaned is not None:
            entry["review_cleaned"] = f"attempts/{path.name}"
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
    global ENABLE_LAMA_ESCALATION
    require_rights(args)
    ENABLE_LAMA_ESCALATION = bool(args.lama)
    assets = args.assets.expanduser().resolve()
    out_dir = prepare_out_dir(args.out, "pilot")
    inventory, summary = build_inventory(assets, out_dir, args.phash_threshold)
    scan_limit = args.max_scan or (args.max_total * 8 if args.watermarked_only else args.max_total)
    sample = choose_pilot_files(inventory, scan_limit, args.seed)
    templates = load_templates()
    ocr_reader = load_ocr_reader(args.ocr)
    rows = []
    skipped_no_watermark = 0
    started = time.time()
    for idx, fname in enumerate(sample, 1):
        row = process_file(
            assets / fname,
            templates,
            args.preset,
            out_dir,
            review=True,
            ocr_reader=ocr_reader,
            require_presence_confirmed=args.watermarked_only,
        )
        if args.watermarked_only and row["status"] == "no_watermark":
            skipped_no_watermark += 1
            presence_reason = row.get("presence_reason", "no_detection")
            print(f"  pilot scan {idx}/{len(sample)} selected {len(rows)}/{args.max_total} skip {presence_reason} {fname}")
            continue
        rows.append(row)
        print(f"  pilot {idx}/{len(sample)} selected {len(rows)}/{args.max_total} {row['status']} {fname}")
        if args.watermarked_only and len(rows) >= args.max_total:
            break
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
        "watermarked_only": bool(args.watermarked_only),
        "scan_limit": scan_limit,
        "selected": len(rows),
        "skipped_no_watermark_during_selection": skipped_no_watermark,
        "seconds": round(time.time() - started, 2),
        "inventory": summary["counts"],
        "counts": counts,
    }
    pdf_path = None
    if args.pdf or args.telegram:
        pdf_path = make_compare_pdf(out_dir, rows)
        final["compare_pdf"] = str(pdf_path)
    if args.telegram:
        if pdf_path is None:
            pdf_path = make_compare_pdf(out_dir, rows)
        caption = f"Clearmark Sunsky review: {len(rows)} files | {counts}"
        final["telegram"] = send_telegram_document(
            pdf_path,
            caption,
            args.telegram_token_env,
            args.telegram_chat_id_env,
            args.telegram_chat_id,
        )
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2))
    print(f"Pilot complete: {out_dir}")
    print(f"Review HTML: {out_dir / 'review.html'}")
    if pdf_path is not None:
        print(f"Compare PDF: {pdf_path}")
    if args.telegram:
        print(f"Telegram: {json.dumps(final['telegram'], indent=2)}")
    print(json.dumps(counts, indent=2))


def cmd_process(args: argparse.Namespace) -> None:
    global ENABLE_LAMA_ESCALATION
    require_rights(args)
    ENABLE_LAMA_ESCALATION = bool(args.lama)
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
        return process_file(
            assets / row["file"],
            templates,
            args.preset,
            out_dir,
            review=False,
            ocr_reader=ocr_reader,
            require_presence_confirmed=True,
        )

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
    p.add_argument("--max-scan", type=int, help="Maximum candidate files to inspect when --watermarked-only is enabled.")
    p.add_argument("--seed", type=int, default=1779606245)
    p.add_argument("--watermarked-only", action="store_true", help="Keep scanning until the pilot has --max-total detected watermark cases.")
    p.add_argument("--pdf", action="store_true", help="Write compare.pdf beside review.html.")
    p.add_argument("--telegram", action="store_true", help="Send compare.pdf to Telegram after the pilot finishes.")
    p.add_argument("--telegram-chat-id")
    p.add_argument("--telegram-token-env", default="TELEGRAM_BOT_TOKEN")
    p.add_argument("--telegram-chat-id-env", default="TELEGRAM_CHAT_ID")
    p.add_argument("--lama", action=argparse.BooleanOptionalAction, default=True, help="Allow optional LaMa/IOPaint escalation for high-residual cases. Pass --no-lama for faster review sampling.")
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
    p.add_argument("--lama", action=argparse.BooleanOptionalAction, default=True, help="Allow optional LaMa/IOPaint escalation for high-residual cases.")
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
