#!/usr/bin/env python3
"""Deterministic reverse-alpha repair engine for the Sunsky text watermark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ALPHA_GAINS = [0.75, 0.90, 1.00, 1.10, 1.25]
LOGO_BGR_CANDIDATES = [
    (150.0, 150.0, 150.0),
    (170.0, 170.0, 170.0),
    (190.0, 190.0, 190.0),
    (210.0, 210.0, 210.0),
    (235.0, 235.0, 235.0),
]
DENOMINATOR_MIN = 0.25
ALPHA_FLOOR = 0.015
ALPHA_MAX = 0.92
FINAL_ALPHA_TEMPLATE_MAX = 0.08
MIN_ALPHA_RESIDUAL_REDUCTION = 0.55


@dataclass(frozen=True)
class SunskyAlphaCandidate:
    name: str
    image: np.ndarray
    alpha_map: np.ndarray
    glyph_bbox: tuple[int, int, int, int]
    alignment_score: float
    alpha_gain: float
    logo_bgr: tuple[float, float, float]
    residual_probe_score: float
    residual_mask_area: int


@dataclass(frozen=True)
class SunskyAlphaDetection:
    detected: bool
    confidence: float
    region: tuple[int, int, int, int]
    scale: float
    offset_xy: tuple[int, int]
    coverage: float


def imread_unicode(path: Path | str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path | str, image: np.ndarray, params: list[int] | None = None) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image, params or [])
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def _load_alpha(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    alpha = img.astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def _template_alpha_from_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    inv = 255 - img
    ys, xs = np.where(inv > 8)
    if not len(xs) or not len(ys):
        return None
    x1 = max(0, int(xs.min()) - 1)
    y1 = max(0, int(ys.min()) - 1)
    x2 = min(img.shape[1], int(xs.max()) + 2)
    y2 = min(img.shape[0], int(ys.max()) + 2)
    alpha = inv[y1:y2, x1:x2].astype(np.float32)
    alpha /= max(float(alpha.max()), 1.0)
    alpha[alpha < 0.01] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def _clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    w = max(1, min(int(w), img_w))
    h = max(1, min(int(h), img_h))
    x = max(0, min(int(x), img_w - w))
    y = max(0, min(int(y), img_h - h))
    return x, y, w, h


def _expanded_box(
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _place_alpha(shape: tuple[int, int], alpha: np.ndarray, x: int, y: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    img_h, img_w = shape
    h, w = alpha.shape[:2]
    x, y, w, h = _clip_box(x, y, w, h, img_w, img_h)
    placed = np.zeros(shape, dtype=np.float32)
    placed[y:y + h, x:x + w] = alpha[:h, :w]
    return placed, (x, y, w, h)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def extract_polarity_aware_sunsky_mask(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        gray = image_bgr
        sat = np.zeros_like(gray)
    elif image_bgr.shape[2] == 4:
        bgr = image_bgr[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

    blur_ksize = max(9, min(41, (max(5, min(gray.shape[:2]) // 2) // 2) * 2 + 1))
    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    luma = gray.astype(np.float32)
    bg = background.astype(np.float32)
    bright_on_dark = np.maximum(luma - bg, 0.0)
    dark_on_light = np.maximum(bg - luma, 0.0)
    delta = np.maximum(bright_on_dark, dark_on_light)

    low_sat = sat <= max(72, int(np.percentile(sat, 78)))
    if float(np.percentile(delta, 97)) < 1.5:
        boosted = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        boosted = np.clip(delta * 8.0, 0, 255).astype(np.uint8)
    threshold = max(6.0, float(np.percentile(boosted, 82)))
    mask = ((boosted >= threshold) & low_sat).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8), iterations=1)
    return mask


def apply_reverse_alpha(
    image_bgr: np.ndarray,
    full_alpha: np.ndarray,
    logo_bgr: tuple[float, float, float],
    alpha_gain: float,
) -> np.ndarray:
    if image_bgr.ndim == 2:
        src = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:
        src = image_bgr[:, :, :3].copy()
    else:
        src = image_bgr.copy()

    a = np.clip(full_alpha.astype(np.float32) * float(alpha_gain), 0.0, ALPHA_MAX)
    mask = a >= ALPHA_FLOOR
    a3 = a[:, :, None]
    logo = np.array(logo_bgr, dtype=np.float32).reshape(1, 1, 3)
    restored = (src.astype(np.float32) - a3 * logo) / np.clip(1.0 - a3, DENOMINATOR_MIN, 1.0)
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    out = src.copy()
    out[mask] = restored[mask]
    return out


def score_alpha_residual(
    image_bgr: np.ndarray,
    mark_box: tuple[int, int, int, int],
    alpha_template: np.ndarray,
) -> float:
    if alpha_template is None or alpha_template.size == 0:
        return 0.0
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    x, y, w, h = _clip_box(*mark_box, img_w, img_h)
    pad_x = max(8, int(round(w * 0.16)))
    pad_y = max(6, int(round(h * 0.60)))
    sx, sy, sw, sh = _expanded_box((x, y, w, h), img_w, img_h, pad_x, pad_y)
    search = gray[sy:sy + sh, sx:sx + sw]
    if search.shape[0] < 8 or search.shape[1] < 24:
        return 0.0
    evidence = extract_polarity_aware_sunsky_mask(cv2.cvtColor(search, cv2.COLOR_GRAY2BGR))
    best = 0.0
    for scale in np.linspace(0.86, 1.16, 13):
        tw = max(8, int(round(w * scale)))
        th = max(4, int(round(tw * alpha_template.shape[0] / max(alpha_template.shape[1], 1))))
        if th > h * 1.35:
            th = max(4, int(round(h * 1.05 * scale)))
            tw = max(8, int(round(th * alpha_template.shape[1] / max(alpha_template.shape[0], 1))))
        if tw >= search.shape[1] or th >= search.shape[0] or tw < 16 or th < 4:
            continue
        sil = cv2.resize(alpha_template, (tw, th), interpolation=cv2.INTER_AREA)
        sil = (sil > 0.08).astype(np.uint8) * 255
        if float(np.std(sil)) < 1e-3:
            continue
        result = cv2.matchTemplate(evidence, sil, cv2.TM_CCOEFF_NORMED)
        if result.size:
            best = max(best, float(np.max(result)))
    return float(max(0.0, min(1.0, best)))


class SunskyAlphaEngine:
    def __init__(
        self,
        alpha_path: Path,
        template_path: Path,
        *,
        logo_bgr: tuple[float, float, float] = (180.0, 180.0, 180.0),
        min_alignment_score: float = 0.35,
    ) -> None:
        self.alpha_path = Path(alpha_path)
        self.template_path = Path(template_path)
        self.logo_bgr = logo_bgr
        self.min_alignment_score = float(min_alignment_score)
        self.alpha = _load_alpha(self.alpha_path)
        self.template_alpha = _template_alpha_from_image(self.template_path)
        if self.alpha is None:
            self.alpha = self.template_alpha

    def alpha_available(self) -> bool:
        return self.alpha is not None and self.alpha.size > 0

    def _scaled_alpha_for_mark_box(
        self,
        mark_box: tuple[int, int, int, int],
        scale: float,
    ) -> np.ndarray | None:
        if not self.alpha_available():
            return None
        assert self.alpha is not None
        _, _, w, h = mark_box
        target_w = max(16, int(round(w * float(scale))))
        target_h = max(4, int(round(target_w * self.alpha.shape[0] / max(self.alpha.shape[1], 1))))
        if target_h > h * 1.35:
            target_h = max(4, int(round(h * 1.04 * float(scale))))
            target_w = max(16, int(round(target_h * self.alpha.shape[1] / max(self.alpha.shape[0], 1))))
        if target_w < 16 or target_h < 4:
            return None
        return cv2.resize(self.alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def align_alpha_to_mark_box(
        self,
        image_bgr: np.ndarray,
        mark_box: tuple[int, int, int, int],
        *,
        roi_class: str,
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
        if not self.alpha_available() or image_bgr.size == 0:
            return []
        if image_bgr.ndim == 2:
            img_h, img_w = image_bgr.shape[:2]
            bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        else:
            img_h, img_w = image_bgr.shape[:2]
            bgr = image_bgr[:, :, :3]
        x, y, w, h = _clip_box(*mark_box, img_w, img_h)
        pad_x = max(12, int(round(w * 0.22)))
        pad_y = max(8, int(round(h * 0.95)))
        sx, sy, sw, sh = _expanded_box((x, y, w, h), img_w, img_h, pad_x, pad_y)
        search = bgr[sy:sy + sh, sx:sx + sw]
        glyph_mask = extract_polarity_aware_sunsky_mask(search)

        results: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []
        for scale in np.linspace(0.80, 1.25, 27):
            alpha_scaled = self._scaled_alpha_for_mark_box((x, y, w, h), float(scale))
            if alpha_scaled is None:
                continue
            ah, aw = alpha_scaled.shape[:2]
            if aw >= sw or ah >= sh:
                continue
            sil = (alpha_scaled > 0.12).astype(np.uint8) * 255
            if float(np.std(sil)) < 1e-3:
                continue
            score_map = cv2.matchTemplate(glyph_mask, sil, cv2.TM_CCOEFF_NORMED)
            if score_map.size:
                _, score, _, loc = cv2.minMaxLoc(score_map)
                px = sx + int(loc[0])
                py = sy + int(loc[1])
                if score >= self.min_alignment_score:
                    placed, bbox = _place_alpha((img_h, img_w), alpha_scaled, px, py)
                    results.append((placed, bbox, float(score)))

            # OCR boxes are often correct but slightly loose. Add a small
            # deterministic placement grid around the box center so exact
            # reverse-alpha is tried even when local contrast pulls NCC to a
            # nearby halo edge.
            base_x = x + (w - aw) / 2.0
            base_y = y + (h - ah) / 2.0
            for dx_frac in (-0.055, 0.0, 0.055):
                for dy_frac in (-0.32, 0.0, 0.32):
                    px = int(round(base_x + aw * dx_frac))
                    py = int(round(base_y + ah * dy_frac))
                    placed, bbox = _place_alpha((img_h, img_w), alpha_scaled, px, py)
                    fixed_score = score_alpha_residual(bgr, bbox, alpha_scaled)
                    if fixed_score >= self.min_alignment_score * 0.72:
                        results.append((placed, bbox, float(max(fixed_score, self.min_alignment_score * 0.72))))

        results.sort(key=lambda item: item[2], reverse=True)
        unique: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []
        for item in results:
            if all(_iou(item[1], old[1]) < 0.90 for old in unique):
                unique.append(item)
            if len(unique) >= 8:
                break
        return unique

    def reverse_alpha_candidates(
        self,
        image_bgr: np.ndarray,
        mark_box: tuple[int, int, int, int],
        *,
        roi_class: str,
    ) -> list[SunskyAlphaCandidate]:
        if not self.alpha_available() or image_bgr.size == 0:
            return []
        if image_bgr.ndim == 2:
            source = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            source = image_bgr[:, :, :3].copy()
        else:
            source = image_bgr.copy()

        placements = self.align_alpha_to_mark_box(source, mark_box, roi_class=roi_class)
        candidates: list[SunskyAlphaCandidate] = []
        risky = roi_class in {"dark_product_surface", "thin_flex_cable", "complex_product_detail", "text_or_label_area"}
        edge_kernel = np.ones((3, 3), np.uint8) if risky else np.ones((3, 5), np.uint8)
        inpaint_radius = 1 if risky else 2
        template = self.alpha if self.alpha is not None else np.ones((4, 16), dtype=np.float32)
        logo_candidates = []
        for logo in [self.logo_bgr, *LOGO_BGR_CANDIDATES]:
            if logo not in logo_candidates:
                logo_candidates.append(logo)

        for place_idx, (full_alpha, bbox, align_score) in enumerate(placements):
            alpha_area = int(np.count_nonzero(full_alpha >= ALPHA_FLOOR))
            if alpha_area <= 0:
                continue
            for gain in ALPHA_GAINS:
                for logo in logo_candidates:
                    restored = apply_reverse_alpha(source, full_alpha, logo, gain)
                    residual = score_alpha_residual(restored, mark_box, template)
                    candidates.append(SunskyAlphaCandidate(
                        name="sunsky_reverse_alpha_aligned",
                        image=restored,
                        alpha_map=full_alpha,
                        glyph_bbox=bbox,
                        alignment_score=float(align_score),
                        alpha_gain=float(gain),
                        logo_bgr=logo,
                        residual_probe_score=float(residual),
                        residual_mask_area=alpha_area,
                    ))

                    edge_mask = cv2.dilate((full_alpha > 0.04).astype(np.uint8) * 255, edge_kernel, iterations=1)
                    thin = cv2.inpaint(restored, edge_mask, inpaint_radius, cv2.INPAINT_NS)
                    thin_residual = score_alpha_residual(thin, mark_box, template)
                    candidates.append(SunskyAlphaCandidate(
                        name="sunsky_reverse_alpha_aligned_thin_ns",
                        image=thin,
                        alpha_map=np.maximum(full_alpha, edge_mask.astype(np.float32) / 255.0 * 0.04),
                        glyph_bbox=bbox,
                        alignment_score=float(align_score),
                        alpha_gain=float(gain),
                        logo_bgr=logo,
                        residual_probe_score=float(thin_residual),
                        residual_mask_area=int(np.count_nonzero(edge_mask)),
                    ))

        candidates.sort(key=lambda cand: (
            cand.residual_probe_score - cand.alignment_score * 0.22,
            cand.residual_probe_score,
            cand.residual_mask_area,
        ))
        return candidates[:40]

    def remove_best(
        self,
        image_bgr: np.ndarray,
        mark_box: tuple[int, int, int, int],
        *,
        roi_class: str,
    ) -> SunskyAlphaCandidate | None:
        candidates = self.reverse_alpha_candidates(image_bgr, mark_box, roi_class=roi_class)
        if not candidates:
            return None
        return candidates[0]
