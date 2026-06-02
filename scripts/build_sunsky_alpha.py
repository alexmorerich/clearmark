#!/usr/bin/env python3
"""Build or calibrate the Sunsky alpha asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sunsky_alpha_engine import (
    LOGO_BGR_CANDIDATES,
    apply_reverse_alpha,
    imread_unicode,
    imwrite_unicode,
    score_alpha_residual,
    solve_alpha_map_from_background,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "templates" / "watermark-template.png"
DEFAULT_ALPHA = PROJECT_ROOT / "templates" / "sunsky-alpha.png"
DEFAULT_META = PROJECT_ROOT / "templates" / "sunsky-alpha-meta.json"


def build_bootstrap_alpha(template_path: Path) -> tuple[np.ndarray, dict]:
    template = imread_unicode(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None or template.size == 0:
        raise SystemExit(f"Unable to read template: {template_path}")

    inv = 255 - template
    ys, xs = np.where(inv > 6)
    if not len(xs) or not len(ys):
        raise SystemExit(f"Template has no usable glyph ink: {template_path}")
    x1 = max(0, int(xs.min()) - 2)
    y1 = max(0, int(ys.min()) - 2)
    x2 = min(template.shape[1], int(xs.max()) + 3)
    y2 = min(template.shape[0], int(ys.max()) + 3)
    cropped = inv[y1:y2, x1:x2].astype(np.float32)
    cropped /= max(float(cropped.max()), 1.0)

    # Preserve anti-aliased halos from the canonical image, but keep empty
    # background pixels at zero so the engine never edits a full rectangle.
    cropped[cropped < 0.012] = 0.0
    alpha = np.uint8(np.clip(cropped * 255.0, 0, 255))

    meta = {
        "mark": "sunsky-online.com",
        "method": "bootstrap_from_canonical_template",
        "logo_bgr": [180, 180, 180],
        "alpha_floor": 0.02,
        "samples_used": 0,
        "source_template": str(template_path),
        "crop_box": [x1, y1, x2 - x1, y2 - y1],
        "created_by": "scripts/build_sunsky_alpha.py",
        "calibration_note": (
            "This bootstrap alpha is derived from templates/watermark-template.png. "
            "Future runs can replace it with averaged alpha solves from high-confidence "
            "OCR-localized samples without changing the engine API."
        ),
    }
    return alpha, meta


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return image[:, :, :3].copy()
    return image.copy()


def build_calibrated_alpha(template_path: Path, samples_dir: Path) -> tuple[np.ndarray, dict]:
    bootstrap_u8, bootstrap_meta = build_bootstrap_alpha(template_path)
    bootstrap = bootstrap_u8.astype(np.float32) / 255.0
    logo_candidates = [(180.0, 180.0, 180.0), *LOGO_BGR_CANDIDATES]
    calibrated_maps: list[np.ndarray] = []
    sample_paths = sorted(
        path for path in samples_dir.glob("*.png")
        if not path.name.startswith("sunsky-alpha") and path.name != template_path.name
    )

    for sample_path in sample_paths:
        sample = imread_unicode(sample_path, cv2.IMREAD_UNCHANGED)
        if sample is None or sample.size == 0:
            continue
        bgr = _as_bgr(sample)
        support = cv2.resize(bootstrap, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
        mark_box = (0, 0, bgr.shape[1], bgr.shape[0])
        best_alpha: np.ndarray | None = None
        best_score = float("inf")
        for logo in logo_candidates:
            solved = solve_alpha_map_from_background(bgr, support, logo)
            if int(np.count_nonzero(solved > 0.015)) < 6:
                continue
            restored = apply_reverse_alpha(bgr, solved, logo, 1.0)
            score = score_alpha_residual(restored, mark_box, support)
            if score < best_score:
                best_score = score
                best_alpha = solved
        if best_alpha is None:
            continue
        calibrated = cv2.resize(best_alpha, (bootstrap.shape[1], bootstrap.shape[0]), interpolation=cv2.INTER_AREA)
        calibrated_maps.append(np.clip(calibrated, 0.0, 1.0))

    if not calibrated_maps:
        return bootstrap_u8, bootstrap_meta

    stacked = np.stack(calibrated_maps, axis=0)
    alpha = np.median(stacked, axis=0)
    alpha = np.maximum(alpha, bootstrap * 0.12)
    alpha[bootstrap <= 0.0] = 0.0
    alpha[alpha < 0.012] = 0.0
    alpha_u8 = np.uint8(np.clip(alpha * 255.0, 0, 255))
    meta = {
        "mark": "sunsky-online.com",
        "method": "sample_crop_background_alpha_solve",
        "logo_bgr": [180, 180, 180],
        "alpha_floor": 0.02,
        "samples_used": len(calibrated_maps),
        "source_template": str(template_path),
        "samples_dir": str(samples_dir),
        "sample_files": [path.name for path in sample_paths],
        "created_by": "scripts/build_sunsky_alpha.py",
        "calibration_note": (
            "Alpha was solved from sample watermark crops using the same local-background "
            "solver used by SunskyAlphaEngine. The canonical template remains the support mask."
        ),
    }
    return alpha_u8, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Sunsky alpha asset")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--samples", type=Path, help="Directory of high-confidence Sunsky crop PNG samples.")
    parser.add_argument("--alpha-out", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    args = parser.parse_args(argv)

    if args.samples:
        alpha, meta = build_calibrated_alpha(args.template, args.samples)
    else:
        alpha, meta = build_bootstrap_alpha(args.template)
    if not imwrite_unicode(args.alpha_out, alpha):
        raise SystemExit(f"Unable to write alpha asset: {args.alpha_out}")
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.alpha_out}")
    print(f"Wrote {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
