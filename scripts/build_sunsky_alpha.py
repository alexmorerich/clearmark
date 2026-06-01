#!/usr/bin/env python3
"""Build the bootstrap Sunsky alpha asset from the canonical text template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sunsky_alpha_engine import imread_unicode, imwrite_unicode


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Sunsky alpha asset")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--alpha-out", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    args = parser.parse_args(argv)

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

