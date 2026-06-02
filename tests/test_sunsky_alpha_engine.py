from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sunsky_alpha_engine import (  # noqa: E402
    ALPHA_FLOOR,
    SunskyAlphaEngine,
    apply_reverse_alpha,
    fill_alpha_support_with_background,
    imread_unicode,
    imwrite_unicode,
    score_alpha_residual,
    solve_alpha_map_from_background,
)
from build_sunsky_alpha import build_calibrated_alpha  # noqa: E402


def engine() -> SunskyAlphaEngine:
    return SunskyAlphaEngine(
        ROOT / "templates" / "sunsky-alpha.png",
        ROOT / "templates" / "watermark-template.png",
    )


def place_alpha(alpha: np.ndarray, shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    scaled = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    full = np.zeros(shape, dtype=np.float32)
    full[y:y + h, x:x + w] = scaled
    return full


def overlay(clean: np.ndarray, alpha: np.ndarray, logo=(180.0, 180.0, 180.0)) -> np.ndarray:
    return np.uint8(np.clip(alpha[:, :, None] * np.array(logo, np.float32) + (1.0 - alpha[:, :, None]) * clean, 0, 255))


def changed_outside_alpha(candidate: np.ndarray, clean: np.ndarray, alpha: np.ndarray) -> int:
    diff = cv2.absdiff(candidate, clean)
    return int(np.count_nonzero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)[alpha < ALPHA_FLOOR]))


def test_synthetic_alpha_recovery_on_common_backgrounds() -> None:
    eng = engine()
    assert eng.alpha_available()
    backgrounds = [
        np.full((180, 260, 3), 248, np.uint8),
        np.dstack([
            np.tile(np.linspace(210, 238, 260, dtype=np.uint8), (180, 1)),
            np.tile(np.linspace(210, 238, 260, dtype=np.uint8), (180, 1)),
            np.tile(np.linspace(210, 238, 260, dtype=np.uint8), (180, 1)),
        ]),
        np.full((180, 260, 3), (70, 70, 210), np.uint8),
        np.full((180, 260, 3), 58, np.uint8),
    ]
    mark_box = (50, 70, 160, 40)
    alpha_box = (52, 80, 156, 21)
    for clean in backgrounds:
        alpha = place_alpha(eng.alpha, clean.shape[:2], alpha_box)
        watermarked = overlay(clean, alpha)
        before = score_alpha_residual(watermarked, mark_box, eng.alpha)
        candidate = eng.remove_best(watermarked, mark_box, roi_class="near_white")
        assert candidate is not None
        after = score_alpha_residual(candidate.image, mark_box, eng.alpha)
        assert after < before * 0.75
        assert changed_outside_alpha(candidate.image, watermarked, candidate.alpha_map) <= 10


def test_alignment_jitter_reduces_residual() -> None:
    eng = engine()
    clean = np.full((190, 280, 3), 242, np.uint8)
    for dx, dy, scale in [(-8, -5, 0.85), (0, 0, 1.0), (8, 5, 1.20)]:
        w = int(round(156 * scale))
        h = int(round(21 * scale))
        alpha_box = (52 + dx, 80 + dy, w, h)
        mark_box = (50, 70, 170, 45)
        alpha = place_alpha(eng.alpha, clean.shape[:2], alpha_box)
        watermarked = overlay(clean, alpha)
        before = score_alpha_residual(watermarked, mark_box, eng.alpha)
        candidate = eng.remove_best(watermarked, mark_box, roi_class="near_white")
        assert candidate is not None
        after = score_alpha_residual(candidate.image, mark_box, eng.alpha)
        assert after < before * 0.85


def test_per_image_alpha_solve_clears_mismatched_overlay() -> None:
    eng = engine()
    clean = np.full((180, 280, 3), 236, np.uint8)
    mark_box = (48, 68, 178, 44)
    support = place_alpha(eng.alpha, clean.shape[:2], (52, 80, 156, 21))
    true_alpha = np.clip(support * 0.42 + (support > ALPHA_FLOOR).astype(np.float32) * 0.025, 0.0, 0.34)
    watermarked = overlay(clean, true_alpha, logo=(205.0, 205.0, 205.0))

    before = score_alpha_residual(watermarked, mark_box, eng.alpha)
    solved = solve_alpha_map_from_background(watermarked, support, (205.0, 205.0, 205.0))
    restored = apply_reverse_alpha(watermarked, solved, (205.0, 205.0, 205.0), 1.0)
    filled = fill_alpha_support_with_background(watermarked, np.maximum(solved, support))
    assert filled is not None
    after = score_alpha_residual(restored, mark_box, eng.alpha)
    filled_after = score_alpha_residual(filled, mark_box, eng.alpha)

    assert int(np.count_nonzero(solved > ALPHA_FLOOR)) > 0
    assert after < before or filled_after <= 0.08


def test_reverse_alpha_generates_solved_candidates() -> None:
    eng = engine()
    clean = np.full((180, 260, 3), 245, np.uint8)
    mark_box = (50, 70, 160, 40)
    alpha = place_alpha(eng.alpha, clean.shape[:2], (52, 80, 156, 21))
    watermarked = overlay(clean, alpha, logo=(210.0, 210.0, 210.0))
    candidates = eng.reverse_alpha_candidates(watermarked, mark_box, roi_class="near_white")

    assert candidates
    assert any(candidate.name.startswith("sunsky_reverse_alpha_solved") for candidate in candidates)


def test_calibration_uses_real_template_samples() -> None:
    alpha, meta = build_calibrated_alpha(
        ROOT / "templates" / "watermark-template.png",
        ROOT / "templates",
    )

    assert alpha.shape == engine().alpha.shape
    assert meta["samples_used"] >= 5
    assert meta["method"] == "sample_crop_background_alpha_solve"


def test_missing_alpha_is_safe_noop(tmp_path: Path) -> None:
    eng = SunskyAlphaEngine(tmp_path / "missing-alpha.png", tmp_path / "missing-template.png")
    img = np.full((64, 128, 3), 240, np.uint8)
    assert not eng.alpha_available()
    assert eng.remove_best(img, (10, 20, 80, 16), roi_class="near_white") is None


def test_grayscale_bgra_and_unicode_io(tmp_path: Path) -> None:
    eng = engine()
    clean = np.full((120, 180, 3), 236, np.uint8)
    alpha = place_alpha(eng.alpha, clean.shape[:2], (35, 50, 110, 17))
    watermarked = overlay(clean, alpha)

    gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    bgra = cv2.cvtColor(watermarked, cv2.COLOR_BGR2BGRA)
    assert eng.remove_best(gray, (30, 44, 120, 28), roi_class="near_white") is not None
    assert eng.remove_best(bgra, (30, 44, 120, 28), roi_class="near_white") is not None

    unicode_path = tmp_path / "测试-sunsky.png"
    assert imwrite_unicode(unicode_path, watermarked)
    read_back = imread_unicode(unicode_path, cv2.IMREAD_COLOR)
    assert read_back is not None
    assert read_back.shape == watermarked.shape


def test_dark_cable_surface_does_not_create_pale_rectangle() -> None:
    eng = engine()
    clean = np.full((180, 280, 3), 45, np.uint8)
    cv2.line(clean, (20, 92), (260, 92), (8, 8, 8), 7)
    cv2.line(clean, (28, 107), (250, 107), (18, 18, 18), 3)
    alpha = place_alpha(eng.alpha, clean.shape[:2], (52, 80, 156, 21))
    watermarked = overlay(clean, alpha, logo=(185.0, 185.0, 185.0))
    candidate = eng.remove_best(watermarked, (50, 70, 175, 42), roi_class="thin_flex_cable")
    assert candidate is not None
    diff = cv2.absdiff(candidate.image, watermarked)
    outside = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)[candidate.alpha_map < ALPHA_FLOOR]
    assert int(np.max(outside)) <= 3
    assert float(np.mean(candidate.image[:, :, 0] > clean[:, :, 0] + 38)) < 0.020
