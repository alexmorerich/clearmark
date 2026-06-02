from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import watermark_pipeline as pipeline  # noqa: E402


def valid_gate_inputs() -> tuple[dict, dict, dict, dict]:
    metrics = {
        "residual_score": 0.0,
        "template_residual_score": 0.0,
        "post_text_score": 0.0,
        "post_text_components": 0,
    }
    dots = {"dot_chain_fail": False}
    bands = {"band_gate_fail": False}
    product = {"product_gate_fail": False}
    return metrics, dots, bands, product


def test_publish_gate_rejects_when_post_clean_ocr_not_checked() -> None:
    metrics, dots, bands, product = valid_gate_inputs()
    gate = pipeline.final_publish_gate(
        metrics,
        0,
        {"ocr_checked": False, "ocr_watermark": None, "ocr_watermark_score": 0.0},
        dots,
        bands,
        product,
        {"alpha_checked": True, "alpha_template_residual_before": 0.04, "alpha_template_residual_after": 0.02},
    )

    assert not gate["publish_ok"]
    assert gate["status"] == "needs_manual"
    assert "post_clean_ocr_unchecked" in gate["reject_reasons"]


def test_publish_gate_rejects_readable_alpha_residual() -> None:
    metrics, dots, bands, product = valid_gate_inputs()
    gate = pipeline.final_publish_gate(
        metrics,
        0,
        {"ocr_checked": True, "ocr_watermark": False, "ocr_watermark_score": 0.0},
        dots,
        bands,
        product,
        {
            "alpha_checked": True,
            "alpha_template_residual_before": 0.53,
            "alpha_template_residual_after": 0.105,
            "alpha_residual_reduction": 0.80,
        },
    )

    assert not gate["publish_ok"]
    assert gate["status"] == "needs_manual"
    assert "alpha_template_residual" in gate["reject_reasons"]


def test_unconfirmed_detection_never_enters_cleaning(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "plain.jpg"
    img = np.full((120, 180, 3), 242, np.uint8)
    cv2.imwrite(str(image_path), img)

    det = pipeline.Detection(
        x=35,
        y=50,
        w=90,
        h=14,
        score=0.60,
        verify_score=0.50,
        template="prior:text_band",
        scale=1.0,
        mark_box={"x": 35, "y": 50, "w": 90, "h": 14},
        mask_area_pct=0.5,
        text_score=0.80,
        text_components=10,
        confidence=0.60,
    )

    monkeypatch.setattr(pipeline, "detect_watermark", lambda *args, **kwargs: [det])
    monkeypatch.setattr(
        pipeline,
        "confirm_watermark_presence",
        lambda *args, **kwargs: {
            "presence_confirmed": False,
            "presence_reason": "unit_test_unconfirmed",
            "presence_score": 0.0,
        },
    )

    def fail_clean(*args, **kwargs):
        raise AssertionError("unconfirmed detections must not be cleaned")

    monkeypatch.setattr(pipeline, "clean_all_detections", fail_clean)

    row = pipeline.process_file(
        image_path,
        [],
        "review",
        tmp_path / "out",
        review=True,
        ocr_reader=None,
        require_presence_confirmed=False,
        write_no_watermark_review=False,
    )

    assert row["status"] == "no_watermark"
    assert row["presence_confirmed"] is False
    assert "review_cleaned" not in row
    assert not (tmp_path / "out" / "cleaned").exists()
    assert not (tmp_path / "out" / "attempts").exists()
