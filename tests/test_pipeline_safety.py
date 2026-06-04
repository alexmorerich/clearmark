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


def test_alpha_candidates_are_reserved_for_gate_evaluation(monkeypatch) -> None:
    eng = pipeline.sunsky_alpha_engine()
    assert eng is not None and eng.alpha_available()
    clean = np.full((180, 260, 3), 242, np.uint8)
    alpha = np.zeros(clean.shape[:2], dtype=np.float32)
    alpha[80:101, 52:208] = cv2.resize(eng.alpha, (156, 21), interpolation=cv2.INTER_AREA)
    watermarked = np.uint8(
        np.clip(alpha[:, :, None] * np.array((190.0, 190.0, 190.0), np.float32) + (1.0 - alpha[:, :, None]) * clean, 0, 255)
    )
    gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    det = pipeline.Detection(
        x=50,
        y=70,
        w=160,
        h=40,
        score=0.90,
        verify_score=0.90,
        template="watermark-template.png",
        scale=1.0,
        mark_box={"x": 50, "y": 70, "w": 160, "h": 40},
        mask_area_pct=0.8,
        text_score=0.90,
        text_components=14,
        contrast_span=20.0,
        line_dominance=0.20,
        confidence=0.95,
        roi_class="near_white",
    )

    monkeypatch.setattr(
        pipeline,
        "cleaned_crop_ocr_check",
        lambda *args, **kwargs: {"ocr_checked": True, "ocr_watermark": False, "ocr_watermark_score": 0.0},
    )
    monkeypatch.setattr(pipeline, "post_clean_detection_count", lambda *args, **kwargs: 0)

    _, _, meta = pipeline.clean_image(watermarked, gray, det, pipeline.load_templates(), ocr_reader=object())

    assert meta["alpha_candidates_generated"] > 0
    assert meta["alpha_candidates_evaluated"] > 0
    assert meta["best_alpha_after_over_all_candidates"] >= 0.0
    assert any(
        trace["alpha_alignment_score"] > 0
        for trace in meta.get("candidate_gate_trace", [])
        if trace["strategy"].startswith("sunsky_reverse_alpha")
    )


def test_residual_second_pass_uses_roi_specific_repairs(monkeypatch) -> None:
    image = np.full((120, 220, 3), 58, np.uint8)
    cv2.putText(image, "sunsky-online.com", (36, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (138, 138, 138), 1, cv2.LINE_AA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = pipeline.Detection(
        x=32,
        y=48,
        w=160,
        h=28,
        score=0.88,
        verify_score=0.86,
        template="watermark-template.png",
        scale=1.0,
        mark_box={"x": 32, "y": 48, "w": 160, "h": 28},
        mask_area_pct=0.7,
        text_score=0.90,
        text_components=12,
        contrast_span=16.0,
        line_dominance=0.25,
        confidence=0.92,
        roi_class="dark_product_surface",
    )

    cleanup_mask = np.zeros(gray.shape, np.uint8)
    cleanup_mask[58:62, 82:116] = 255
    roi_repair = image.copy()
    roi_repair[0, 0] = (77, 77, 77)

    monkeypatch.setattr(pipeline, "ENABLE_LAMA_ESCALATION", False)
    monkeypatch.setattr(pipeline, "sunsky_alpha_engine", lambda: None)
    monkeypatch.setattr(pipeline, "create_mask", lambda *args, **kwargs: (cleanup_mask, pipeline._mask_area(cleanup_mask)))
    monkeypatch.setattr(
        pipeline,
        "build_residual_cleanup_mask",
        lambda *args, **kwargs: (
            cleanup_mask,
            {"eligible": True, "reason": "unit_residual_evidence_mask", "category": "candidate_failed_residual_only"},
        ),
    )

    def fake_roi_repairs(original, candidate, mask, detection):
        return [(
            "dark_surface_low_alpha_scrub",
            roi_repair,
            mask,
            {
                "operator": "dark_surface_low_alpha_scrub",
                "dark_surface_scrub_used": True,
                "protected_edge_loss": 0.0,
                "repair_mask_area_pct": pipeline._mask_area(mask) * 100,
            },
        )]

    monkeypatch.setattr(pipeline, "roi_specific_repair_candidates", fake_roi_repairs)

    residual_metrics = {
        "residual_score": 0.55,
        "template_residual_score": 0.34,
        "post_text_score": 0.80,
        "post_text_components": 4,
    }
    clean_metrics = {
        "residual_score": 0.03,
        "template_residual_score": 0.02,
        "post_text_score": 0.01,
        "post_text_components": 0,
    }
    dot_metrics = {
        "dot_chain_score": 0.0,
        "dot_component_count": 0,
        "dot_horizontal_span": 0.0,
        "dot_component_area_ratio": 0.0,
        "dot_chain_fail": False,
        "component_mask": cleanup_mask,
    }
    band_metrics = {"band_gate_fail": False, "visible_band_score": 0.0, "band_luma_delta": 0.0, "band_edge_box": 0.0}
    product_metrics = {
        "product_gate_fail": False,
        "product_color_delta": 0.0,
        "product_edge_retention": 1.0,
        "product_blob_score": 0.0,
        "product_changed_area_ratio": 0.0,
    }
    alpha_metrics = {
        "alpha_checked": True,
        "alpha_template_residual_before": 0.04,
        "alpha_template_residual_after": 0.02,
        "alpha_residual_reduction": 0.70,
    }

    def fake_evaluate(original, candidate, mask, detection, templates, ocr_reader):
        metrics = clean_metrics if int(candidate[0, 0, 0]) == 77 else residual_metrics
        ocr_meta = {"ocr_checked": True, "ocr_watermark": False, "ocr_watermark_score": 0.0}
        gate = pipeline.final_publish_gate(metrics, 0, ocr_meta, dot_metrics, band_metrics, product_metrics, alpha_metrics)
        return (
            cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
            metrics,
            0,
            ocr_meta,
            dot_metrics,
            band_metrics,
            product_metrics,
            alpha_metrics,
            gate,
        )

    monkeypatch.setattr(pipeline, "evaluate_cleaned_output", fake_evaluate)

    cleaned, mask, meta = pipeline.clean_image(image, gray, det, pipeline.load_templates(), ocr_reader=object())

    assert cleaned is not None
    assert mask is not None
    assert meta["status"] == "cleaned"
    assert meta["residual_cleanup_eligible"] is True
    assert meta["second_pass_attempted"] is True
    assert meta["cleanup_strategy"] == "dark_surface_low_alpha_scrub"
    assert meta["roi_repair_operator"] == "dark_surface_low_alpha_scrub"
    assert meta["dark_surface_scrub_used"] is True


def test_solid_background_direct_cover_expands_beyond_tight_mask() -> None:
    image = np.full((320, 640, 3), 248, np.uint8)
    cv2.putText(image, "sunsky-online.com", (212, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (178, 178, 178), 1, cv2.LINE_AA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = pipeline.Detection(
        x=204,
        y=146,
        w=170,
        h=28,
        score=0.92,
        verify_score=0.91,
        template="ocr:unit",
        scale=1.0,
        mark_box={"x": 204, "y": 146, "w": 170, "h": 28},
        mask_area_pct=0.5,
        text_score=0.92,
        text_components=12,
        contrast_span=22.0,
        line_dominance=0.15,
        confidence=0.94,
        roi_class="plain_white",
    )
    tight_mask = np.zeros(gray.shape, np.uint8)
    tight_mask[156:164, 264:292] = 255

    repaired, cover_mask, area, meta = pipeline.solid_background_direct_cover_repair(
        image,
        gray,
        tight_mask,
        det,
        risky=False,
    )

    assert repaired is not None
    assert cover_mask is not None
    assert meta["solid_background_cover_used"] is True
    assert meta["solid_cover_target"] == "confirmed_text_line"
    assert area > pipeline._mask_area(tight_mask)
    assert int(np.median(repaired[150:172, 212:366])) >= 245
    assert int(np.min(repaired[157:164, 212:366])) >= 242


def test_solid_background_block_cover_copies_nearest_same_size_patch() -> None:
    image = np.full((340, 680, 3), 248, np.uint8)
    for y in range(image.shape[0]):
        image[y, :, :] = 246 + (y % 5)
    cv2.putText(image, "sunsky-online.com", (236, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (178, 178, 178), 1, cv2.LINE_AA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = pipeline.Detection(
        x=220,
        y=154,
        w=218,
        h=34,
        score=0.93,
        verify_score=0.92,
        template="ocr:unit",
        scale=1.0,
        mark_box={"x": 220, "y": 154, "w": 218, "h": 34},
        mask_area_pct=0.5,
        text_score=0.93,
        text_components=14,
        contrast_span=24.0,
        line_dominance=0.16,
        confidence=0.95,
        ocr_watermark_score=0.93,
        roi_class="near_white",
    )
    tight_mask = np.zeros(gray.shape, np.uint8)
    tight_mask[166:176, 292:326] = 255

    repaired, cover_mask, area, meta = pipeline.solid_background_block_cover_repair(
        image,
        gray,
        tight_mask,
        det,
        risky=False,
    )

    assert repaired is not None
    assert cover_mask is not None
    assert meta["solid_block_cover_used"] is True
    assert meta["solid_cover_target"] == "nearest_same_size_block"
    assert area > pipeline._mask_area(tight_mask)
    target = meta["solid_block_target_box"]
    donor = meta["solid_block_donor_box"]
    assert target["w"] == donor["w"]
    assert target["h"] == donor["h"]
    target_patch = repaired[target["y"]:target["y"] + target["h"], target["x"]:target["x"] + target["w"]]
    donor_patch = image[donor["y"]:donor["y"] + donor["h"], donor["x"]:donor["x"] + donor["w"]]
    center = target_patch[3:-3, 3:-3]
    donor_center = donor_patch[3:-3, 3:-3]
    assert np.array_equal(center, donor_center)
    outside = cv2.bitwise_not(cover_mask)
    assert int(np.count_nonzero(cv2.cvtColor(cv2.absdiff(image, repaired), cv2.COLOR_BGR2GRAY)[outside > 0])) == 0
    assert int(np.min(target_patch)) >= 246


def test_solid_background_block_cover_skips_product_label_area() -> None:
    image = np.full((220, 460, 3), 248, np.uint8)
    cv2.putText(image, "Battery", (175, 96), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "sunsky-online.com", (154, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (178, 178, 178), 1, cv2.LINE_AA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = pipeline.Detection(
        x=142,
        y=102,
        w=180,
        h=28,
        score=0.91,
        verify_score=0.90,
        template="ocr:unit",
        scale=1.0,
        mark_box={"x": 142, "y": 102, "w": 180, "h": 28},
        mask_area_pct=0.6,
        text_score=0.92,
        text_components=20,
        contrast_span=170.0,
        line_dominance=0.22,
        confidence=0.95,
        ocr_watermark_score=0.94,
        roi_class="text_or_label_area",
    )
    tight_mask = np.zeros(gray.shape, np.uint8)
    tight_mask[112:118, 208:250] = 255

    repaired, cover_mask, area, meta = pipeline.solid_background_block_cover_repair(
        image,
        gray,
        tight_mask,
        det,
        risky=True,
    )

    assert repaired is None
    assert cover_mask is None
    assert area == 0.0
    assert meta["reason"] == "text_label_area_block_cover_disabled"


def test_textured_panel_strip_clone_covers_full_ocr_line() -> None:
    image = np.full((280, 520, 3), 248, np.uint8)
    cv2.rectangle(image, (90, 110), (430, 190), (18, 22, 28), -1)
    yy, xx = np.indices((70, 238))
    texture = np.uint8(np.clip(190 + ((xx * 7 + yy * 13) % 33) - 16, 0, 255))
    panel = cv2.merge([texture, texture + 1, texture + 3])
    image[126:196, 140:378] = panel
    cv2.putText(image, "sunsky-online.com", (156, 151), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = pipeline.Detection(
        x=142,
        y=124,
        w=242,
        h=52,
        score=0.90,
        verify_score=0.86,
        template="ocr:crop:sunsky-online.com",
        scale=1.0,
        mark_box={"x": 142, "y": 124, "w": 242, "h": 52},
        mask_area_pct=1.2,
        text_score=0.95,
        text_components=40,
        contrast_span=180.0,
        line_dominance=0.40,
        confidence=0.96,
        ocr_watermark_score=0.92,
        roi_class="text_or_label_area",
        product_overlap=0.70,
    )
    mask, _ = pipeline.create_mask(gray, det, img=image, pad_x=8, pad_y=5, dilate_px=2, glyph=True)

    repaired, strip_mask, area, meta = pipeline.textured_panel_strip_clone_repair(
        image,
        gray,
        mask,
        det,
        risky=True,
    )

    assert repaired is not None
    assert strip_mask is not None
    assert meta["textured_panel_strip_clone_used"] is True
    assert area > pipeline._mask_area(mask) * 0.35
    target = meta["textured_strip_target_box"]
    donor = meta["textured_strip_donor_box"]
    assert target["w"] == donor["w"]
    assert target["h"] == donor["h"]
    assert target["w"] >= 180
    changed = cv2.cvtColor(cv2.absdiff(image, repaired), cv2.COLOR_BGR2GRAY)
    assert int(np.count_nonzero(changed[strip_mask > 0])) > 0


def test_solid_background_direct_cover_handles_dark_solid_surface() -> None:
    image = np.full((320, 640, 3), 248, np.uint8)
    cv2.rectangle(image, (80, 128), (520, 230), (42, 42, 42), -1)
    cv2.putText(image, "sunsky-online.com", (188, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = pipeline.Detection(
        x=170,
        y=160,
        w=230,
        h=34,
        score=0.93,
        verify_score=0.92,
        template="ocr:unit",
        scale=1.0,
        mark_box={"x": 170, "y": 160, "w": 230, "h": 34},
        mask_area_pct=0.7,
        text_score=0.93,
        text_components=14,
        contrast_span=44.0,
        line_dominance=0.18,
        confidence=0.95,
        ocr_watermark_score=0.93,
        roi_class="dark_product_surface",
        product_overlap=0.70,
    )
    tight_mask = np.zeros(gray.shape, np.uint8)
    tight_mask[174:184, 260:298] = 255

    repaired, cover_mask, area, meta = pipeline.solid_background_direct_cover_repair(
        image,
        gray,
        tight_mask,
        det,
        risky=True,
    )

    assert repaired is not None
    assert cover_mask is not None
    assert meta["solid_cover_target"] == "confirmed_text_line"
    assert area > pipeline._mask_area(tight_mask)
    changed = cv2.absdiff(image, repaired)
    assert int(np.count_nonzero(cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY)[cover_mask > 0])) > 0
    assert int(np.median(repaired[170:190, 188:390])) <= 62
