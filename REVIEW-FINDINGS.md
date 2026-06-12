# Clearmark — Pilot-Test-50 Review & Corrected Improvement Plan

> Author: review pass over `outputs/pilot-test-50/` (manifest.jsonl + visual inspection of originals/overlays/cleaned).
> This supersedes the premises in `PATCH-PLAN.md`. Read the "Corrections" section first — the existing plan targets a problem that does not exist in the data.

## Headline numbers

| Status | Count | Notes |
|---|---|---|
| no_watermark | 10 | includes **≥1 false negative** (missed mark) |
| cleaned | 6 | |
| needs_manual | 34 | |

Auto-clean rate on watermarked images: **6/40 ≈ 15%**.
But ~13 of the 34 `needs_manual` are **already visually clean** and were wrongly rejected by the gating logic. Fixing the gates alone lifts the true rate to **~45–50%** with zero change to detection or inpainting.

---

## Corrections to the existing PATCH-PLAN.md (verify before building)

The current `PATCH-PLAN.md` is built on two premises that the images do not support:

1. **"Layer B — tiled faint watermark covering 40–60%" does not exist.**
   The plan cites `100-sets-front-camera-...-iphone-13-series-4.jpg` as a tile-pattern source. That image (#1) has a **pure clean white background, no tiling**. Every image inspected has a *single* center "sunsky-online.com" mark only. → **P0 (FFT/alpha tile subtraction) and the "tiled residual inflates QA" part of P2 should be dropped.** They solve a non-problem and will introduce color shift on clean backgrounds.

2. **Keyboard/product legends are NOT being destroyed.**
   The plan cites `for-macbook-air-m4-...-keyboard-black-5.jpg` as evidence Telea/NS destroys the "command" key label. In the actual cleaned output the `esc / control / option / command` legends are **fully intact**. A better inpainter is still worth having (see Tier 2) but the cited failure mode is wrong.

The real QA problem is **over-conservative gating**, not residual tile detection. Details below.

---

## Root-cause analysis (all in `scripts/watermark_pipeline.py`, `clean_image()` ~L988–1100)

### Bug 1 — sharpness gate rejects good flat-fills  *(biggest single win)*
`SHARPNESS_MIN_RATIO = 0.30` (L69) + the `ratio >= 0.30` requirement at L1066 + the standalone `blurry` reason at L1084.

When a mark sits on a uniform background (dark PCB, white seamless) a correct inpaint produces a **smooth** patch → `sharpness_ratio` legitimately drops near 0. The pipeline reads this low ratio as a defect and flags `blurry`.

Evidence — these were flagged `needs_manual / blurry` but are **flawless cleans** (residual≈0, OCR finds nothing post-clean, `post_text_components=0`):
- `power-button-volume-button-flex-cable-for-iphone-12-pro-max-3` — res 0.072, shp 0.003 → watermark 100% gone, confirmed visually.
- `tft-lcd-screen-for-iphone-8-plus-black-3` — res 0.049, shp 0.001.
- `touch-panel-with-home-button-for-ipad-10-2...` — res 0.089, shp 0.004.
- `infrared-fpc-flex-cable-for-iphone-xr-2` — res 0.050.

### Bug 2 — provenance gate rejects by detector source, not result quality
`position_confident = not det.template.startswith("prior:")` (L1063). Any detection from the prior/OCR text-band path is forced to `needs_manual` (reason `low_position_confidence`) **even when the clean is perfect**. 9 cases hit this; several have residual < 0.10.
- `original-pcb-membrane-flex-cable-for-ipad-4-black-4` — res 0.066, `post_text_components=0`, `cleaned_detection_count=0` → clean, rejected purely because detection came from the prior path.

### Confirmed-good gates (keep these)
- `residual_visible_or_wrong_detection` (residual ≥ `CLEAN_VISIBLE_RESIDUAL_MAX=0.48`) fires correctly. Verified on `front-facing-camera-for-iphone-12-3` (res 0.659): the mark crosses the dark flex-cable/white boundary and the Telea fill leaves a real grey ghost. Genuine failure, correctly caught.

### Note on metrics
**All 34 `needs_manual` have `cleaned_detection_count == 0`** — the detector re-finds no watermark after cleaning. That signal is currently ignored in favor of sharpness/provenance. It should be a primary input.

---

## Tier 1 — Fix gating (cheap, no new deps, recovers ~13 images)

Rewrite the decision block in `clean_image()` (L1051–1099) so the clean/fail decision is driven by **post-clean evidence**, not sharpness or detector provenance.

Proposed rule:
```
PASS (cleaned) when:
    residual_score < 0.25
    AND template_residual_score < 0.30
    AND post_text_components <= 1
    (regardless of sharpness_ratio or det.template provenance)

FAIL (needs_manual, residual_visible) when:
    residual_score >= 0.45  OR  post_text_components >= 4

GRAY ZONE (0.25 <= residual < 0.45, or 2-3 text components):
    run the OCR confirmation step (below); pass if no watermark text, else manual.
```
- **Delete** the standalone `blurry` reason (L1084) and the `ratio >= 0.30` term in the `cleaned` condition (L1066).
- **Keep** `blurry_and_residual` (L1051) — low sharpness *with* high residual is still a real artifact signal.
- **Drop** `position_confident` from the gate (L1063, L1082); keep it only as a logged hint, not a blocker.
- Feed `cleaned_detection_count == 0` in as a positive signal.

**OCR confirmation step (gray zone only):** re-run the existing OCR reader on just the mark-box crop of the *cleaned* image; if no watermark-like text (`ocr_text_matches_watermark`) → pass. This is the authoritative check and avoids tuning-by-threshold guesswork.

Expected after Tier 1: auto-clean ~19/40 (~48%), false `needs_manual` rate < 20%.

---

## Tier 2 — Better inpainting for genuine residual (~13 images)

The genuine failures share a pattern: the mark overlaps **textured / high-contrast product** (camera modules, connectors, dark-cable/white-bg boundaries). Telea/NS smear there.

- Integrate **LaMa / IOPaint** (Apache-2.0) as an **escalation backend only**: run the cheap OpenCV path first; if `residual_score >= 0.40` after it, retry the same mask with LaMa and keep whichever scores lower. Keep OpenCV for flat backgrounds (it's already perfect there and faster).
- Fallback to OpenCV + `needs_manual` if LaMa is unavailable.
- Target images: the 13 high-residual cases (`front-facing-camera-*`, `embedded-display-port-*`, all `apple-watch-*-lcd-screen`, `3-pcs-set-back-camera-*`, `50-pcs-oca-*-44mm`, etc.).

**Mask fix (helps both paths):** the ghost in `front-facing-camera-12` indicates the glyph mask leaves a faint halo where the mark crosses a contrast edge. When a detection's `contrast_span` is high, use the dilated bold bounding-box mask (with feather) instead of the tight glyph mask so anti-aliased edges are fully covered.

---

## Tier 3 — Recall: faint marks on bright backgrounds (false negatives)

`earpiece-speaker-assembly-for-iphone-12-2` was classified `no_watermark` but **has a visible faint center "sunsky-online.com"** over the white area. This is the worst error class — it ships uncleaned.

Recent commits added CLAHE recall for faint marks on *dark* PCBs; the *bright/white* background analog is the open gap. Add a low-contrast detection pass over high-luminance regions (CLAHE on a white-background mask, lower `verify_score` threshold there). Then re-audit all 10 `no_watermark` outputs — target **zero false negatives**.

---

## Execution order

1. **Tier 1 gating rewrite** — highest ROI, no deps, ~half-day. Re-run pilot, confirm ~48% and inspect the newly-passed set.
2. **Tier 3 recall pass** — close false negatives (safety-critical; a missed watermark is worse than an over-flag).
3. **Tier 2 LaMa escalation + mask fix** — recovers the genuinely hard ~13.
4. Re-run pilot-test-50; target **>75% auto-clean, 0 false negatives**, spot-check 10 cleaned outputs.

## Test harness note
`review.html` is good but add a 4th "diff" pane (`abs(original-cleaned)`, ×3 gain) and surface the gate that fired per image — this makes Tier 1 regressions obvious at a glance. (This part of the old P3 is valid, keep it.)
