# Clearmark Patch Plan — Post Pilot-Test-50

> ⚠️ **SUPERSEDED — DO NOT START HERE.** A later review found this plan's two central premises are not supported by the pilot images:
> 1. There is **no tiled 40–60% watermark layer** — every image has only a single center mark (the cited tile source `100-sets-...-iphone-13-series-4.jpg` has a clean white background). → P0 and the tiled half of P2 solve a non-problem.
> 2. Keyboard/product legends are **not** destroyed by inpainting (the cited macbook image has intact legends).
>
> The real issue is over-conservative gating, not residual tiling. **Use `REVIEW-FINDINGS.md` instead.** Kept below for reference only.

## Context

Pilot ran 50 sunsky-online.com product images through the current pipeline.
Results: 10 no-watermark, 5 auto-cleaned, 35 needs-manual. **12.5% auto-clean rate on watermarked images.**

The core problem: sunsky images have **two watermark layers** and the pipeline only handles one.

---

## Layer A — Bold Center Watermark

Single "sunsky-online.com" in medium-opacity text, positioned near image center.
Covers ~0.7–2% of image area. **Currently detected and inpainted.**

## Layer B — Tiled Faint Watermark (NOT HANDLED)

Small "sunsky-online.com" repeated diagonally across the entire image at low opacity (~5–15% alpha).
Covers 40–60% of image. **Every "cleaned" output still has this fully visible.**

---

## P0 — Tiled Watermark Removal (alpha subtraction)

**Goal:** Remove Layer B before running the existing bold-watermark pipeline.

### Steps

1. **Extract the tile pattern**
   - Take 3–5 sunsky images with pure-white backgrounds (e.g. `100-sets-front-camera-dustproof-sponge-foam-pads-small-ring-for-iphone-13-series-4.jpg`)
   - Crop regions that are background-only (no product pixels)
   - Average them to isolate the repeating watermark pattern with sub-pixel noise removed
   - Save as `templates/tiled-watermark-pattern.png`

2. **Detect tile presence**
   - Compute FFT of input image; look for periodic peaks matching the tile spacing
   - Or: normalized cross-correlation of extracted tile pattern against the image at known repeat intervals
   - Threshold: if correlation > 0.25 at 3+ expected grid points, tile layer is present

3. **Estimate per-image alpha**
   - Sample 20+ background pixels (high-luminance, low-saturation regions)
   - For each sample: `alpha = (white - observed) / (white - watermark_color)`
   - Use median alpha as the blend factor (expect 0.05–0.15)
   - Clamp outliers; reject if variance is too high (means non-uniform watermark)

4. **Subtract**
   ```python
   # For each pixel where tile pattern is non-white:
   clean = (original.astype(float) - alpha * tile_pattern) / (1.0 - alpha)
   clean = np.clip(clean, 0, 255).astype(np.uint8)
   ```
   - Apply only to pixels where the tile pattern has content (mask from the extracted pattern)
   - Blend edges with a 1–2px feather to avoid hard transitions

5. **Integration point**
   - Run tile removal as the **first step** in `clean_image()`, before bold-watermark inpainting
   - Pass the de-tiled image to the existing mask/inpaint pipeline

### Acceptance criteria
- Tiled text no longer visible on white/light backgrounds
- No color shift >2 delta-E on product pixels
- Works on at least 90% of sunsky images that have the tiled layer

---

## P1 — Deep Inpainter for High-Detail Regions

**Goal:** Stop destroying product detail (keyboard legends, PCB traces, connector pins) when inpainting the bold watermark.

### Problem

OpenCV Telea/NS inpainting fills from surrounding pixels. Works on smooth backgrounds, fails on textured regions. See: `for-macbook-air-m4-13-inch-a3240-us-version-keyboard-black-5.jpg` — the "command" key label is destroyed after inpainting.

### Steps

1. **Add region complexity classifier**
   - After generating the inpaint mask, compute Laplacian variance and edge density in the masked region
   - Threshold: if Laplacian variance > 500 or edge pixel ratio > 0.15, flag as high-detail

2. **Integrate LaMa inpainter**
   - Use `lama-cleaner` or the original LaMa checkpoint (Apache 2.0 license)
   - Wrap as an optional backend in `clean_image()`:
     ```python
     if region_complexity == "high":
         result = lama_inpaint(de_tiled_image, mask)
     else:
         result = cv2.inpaint(de_tiled_image, mask, radius, method)
     ```
   - LaMa handles large irregular masks and preserves texture/structure

3. **Fallback**
   - If LaMa is not installed or inference fails, fall back to current OpenCV path
   - Log a warning so the image gets flagged for review

### Acceptance criteria
- Text/labels under the bold watermark are preserved after inpainting
- No new blurring artifacts on high-detail product images
- Inference time < 2s per image on CPU (or GPU if available)

---

## P2 — Fix QA Scoring

**Goal:** Bring the auto-clean pass rate from 12.5% to >60%.

### Problem

The `visible` residual score re-detects the tiled watermark (Layer B) as leftover signal, inflating scores and causing false `needs_manual` flags. Many images with clean bold-watermark removal score visible > 0.48 purely because of the tiled layer.

### Steps

1. **Scope residual scoring to the inpaint region only**
   - Current: `residual_quality_metrics` may be measuring the full image or a region that includes tiled watermark
   - Fix: Compute `template_residual` and `post_text_score` strictly within the bold-watermark mask bounding box, not the surrounding area

2. **Separate metrics for each layer**
   - Add `tiled_residual_score` (post-subtraction correlation with tile pattern)
   - Keep `visible` for bold-watermark region only
   - Combined QA: both must pass independently

3. **Tune thresholds after P0/P1**
   - Re-run pilot-test-50 with the improved pipeline
   - Empirically set thresholds based on the new score distributions
   - Target: `visible < 0.20` for bold region, `tiled_residual < 0.10` for tile layer

### Acceptance criteria
- Images that look visually clean pass QA automatically
- False `needs_manual` rate drops below 20%

---

## P3 — Improve Review/Debugging UX

**Goal:** Make it faster to diagnose failures.

### Steps

1. **Add rejection reason to review.html metadata**
   - Currently shows: `needs_manual | glyph_medium_ns | mask 0.64% | visible 0.66 | template 0.28`
   - Add: `rejected_by: visible_threshold (0.66 > 0.48)` or `rejected_by: no_clean_produced (mask_area_exceeded)`

2. **Show which gate failed for images with no cleaned output**
   - Items #20, #25, #39 have overlays but no cleaned image and no explanation
   - Log the specific pre-inpaint check that rejected them

3. **Add a 4th column to the review grid: difference map**
   - `abs(original - cleaned)` amplified 3x, showing exactly what changed
   - Makes it trivial to spot both successful removal and collateral damage

---

## Execution Order

```
P0 (tile removal)  ──→  P2 (fix QA scoring)  ──→  re-run pilot
       │
       └── P1 (LaMa inpainter) can be done in parallel
                                P3 (review UX) can be done anytime
```

P0 is the prerequisite for everything — without it, no output is truly clean regardless of how well the bold watermark is handled.

---

## Test Plan

After implementing P0+P1+P2:

1. Re-run the same 50-image pilot set
2. Target metrics:
   - Auto-clean rate: >60% (up from 12.5%)
   - No visible tiled watermark remnants on white/light backgrounds
   - No product detail damage on high-complexity images
   - Zero false `no_watermark` classifications
3. Spot-check 10 random cleaned outputs visually for quality
