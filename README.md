# ClearMark

ClearMark is a local, evidence-first pipeline for detecting, removing, and reviewing the visible `sunsky-online.com` watermark from authorized product-image copies.

The project is built around one rule:

```text
confirm the Sunsky watermark position
-> clean only the confirmed watermark footprint
-> publish only if independent post-clean evidence is clean
```

Source assets are read-only input. ClearMark writes cleaned copies, failed attempts, masks, manifests, HTML review pages, PDFs, and optional Telegram review attachments to a separate output directory.

## Project Structure

```text
scripts/
  watermark_pipeline.py       main inventory, pilot, process, QA, and review workflow
  sunsky_alpha_engine.py      deterministic Sunsky reverse-alpha repair engine
  build_sunsky_alpha.py       reproducible alpha asset bootstrap/calibration utility
templates/
  watermark-template.png      canonical Sunsky text template
  sunsky-alpha.png            generated alpha map used by reverse-alpha repair
  sunsky-alpha-meta.json      alpha asset metadata and provenance
tests/
  test_sunsky_alpha_engine.py synthetic recovery and safety tests
  test_pipeline_safety.py      publish-gate and no-mark regression tests
```

## Execution Flow

```text
inventory images
-> skip known-clean iPhone 14+ assets
-> sample master files and avoid duplicates
-> detect possible Sunsky watermark locations
-> confirm Sunsky presence with OCR/template evidence
-> classify the region under the mark
-> generate repair candidates
   -> Sunsky reverse-alpha candidates
   -> thin alpha-edge cleanup
   -> near-white/background fill
   -> Telea / Navier-Stokes inpaint
   -> optional LaMa escalation
   -> residual-only cleanup
      -> residual evidence mask
      -> row/ring fill, thin inpaint, and ROI-specific repair candidates
   -> solid-background direct cover for confirmed marks on pure color areas
-> run the strict final publish gate
-> write cleaned/ only for gate-passed outputs
-> write attempts/ for failed best attempts
-> write review.html, compare.pdf, manifest.jsonl, summary.json
```

## Identifying The Watermark Position

ClearMark treats position detection as a three-step evidence problem: propose, confirm, then localize.

### Candidate Proposal

`watermark_pipeline.py` proposes mark locations from:

- EasyOCR full-image detections, when OCR is enabled.
- OCR crop re-detection around likely template/prior hits.
- Canonical template matching from `templates/watermark-template.png`.
- Low-contrast text-band priors for faint watermark recall.
- Bright-background recall for pale gray marks on near-white product photos.

Priors are recall helpers only. They do not become publishable evidence by themselves when OCR or template evidence disagrees. If Sunsky presence is not confirmed, the image is returned as `no_watermark` and does not enter the cleaning stage.

### Sunsky Text Confirmation

OCR text is scored by domain shape, not generic text likeness. The matcher looks for a Sunsky/sky-like token plus an online/.com-like token, while rejecting ordinary product labels and instructions.

Confirmed examples include OCR variants such as:

```text
sunsky-online.com
sunsky-onlinecom
sky-online.com
sunsky-oniine com
```

Rejected examples include product text such as:

```text
LCD Digitizer
Flex Cable
Stable Bracket
12mini
```

### OCR Crop Localization

When a broad detector finds the neighborhood, ClearMark crops around it and runs OCR again. The final repair box follows the text that OCR actually read, not a nearby high-contrast product edge.

This is important for images where the watermark crosses:

- dark flex cables;
- product labels;
- screws and connectors;
- screen assemblies;
- red, teal, blue, gray, or black product surfaces.

### Box Normalization

OCR can merge the watermark with nearby text, for example `sunsky-online 12mini`. ClearMark normalizes OCR boxes by:

- clamping the box toward the canonical Sunsky aspect ratio;
- anchoring left when trailing model tokens are present;
- keeping enough height for anti-aliased watermark halo;
- preventing expansion into unrelated product labels unless residual evidence later proves it is necessary.

### Region Classification

After localization, ClearMark classifies the pixels under the mark. The ROI class controls mask size and repair strategy.

| ROI class | Meaning |
| --- | --- |
| `plain_white` | mostly white, very low edge density |
| `near_white` | bright and low-detail |
| `low_texture_background` | smooth background or smooth product plane |
| `simple_product_surface` | product surface with moderate structure |
| `dark_product_surface` | black or dark product region |
| `thin_flex_cable` | line-dominant cable or connector structure |
| `complex_product_detail` | dense edges, screws, assemblies, or contours |
| `text_or_label_area` | real product text or label near the watermark |
| `unknown` | no confident class |

High-risk classes use narrower masks, thinner cleanup, and stricter product-damage review.

## Cleaning Strategies

ClearMark generates candidates, then lets the final gate decide. No cleaning method can publish on its own.

### Sunsky Reverse-Alpha Engine

The primary repair path is `SunskyAlphaEngine` in `scripts/sunsky_alpha_engine.py`.

The engine models the watermark as a semi-transparent fixed text overlay:

```text
watermarked = alpha * logo_color + (1 - alpha) * clean_image
```

It reverses that blend only where the alpha map is present:

```text
restored = (watermarked - alpha * logo_color) / clamp(1 - alpha, 0.25, 1.0)
```

Pixels outside the alpha footprint are not modified.

The engine uses:

- `templates/sunsky-alpha.png` as the registered Sunsky alpha asset;
- calibrated sample-crop alpha metadata from `templates/sunsky-alpha-meta.json`;
- shape-consistent NCC alignment inside the confirmed mark box;
- a small search over scale, x/y offset, alpha gain, and logo luma;
- per-image background alpha solving inside the aligned glyph support;
- polarity-aware glyph extraction for bright-on-dark and dark-on-light watermarks;
- a reserved gate-evaluation budget for alpha candidates so they are not crowded out by glyph/inpaint variants;
- candidate self-arbitration by residual evidence;
- safe no-op behavior when the alpha asset or alignment evidence is missing.

Candidate strategy names include:

```text
sunsky_reverse_alpha_aligned
sunsky_reverse_alpha_aligned_thin_ns
sunsky_reverse_alpha_aligned_thin_ns_r2
sunsky_reverse_alpha_solved
sunsky_reverse_alpha_solved_thin_ns
sunsky_reverse_alpha_solved_thin_ns_r2
sunsky_reverse_alpha_solved_bg_fill
```

`sunsky_reverse_alpha_solved` treats the stored alpha image as support, then solves the actual opacity from the current image and a local background estimate. On safe low-variance regions, `sunsky_reverse_alpha_solved_bg_fill` can replace only the solved glyph support with ring/background color. That candidate still goes through the same visible-band, product-damage, OCR, detector, dot-chain, and alpha residual gates.

### Thin Alpha-Edge Cleanup

Reverse-alpha removes the blended watermark first. A tiny Navier-Stokes pass can then run over the alpha edge footprint only.

Rules:

- radius 1 remains the conservative option for risky product regions;
- radius 2 is also generated as a stronger risky-region candidate when residual evidence remains;
- radius is 2 for safer low-texture regions;
- cleanup never uses a full mark-box rectangle;
- product detail, cables, and labels still have to pass product-damage QA.

### Residual-Driven Second Pass

If a first-pass candidate fails only because watermark evidence remains, ClearMark builds a second-pass mask with `build_residual_cleanup_mask()`. This mask combines the original glyph mask, post-clean residual text components, dot-chain evidence, template halo evidence, and OCR-supported text-line coverage. It is anisotropically dilated: wider horizontally along the Sunsky baseline and tighter vertically to avoid product damage.

The second pass is not allowed when the blocker is a visible band, product damage, missing metrics, uncertain detection, or an oversized mask. Eligible residual masks feed row/ring fill, residual inpaint, and the ROI-specific operators listed below. The final publish gate is run again after every second-pass candidate.

### Near-White Row Fill

For `plain_white`, `near_white`, and `low_texture_background`, ClearMark can use a row-local background fill. It estimates nearby luma/chroma from a clean context ring and fills only the watermark mask with small matched noise.

This avoids gray halos and avoids broad white rectangles.

### Solid Background Direct Cover

For confirmed Sunsky text on pure white, dark solid product surfaces, or other low-texture solid color areas, ClearMark generates a direct background-cover candidate. Instead of asking inpaint to infer the background from narrow glyph strokes, it expands to the OCR/template-supported glyph halo and left/right tails, then writes a local background estimate directly over that watermark footprint.

This strategy is intended for the exact residual pattern where faint `sunsky-online.com` glyphs remain on a plain white, black, gray, or colored solid surface. It does not bypass safety review: if the expanded cover touches product text, cable edges, connector details, or creates a visible band, the final gate keeps the image in `needs_manual`.

### Nearest Same-Size Solid Block Cover

For monochrome LCD panels, white product cards, smooth adhesive sheets, and other genuinely single-color backgrounds, ClearMark also generates a stronger rectangular cover candidate. It builds an OCR/template-confirmed text-line rectangle, trims that rectangle back to the same-color surface, searches the nearest above, below, left, and right positions for a same-size clean color block, and copies that block over the watermark line. The center of the mark is replaced directly; the block boundary is feathered only inside the confirmed rectangle so hard seams do not become false residual components.

This is only allowed when both the local context and the donor block are low-texture, low-edge, and color-consistent. It is disabled for `text_or_label_area` and rejected when the surrounding pixels look like cables, connector detail, product contours, or mixed surfaces. Manifest fields identify the decision:

```json
{
  "solid_block_cover_used": true,
  "solid_cover_target": "nearest_same_size_block",
  "solid_cover_background_source": "nearest_same_size_block",
  "solid_block_target_box": {"x": 0, "y": 0, "w": 0, "h": 0},
  "solid_block_donor_box": {"x": 0, "y": 0, "w": 0, "h": 0}
}
```

The final OCR, detector, residual, dot-chain, product-damage, and visible-band gates are unchanged. A copied block can become `cleaned/` only when those independent checks agree that no readable Sunsky mark or visible patch remains.

### Textured Panel Strip Clone

For small metal plates and textured panels where the OCR-confirmed watermark sits on a narrow horizontal strip, ClearMark can clone a same-width texture strip from the closest matching row above or below the mark. This is used when a glyph-only mask would leave most of the logo visible, but a broad rectangular inpaint would smear the product surface. The strategy records `textured_panel_strip_clone_used`, `textured_strip_target_box`, and `textured_strip_donor_box` in the manifest. It still cannot publish unless the final gate passes.

### Dark Surface Scrub

For `dark_product_surface`, ClearMark avoids pale fills. It samples nearby dark pixels, scrubs only low-alpha watermark residue, and preserves strong product edges.

Manifest evidence includes `dark_surface_scrub_used` when this path is selected.

### Thin Flex Cable Protected Cleanup

For `thin_flex_cable`, ClearMark builds a product-edge protection mask from dark connected components and strong edges. Cleanup is allowed only on low-contrast residual watermark pixels.

Review metrics include:

```json
{
  "protected_edge_loss": 0.0,
  "cable_silhouette_delta": 0.0
}
```

### Solid Color Surface Fill

For smooth colored surfaces such as red LCD backing, blue adhesive film, teal pads, or gray metal plates, ClearMark fits a local color plane and fills only the residual alpha/glyph footprint.

The visible-band gate remains strict.

### Repeated Object Clone

When a repeated object is confidently available nearby, ClearMark may clone from a similar neighboring component and feather only inside the residual watermark mask.

This is optional and requires high similarity. Shape mismatch is rejected by the final gate.

### Generic Inpaint And Optional LaMa

Telea, Navier-Stokes, and optional LaMa remain fallback candidates. They are not allowed to bypass OCR, dot-chain, product-damage, visible-band, or alpha-template residual checks.

## Quality Review Methods And Criteria

ClearMark keeps the final publish gate strict. The patch adds better candidates; it does not loosen approval thresholds to force more `cleaned/` outputs.

### Final Gate Evidence

A cleaned candidate must pass:

- residual visibility score;
- canonical template residual score;
- alpha-template residual score;
- post-clean OCR Sunsky check, which must actually run before publishing;
- post-clean detector count;
- dot-chain residual detection;
- visible rectangular band detection;
- product-damage detection;
- required metric validity checks.

If any required signal fails, the image is marked `needs_manual`. Runs with `--no-ocr` are useful for fast review diagnostics, but they cannot write automatic `cleaned/` outputs because the independent OCR double-check is unavailable.

### Residual Watermark Criteria

The result must not contain readable or template-matching Sunsky remnants.

Tracked fields include:

```json
{
  "residual_score": 0.0,
  "template_residual_score": 0.0,
  "alpha_template_residual_before": 0.0,
  "alpha_template_residual_after": 0.0,
  "alpha_residual_reduction": 0.0,
  "post_clean_ocr_score": 0.0,
  "post_text_components": 0,
  "dot_chain_score": 0.0
}
```

### Band Visibility Criteria

The repair must not create a visible rectangle, halo, smear, or row/box patch. The gate checks local luma delta and edge-box structure around the changed mask.

### Product Damage Criteria

The repair must not damage product contours, labels, dark surfaces, cables, or connector details. Product QA checks:

- color delta on product pixels;
- product edge retention;
- bright/dark blob formation;
- changed area ratio inside protected regions.

### Manifest And Review Fields

Each processed image records diagnostic fields such as:

```json
{
  "alpha_engine_used": true,
  "alpha_asset": "templates/sunsky-alpha.png",
  "alpha_asset_version": "sample_crop_background_alpha_solve",
  "alpha_alignment_score": 0.0,
  "alpha_candidate_count": 0,
  "alpha_candidates_generated": 0,
  "alpha_candidates_evaluated": 0,
  "best_alpha_after_over_all_candidates": 0.0,
  "alpha_best_gain": 1.0,
  "alpha_best_logo_bgr": [180, 180, 180],
  "thin_residual_inpaint": true,
  "solid_background_cover_used": false,
  "solid_cover_target": "",
  "solid_cover_area_pct": 0.0,
  "solid_cover_context_pixels": 0,
  "candidate_count": 0,
  "best_candidate_id": "",
  "first_pass_reason": "",
  "second_pass_attempted": false,
  "second_pass_strategy": "",
  "second_pass_mask_area_pct": 0.0,
  "second_pass_reason": "",
  "residual_cleanup_eligible": false,
  "residual_cleanup_reason": "",
  "roi_repair_operator": "",
  "dark_surface_scrub_used": false,
  "protected_edge_loss": 0.0,
  "cable_silhouette_delta": 0.0,
  "clone_similarity": 0.0,
  "final_blocker_type": "residual_watermark"
}
```

Review HTML and PDF header lines include strategy, ROI class, alpha alignment, alpha before/after residual, OCR score, template score, component count, and rejection reason. `summary.json` also records `reject_reason_histogram`, `alpha_candidates_generated_total`, and `alpha_candidates_evaluated_total` so a review run shows whether the dominant blocker is residual watermark, alpha residual, visible banding, product damage, or missing evidence.

## Commands

Build or refresh the calibrated alpha asset:

```bash
python3 scripts/build_sunsky_alpha.py --samples templates/
```

Without `--samples`, the script falls back to a bootstrap alpha from `templates/watermark-template.png`. With sample crops, `sunsky-alpha-meta.json` records the calibration method, sample directory, sample file names, and `samples_used`.

Run checks:

```bash
python3 -m py_compile scripts/watermark_pipeline.py scripts/sunsky_alpha_engine.py scripts/build_sunsky_alpha.py
python3 -m pytest -q
git diff --check
```

Run a 50-image review pilot:

```bash
python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --max-scan 700 \
  --watermarked-only \
  --preset review \
  --pdf \
  --telegram \
  --out /Users/alexkou/Downloads/clearmark-alpha-v1 \
  --rights-confirmed
```

## Safety Guarantees

ClearMark does not:

- modify source images;
- clean images whose Sunsky presence is not confirmed;
- write failed repairs to `cleaned/`;
- publish when OCR still reads Sunsky or when post-clean OCR was not checked;
- disable dot-chain, visible-band, product-damage, or alpha-template gates;
- use broad rectangle fills over product-overlap regions;
- strip metadata, remove invisible watermarks, regenerate images, or perform general AI-label removal.
