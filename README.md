# ClearMark

ClearMark is a local, evidence-first pipeline for detecting, removing, and
reviewing the `sunsky-online.com` watermark from authorized product-image
copies. It is built for the B2B asset library at:

```bash
/Users/alexkou/Documents/github/b2bweb/content/products/assets
```

The source asset directory is read-only input. ClearMark writes cleaned copies,
failed attempts, masks, manifests, HTML reports, PDFs, and Telegram attachments
to a separate output directory such as `/Users/alexkou/Downloads/...`.

The core rule is simple:

```text
confirm the watermark position -> clean only that footprint -> publish only if post-clean evidence is clean
```

If any required evidence is weak, the image is marked `needs_manual`. A failed
best attempt may be written to `attempts/` for visual review, but it is not a
publishable cleaned image.

## Current Design

ClearMark is intentionally conservative. It optimizes for not shipping bad
cleaned images:

- Do not include images without confirmed Sunsky watermark text.
- Do not use broad masks when the mark overlaps product detail.
- Do not trust a repair method just because it ran successfully.
- Do not place failed attempts in `cleaned/`.
- Do not modify or overwrite source images.

The pipeline has three major responsibilities:

1. Locate the actual watermark text position.
2. Repair the watermark footprint with the safest available strategy.
3. Review the cleaned result with independent post-clean quality gates.

## Processing Workflow

```text
image inventory
  -> skip known-clean iPhone 14+ files
  -> sample master images, avoiding duplicates
  -> detect Sunsky watermark candidates
  -> localize the true text footprint
  -> confirm watermark presence
  -> classify the watermark region
  -> build a narrow mask
  -> generate repair candidates
  -> optionally clean residual glyph fragments
  -> run final publish gate
  -> write cleaned/ or attempts/
  -> generate review.html, compare.pdf, manifest.jsonl, summary.json
```

## Watermark Position Detection

Watermark position accuracy is the most important part of the project. If the
mask is wrong, every cleaning method becomes dangerous. ClearMark therefore
separates detection into proposal, confirmation, and localization.

### 1. Candidate Proposal

`detect_watermark()` builds initial candidates from several sources:

- Full-image EasyOCR detections when OCR is enabled.
- Canonical `sunsky-online.com` template matching.
- Low-contrast text-band priors for faint grey marks.
- Bright-background recall for watermark text on white product photos.

The low-contrast priors are recall helpers only. They are allowed to suggest
where a watermark might be, but they should not be the final authority when OCR
can read the domain text.

### 2. Domain-Specific OCR Scoring

OCR text is scored by `watermark_text_score()`. The matcher is deliberately
specific to the Sunsky domain shape:

```text
sunsky/sky-like token + online/.com-like token
```

This accepts OCR variants commonly produced by faint watermarks:

```text
sunsky-online.com
sunsky-onlinecom
S unsky-online.co
sky-online.com
~sunsky-oniine com
Junsky-online com
```

It rejects generic product or instruction text such as:

```text
More product details
Stable Bracket
Flex Cable
12mini
LCD Digitizer
```

### 3. OCR Crop Localization

Some images have the real watermark above a dark cable or product surface. The
old failure mode was:

```text
OCR confirms "sunsky-online.com" in the crop,
but the mask lands on dark product printing below the text.
```

ClearMark fixes that by running `ocr_crop_watermark_detections()` after initial
candidate proposal. It takes a beam of likely neighborhoods, crops around each
one, runs OCR again, and turns the actual OCR text box into a high-priority
`ocr:crop:*` detection.

This means the final repair target follows the text that OCR actually read, not
the high-contrast product detail that happened to score well.

### 4. OCR Box Normalization

EasyOCR sometimes merges the watermark with nearby product labels, for example:

```text
sunsky-online 12mini
```

`ocr_mark_box_from_points()` normalizes OCR boxes before they become repair
targets:

- It keeps the box centered on the domain text line.
- It clamps very wide OCR lines toward the canonical Sunsky aspect ratio.
- It anchors left when trailing model tokens are detected.
- It allows a slightly larger detection box for OCR localization while keeping
the final repair mask area guarded.

This prevents the mask from covering nearby labels such as `12mini` unless the
watermark genuinely overlaps them.

### 5. Candidate Ranking And NMS

Detections are ranked with `detection_rank()`:

- Direct OCR and OCR-crop detections rank highest.
- Template detections rank next.
- Prior text-band detections rank lowest, especially on dark product surfaces,
  thin flex cables, complex detail, or text/label areas.

Non-maximum suppression keeps only distinct candidates. This avoids cleaning
several overlapping versions of the same mark.

### 6. Presence Confirmation

`confirm_watermark_presence()` decides whether an image belongs in a
watermarked-only pilot:

- Direct OCR confirmation passes.
- OCR crop confirmation passes.
- Without OCR, only strong canonical-template evidence passes.
- Text-dense layouts require direct evidence.

If presence is not confirmed, the image is recorded as `no_watermark` and is not
included in a watermarked-only 50-image review.

## Region Classification

After the mark box is selected, `estimate_product_overlap_v13()` classifies what
is under the watermark. The classifier uses interior pixels, not detector type.

It measures:

- mean and standard deviation of luma;
- dark, non-white, and white pixel ratios;
- Canny edge density;
- local text-like components;
- horizontal line dominance;
- contrast span.

Current ROI classes:

| Class | Meaning |
| --- | --- |
| `plain_white` | mostly white, very low edge density |
| `near_white` | bright, low-detail background |
| `low_texture_background` | smooth background or product plane |
| `simple_product_surface` | product surface with moderate structure |
| `dark_product_surface` | black or dark product region |
| `thin_flex_cable` | thin dark cable or line-dominant structure |
| `complex_product_detail` | high edge/contrast product detail |
| `text_or_label_area` | real product text or label nearby |
| `unknown` | no confident class |

High-risk classes force conservative mask and repair choices.

## Mask Strategy

ClearMark uses narrow masks first. The goal is to cover the watermark glyphs,
not a large rectangle around them.

### Canonical Glyph Mask

For OCR, OCR-crop, prior, and canonical template detections, `create_mask()` now
uses the canonical Sunsky ink shape from:

```text
templates/watermark-template.png
```

The canonical ink is scaled into the localized mark box and then lightly
dilated. This avoids the earlier failure where local contrast inside a dark
product region selected product printing or cable edges instead of the faint
watermark.

### Mask Variants

The cleaning loop tries several mask variants:

- `glyph_tight`
- `glyph_medium`
- `glyph_strong`
- tight and medium box masks only on safer low-risk regions
- high-contrast box masks only when the ROI is not product-risky

Risky ROI classes use glyph-only variants:

```text
dark_product_surface
thin_flex_cable
complex_product_detail
text_or_label_area
text-dense layouts
```

### Mask Area Guards

Mask size is limited by:

- `MAX_MASK_AREA`
- `OCR_DETECTION_MAX_AREA`
- `PILOT_MASK_AREA`
- combined multi-detection mask area checks

The OCR detection box may be larger than a normal candidate box because it is a
positioning aid. The actual repair mask still has to pass the stricter mask
area gates.

## Cleaning Strategies

ClearMark currently uses a compact strategy bank. It does not try to hide bad
repairs by calling them clean.

### 1. Near-Area Background Fill

`near_area_background_fill_repair()` is the first choice for pure white,
near-white, and low-texture background pixels.

Instead of asking inpainting to hallucinate a plain background, it:

1. Builds a context window around the mark.
2. Excludes the watermark mask from that window.
3. Samples the nearby clean background ring.
4. Estimates local color and tiny texture/noise.
5. Feather-blends that background into only the background part of the mask.

On product-overlap pixels, it uses a much smaller inpaint pass and refuses broad
white fills. This directly addresses the common case where the right answer is
to imitate the nearby clean background, not smear the product.

### 2. OpenCV Telea Inpaint

Telea is tried with the current mask variant. It works best on smooth
backgrounds and small glyph masks. It is not automatically trusted; it must pass
the final QA gate.

### 3. OpenCV Navier-Stokes Inpaint

Navier-Stokes is tried as another classical candidate. It can behave better on
some gradients and edges, but it is judged by the same QA metrics.

### 4. Optional LaMa / IOPaint Escalation

If enabled and available, LaMa/IOPaint can be attempted for high-residual
cases. Review pilots normally use `--no-lama` for speed. Production-quality
experiments can enable it when runtime is less important.

### 5. Residual Component Cleanup

If the first selected repair still has broken glyphs or dot-chain residue,
ClearMark may attempt a second small cleanup:

- `cleanup_residual_components_with_ring_fill()`
- `cleanup_residual_components_with_inpaint()`

These methods operate on detected residual components, not on the full
watermark rectangle. They are only attempted when the blocker is residual text
and not product damage or a visible band.

## Cleaning Quality Review

ClearMark treats review as a publish decision, not a diagnostic afterthought.
The final output must pass independent checks after cleaning.

### Final Publish Gate

`final_publish_gate()` returns `cleaned` only if all required checks pass:

| Check | Criterion |
| --- | --- |
| Metrics valid | Required QA metrics exist and are numeric |
| Residual score | `residual_score <= FINAL_RESIDUAL_MAX` |
| Template residual | `template_residual_score <= FINAL_TEMPLATE_MAX` |
| Text components | `post_text_components <= FINAL_TEXT_COMPONENTS_MAX` |
| Re-detection | post-clean detector count is zero |
| OCR double-check | cleaned mark-box crop does not still read Sunsky |
| Dot-chain gate | broken glyph fragments do not form a readable row |
| Band gate | repair does not create a visible rectangular band |
| Product gate | product detail is not damaged beyond thresholds |

If any check fails, status is `needs_manual`.

### Residual Criteria

Residual text is measured inside the known watermark footprint:

- `residual_score` combines template residue, text-likeness, and component
  count.
- `template_residual_score` reruns canonical template matching after cleaning.
- `post_text_components` counts text-like connected components remaining in the
  mark box.
- `post_clean_detection_count` detects whether a new watermark candidate still
  exists after repair.

### OCR Double-Check

`cleaned_crop_ocr_check()` crops around the cleaned mark box and reruns OCR.

The gate fails if OCR sees a domain-like Sunsky string above the suspect
threshold, even if OCR confidence is low. This catches cases where a repair
visually smears the word but still leaves readable fragments like:

```text
sunsky-online.co
sky-online.com
sunsky-onlin
```

### Dot-Chain / Broken-Glyph Gate

`residual_component_metrics()` searches a horizontally expanded mark footprint
for small aligned residual components. It fails when the fragments form a row
that a human can still read as watermark text.

### Visible Band Gate

`detect_rectangular_band_visibility()` compares original and candidate pixels
around the changed region. It rejects obvious rectangular bands or hard luma
boundaries caused by a repair.

### Product Damage Gate

`detect_product_damage_v13()` evaluates product-overlap changes:

- color delta;
- edge retention;
- bright or dark blob score;
- changed area ratio.

Dark product surfaces and thin flex cables have stricter handling because white
or pale fills are especially visible there.

### Review Outputs

Each review run writes:

```text
originals/        source copies for review
overlays/         original plus red mask overlay
cleaned/          publish-gate-passed outputs only
attempts/         failed best attempts for inspection
masks/            binary masks
diffs/            amplified visual diff
manifest.jsonl    per-image metadata
summary.json      run summary
review.html       browser review
compare.pdf       PDF review and Telegram attachment
```

The HTML/PDF columns are:

```text
Original | Mask overlay | Result | Diff x3
```

For `needs_manual`, the result column is labeled `Attempt (failed QA)`. Those
files are review artifacts only.

## Status Semantics

| Status | Publishable | Meaning |
| --- | --- | --- |
| `cleaned` | yes | Final publish gate passed; file is written to `cleaned/` |
| `needs_manual` | no | Detection exists, but cleaning failed QA; best attempt may be in `attempts/` |
| `no_watermark` | no action | No confirmed Sunsky watermark |
| `skipped` | no action | Known-clean iPhone 14+ image |
| `cleaned_duplicate` | yes | Duplicate reused a cleaned master result |
| `duplicate_no_action` | no action | Duplicate skipped because the master was not cleaned |

## Commands

### Install

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 -m pip install -r requirements.txt
```

Optional OCR requires EasyOCR. Optional neural inpainting requires
`simple_lama_inpainting` or a local `iopaint` command.

### 50-Image Review Pilot

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --max-scan 700 \
  --watermarked-only \
  --preset review \
  --no-lama \
  --pdf \
  --out /Users/alexkou/Downloads/clearmark-sunsky-50 \
  --rights-confirmed
```

### Send Review PDF To Telegram

```bash
set -a
source /Users/alexkou/.claude/channels/telegram/.env
set +a

python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --max-scan 700 \
  --watermarked-only \
  --preset review \
  --no-lama \
  --pdf \
  --telegram \
  --telegram-chat-id 8339510717 \
  --out /Users/alexkou/Downloads/clearmark-sunsky-50 \
  --rights-confirmed
```

### One-Pass Processing

```bash
python3 scripts/watermark_pipeline.py process \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --preset review \
  --ocr \
  --rights-confirmed
```

One-pass logic:

```text
for each non-iPhone-14+ master image:
    detect and localize the Sunsky watermark
    if no confirmed watermark -> no_watermark
    if confirmed watermark    -> repair and QA
    if publish gate passes    -> cleaned/
    otherwise                 -> attempts/ + needs_manual

duplicates:
    reuse master output only when master status is cleaned
```

## Manifest Fields

Useful fields in `manifest.jsonl`:

- `file`
- `status`
- `presence_reason`
- `presence_score`
- `strategy`
- `mask_area_pct`
- `residual_score`
- `template_residual_score`
- `post_text_components`
- `post_clean_detection_count`
- `sunsky_check_pass`
- `post_clean_ocr_score`
- `roi_class`
- `product_overlap`
- `cleanup_attempted`
- `cleanup_strategy`
- `reason`
- `detection.mark_box`
- `detection.template`

These fields are the first place to look when a mask is misplaced or a cleaned
result is rejected.

## Project Structure

```text
clearmark/
  README.md
  requirements.txt
  scripts/
    watermark_pipeline.py
  templates/
    watermark-template.png
  outputs/
```

`outputs/` is ignored generated output. Production source assets live outside
this repo and are never edited.

## Validation Checklist

Before committing code:

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 -m py_compile scripts/watermark_pipeline.py
git diff --check
```

Before trusting a visual-quality change:

```bash
python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --max-scan 700 \
  --watermarked-only \
  --preset review \
  --no-lama \
  --pdf \
  --out /Users/alexkou/Downloads/clearmark-review \
  --rights-confirmed
```

Then inspect:

- mask overlay position;
- whether `cleaned/` contains only publish-gate-passed files;
- whether `attempts/` explains failures clearly;
- OCR residual and dot-chain metrics;
- product-damage metrics on dark surfaces and flex cables.

## Operating Rules

- Never push generated cleaned images, attempts, reports, or PDFs to GitHub.
- Never write output into `/Users/alexkou/Documents/github/b2bweb`.
- Keep OCR enabled for official review pilots.
- Use `--no-lama` when speed matters; enable LaMa only for quality experiments.
- Treat `needs_manual` as a failed output, not as a cleaned image.
