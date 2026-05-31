# ClearMark

ClearMark removes authorized `sunsky-online.com` watermarks from product-image
copies while keeping the source asset folder untouched.

Source assets:

```bash
/Users/alexkou/Documents/github/b2bweb/content/products/assets
```

Project root:

```bash
/Users/alexkou/Documents/openai/clearmark
```

## Safety Rules

- Source images are read-only inputs.
- Output paths inside `/Users/alexkou/Documents/github/b2bweb` are rejected.
- Cleaned images, masks, PDFs, review HTML, and manifests are generated output;
  they must not be committed to git.
- iPhone 14+ only filenames are skipped because supplier images are known clean.
- Cleaning commands require `--rights-confirmed`.
- Oversized or uncertain masks route to `needs_manual` instead of being applied.

## Current Design

ClearMark now follows a compact "visual truthfulness" workflow inspired by the
V10-V13 Mark Remover design, but implemented against this project's existing
single-file pipeline rather than a separate 100-tool rewrite.

Core principle:

```text
detect accurately -> repair conservatively -> publish only if post-clean evidence is clean
```

Implemented stages:

1. Candidate selection skips known-clean iPhone 14+ assets and samples only
   master images from the perceptual-hash inventory.
2. Detection uses OCR when enabled, template matching, low-contrast text-band
   matching, and a bright-background recall pass for faint marks. OCR now uses
   a domain-structure score: a match must look like `sunsky`/`sky` plus
   `online`/`.com`, so product text such as labels, steps, and random fragments
   no longer qualifies as a Sunsky watermark.
3. A V13-style layout guard detects text-dense instruction sheets and disables
   weak prior text-band detection there. Those images require direct OCR or
   strong template evidence before cleaning.
4. Each detected footprint is classified by the pixels underneath it:
   `plain_white`, `near_white`, `dark_product_surface`, `thin_flex_cable`,
   `complex_product_detail`, `text_or_label_area`, and related classes. High
   product-overlap regions use glyph-only repair attempts; broad box masks are
   blocked to avoid wiping flex cables, printed labels, or product contours.
5. Mask construction stays narrow around the detected text footprint, with a
   high-contrast fallback box only when the ROI is not product-risky.
6. Repair tries OpenCV Telea/Navier-Stokes first and optionally escalates to
   LaMa/IOPaint when available for high-residual cases.
7. The publish decision is driven by post-clean evidence: residual score,
   template residual, post-clean text components, post-clean detection count,
   dot-chain fragments, rectangular-band score, product-damage score, and OCR
   on the cleaned mark-box crop.
8. Review output shows original, mask overlay, result, and `diff x3` so visible
   bands, damaged product detail, or leftover glyph fragments are easy to spot.

The third-party V12 ideas that matter for this repo are now documented as the
target direction:

- one publish gate as the source of truth;
- no zero-metric or provenance-only passes;
- reject readable residuals, dot-chain fragments, obvious rectangular bands,
  and product-surface damage;
- do not let weak prior bands select watermarked-only samples in text-heavy
  layouts;
- classify the watermark ROI by product overlap before choosing broad vs.
  glyph-only masks;
- use cover-style fallback only as a clearly reported best effort;
- keep provenance in HTML/PDF/JSON so review findings are traceable.

The current script still reports `cleaned`, `needs_manual`, and `no_watermark`.
Future status work should split successful outputs into `clean_repaired` and
`clean_covered` only after the cover gate is implemented and visually honest.

## Commands

Create image inventory, buckets, and perceptual-hash duplicate groups:

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py inventory \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets
```

Run a 50-image watermarked-only review pilot and create a PDF compare:

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --watermarked-only \
  --preset review \
  --no-lama \
  --pdf \
  --out /Users/alexkou/Downloads/clearmark-sunsky-50 \
  --rights-confirmed
```

Send the generated PDF to Telegram after the pilot finishes:

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."

python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --watermarked-only \
  --preset review \
  --no-lama \
  --pdf \
  --telegram \
  --out /Users/alexkou/Downloads/clearmark-sunsky-50 \
  --rights-confirmed
```

Pilot mode uses OCR and optional LaMa by default for higher accuracy. For
`--watermarked-only`, OCR confirmation is the safest way to avoid including
non-watermarked product-detail false positives. Pass `--no-lama` for faster
review sampling; use `--no-ocr` only for diagnostic speed runs where the sampler
will conservatively drop weak prior-band detections.

Run the one-pass production workflow:

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py process \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --preset review \
  --workers 8 \
  --ocr \
  --rights-confirmed
```

One-pass workflow:

```text
for each non-iPhone-14+ master image:
    detect watermark using review preset
    confirm it is really sunsky-online.com, not product text/details
    if no confirmed detection -> record no_watermark and skip
    if confirmed detection    -> build narrow/risk-routed mask, repair, validate, record cleaned/needs_manual
duplicates:
    reuse master result and copy output if master was cleaned
```

For speed-only process runs, omit `--ocr`; weak prior detections will then be
skipped unless the canonical template evidence is strong enough.

## Output Layout

Each run creates:

```text
outputs/<run-id>/ or custom --out/
  cleaned/          cleaned image copies when post-clean QA passes
  masks/            black/white masks
  originals/        copied originals for review only
  overlays/         original images with red mask overlay
  diffs/            absolute visual difference, amplified 3x
  manifest.jsonl    one entry per selected source file
  summary.json      counts, settings, inventory stats, Telegram result if used
  review.html       side-by-side browser review
  compare.pdf       optional PDF compare when --pdf or --telegram is used
```

## Project Structure

```text
clearmark/
  README.md
  requirements.txt
  scripts/
    watermark_pipeline.py   detection, repair, QA, review HTML/PDF, Telegram send
  templates/
    sunsky-online.png       canonical watermark template
  outputs/                 ignored generated output
```

## QA Direction

The next code improvements should be incremental and evidence-driven:

1. Improve actual repair quality for difficult `thin_flex_cable`,
   `complex_product_detail`, and `text_or_label_area` cases. The detector now
   routes them safely; the remaining work is better pixel reconstruction.
2. Add a real `clean_covered` status only after cover output passes the same
   visual-fidelity gate as repaired output.
3. Add regression fixtures for OCR false positives, step-layout images, and
   dark product-overlap cases so future threshold changes cannot reintroduce
   clean-image selection or large product masks.

Do not implement a broad tiled-watermark removal pass unless fresh evidence
shows a real repeated watermark layer in the current dataset. Prior review
found that premise was not supported by the inspected images.

## Notes

The pipeline uses OpenCV DCT for perceptual hashes, so no `imagehash` package
is required. Optional OCR requires EasyOCR. Optional neural inpainting is used
only when `simple_lama_inpainting` or a local `iopaint` command is available.
