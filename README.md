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
V10-V12 Mark Remover design, but implemented against this project's existing
single-file pipeline rather than a separate 100-tool rewrite.

Core principle:

```text
detect accurately -> repair conservatively -> publish only if post-clean evidence is clean
```

Implemented stages:

1. Candidate selection skips known-clean iPhone 14+ assets and samples only
   master images from the perceptual-hash inventory.
2. Detection uses OCR when enabled, template matching, low-contrast text-band
   matching, and a bright-background recall pass for faint marks.
3. Mask construction stays narrow around the detected text footprint, with a
   high-contrast fallback box only when glyph masks are unsafe.
4. Repair tries OpenCV Telea/Navier-Stokes first and optionally escalates to
   LaMa/IOPaint when available for high-residual cases.
5. The publish decision is driven by post-clean evidence: residual score,
   template residual, post-clean text components, post-clean detection count,
   dot-chain fragments, rectangular-band score, and OCR on the cleaned mark-box
   crop.
6. Review output shows original, mask overlay, result, and `diff x3` so visible
   bands, damaged product detail, or leftover glyph fragments are easy to spot.

The third-party V12 ideas that matter for this repo are now documented as the
target direction:

- one publish gate as the source of truth;
- no zero-metric or provenance-only passes;
- reject readable residuals, dot-chain fragments, obvious rectangular bands,
  and product-surface damage;
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
  --rights-confirmed
```

One-pass workflow:

```text
for each non-iPhone-14+ master image:
    detect watermark using review preset
    if no detection -> record no_watermark and skip
    if detected     -> build narrow mask, repair, validate, record cleaned/needs_manual
duplicates:
    reuse master result and copy output if master was cleaned
```

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

1. Add a named `final_publish_gate()` that returns `publish_ok`, `status`, and
   explicit reject reasons for residual, band, product-damage, and geometry
   gates.
2. Add a dot-chain/broken-glyph residual detector inside the known watermark
   footprint.
3. Add rectangular-band scoring for every candidate, not only fallback covers.
4. Add product-damage scoring over the full changed region, especially for
   dark product surfaces and thin flex cables.
5. Only after those gates are trustworthy, introduce honest `clean_covered`
   status for best-effort surface reconstruction.

Do not implement a broad tiled-watermark removal pass unless fresh evidence
shows a real repeated watermark layer in the current dataset. Prior review
found that premise was not supported by the inspected images.

## Notes

The pipeline uses OpenCV DCT for perceptual hashes, so no `imagehash` package
is required. Optional OCR requires EasyOCR. Optional neural inpainting is used
only when `simple_lama_inpainting` or a local `iopaint` command is available.
