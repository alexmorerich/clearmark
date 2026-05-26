# ClearMark

Goal: remove authorized `sunsky-online.com` watermarks from product images without ever modifying the source asset folder.

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
- All generated files stay under this project by default: `outputs/<run-id>/`.
- Output paths inside the source repo are rejected.
- iPhone 14+ only filenames are skipped because supplier images are known clean.
- Cleaning commands require `--rights-confirmed`.
- Oversized masks are rejected into `needs_manual` instead of being applied.

## Commands

Create image inventory, buckets, and perceptual-hash duplicate groups:

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py inventory \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets
```

Run a small QA pilot with original / mask / cleaned preview HTML.
Pilot mode uses OCR by default for higher watermark-position accuracy. Pass
`--no-ocr` only for a faster heuristic-only check.

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --preset review \
  --out /Users/alexkou/Documents/openai/clearmark/outputs/pilot-test-50 \
  --rights-confirmed
```

Output images for this test are written here:

```text
/Users/alexkou/Documents/openai/clearmark/outputs/pilot-test-50/
  originals/   copied source images for review only
  overlays/    original images with red mask overlay
  masks/       black/white mask images
  cleaned/     cleaned image copies when auto-clean passed QA
  review.html  side-by-side original / mask overlay / result review page
```

Open this file after the pilot finishes:

```text
/Users/alexkou/Documents/openai/clearmark/outputs/pilot-test-50/review.html
```

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
    if detected     -> build narrow mask, safety-check, clean, validate
duplicates:
    reuse master result and copy output if master was cleaned
```

## Output Layout

Each run creates:

```text
outputs/<run-id>/
  cleaned/          cleaned image copies
  masks/            black/white masks
  originals/        copied originals for pilot review only
  overlays/         original images with red mask overlay
  failed/           optional review artifacts for needs_manual
  manifest.jsonl    one entry per source file
  summary.json      counts, settings, buckets, duplicate stats
  review.html       pilot contact sheet when applicable
```

## Notes

The pilot pipeline uses OCR confirmation first, then conservative text-template
and low-contrast text-band fallbacks. Non-text real-image strips are not allowed
to trigger masking because they caused false positives on screws, connectors,
and other small metal parts. Perceptual hashing is implemented locally with
OpenCV DCT, so no `imagehash` package is required.
