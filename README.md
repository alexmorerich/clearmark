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

Run a small QA pilot with original / mask / cleaned preview HTML:

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --preset review \
  --rights-confirmed
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
  failed/           optional review artifacts for needs_manual
  manifest.jsonl    one entry per source file
  summary.json      counts, settings, buckets, duplicate stats
  review.html       pilot contact sheet when applicable
```

## Notes

The pipeline uses OpenCV template matching first. It intentionally avoids MSER by default because MSER caused false positives on screws, connectors, and other small metal parts. Perceptual hashing is implemented locally with OpenCV DCT, so no `imagehash` package is required.
