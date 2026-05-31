# ClearMark

ClearMark is a local Sunsky watermark detection, removal, and visual-review
pipeline for authorized product-image copies. It is built for the
`sunsky-online.com` watermark pattern in the B2B product asset library.

The project is intentionally conservative:

```text
detect accurately -> repair narrowly -> publish only if post-clean evidence is clean
```

The source asset folder is never modified. Every cleaned image, failed attempt,
mask, HTML report, PDF, manifest, and Telegram attachment is generated output
and must stay out of Git.

## Paths

Source assets:

```bash
/Users/alexkou/Documents/github/b2bweb/content/products/assets
```

Project root:

```bash
/Users/alexkou/Documents/openai/clearmark
```

Typical review output:

```bash
/Users/alexkou/Downloads/clearmark-sunsky-50
```

## Safety Rules

- Use ClearMark only for images you own or are authorized to modify.
- All cleaning commands require `--rights-confirmed`.
- Source images are read-only inputs.
- Output paths inside `/Users/alexkou/Documents/github/b2bweb` are rejected.
- Generated output is never committed: `cleaned/`, `attempts/`, `masks/`,
  `originals/`, `overlays/`, `diffs/`, `manifest.jsonl`, `summary.json`,
  `review.html`, and `compare.pdf`.
- iPhone 14+ only filenames are skipped. The supplier stopped watermarking
  those series images, so they are treated as known clean.
- A failed repair is not a cleaned image. Failed best-effort results are written
  to `attempts/` for review only.

## Design Concept

Early versions tried to remove the watermark whenever a detector found a
text-like band. That caused two bad outcomes:

- clean images or product text were selected as watermarked;
- broad masks sometimes damaged product details while the watermark still
  remained readable.

ClearMark now follows a compact visual-truthfulness design inspired by the
V10-V13 Mark Remover architecture, but without copying the large 100-tool
strategy bank. The important ideas are:

- confirm that the text is actually `sunsky-online.com`;
- route repair by the pixels under the watermark, not by detector provenance;
- use narrow glyph/component masks before any broader box mask;
- imitate nearby background directly for pure white or low-texture background
  pixels instead of asking an inpainting algorithm to hallucinate them;
- run a final publish gate on the cleaned result, not on the method name;
- keep failed attempts visible in review output but out of `cleaned/`.

## Processing Pipeline

```text
Asset inventory
  -> iPhone 14+ skip
  -> perceptual-hash master selection
  -> watermark detection
  -> presence confirmation
  -> ROI classification
  -> mask construction
  -> repair candidates
  -> residual micro-cleanup
  -> final publish gate
  -> review HTML/PDF/Telegram
```

### 1. Inventory And Sampling

The `inventory` step scans image metadata and computes OpenCV DCT perceptual
hashes. Duplicate or near-duplicate images are grouped so pilots can sample
master images instead of repeatedly testing identical assets.

iPhone 14+ exclusion is handled by `should_scan_file()` in
`scripts/watermark_pipeline.py`. The constant is:

```python
SKIP_IPHONE_MIN = 14
```

Change this only if the supplier policy changes.

### 2. Watermark Detection

Detection combines several signals:

- EasyOCR full-image text detections when OCR is enabled;
- canonical `sunsky-online.com` template matching;
- low-contrast text-band recall for faint marks;
- bright-background recall for watermark text on white product photos.

The OCR matcher is deliberately specific. A match must look like the domain
structure:

```text
sunsky/sky-like token + online/.com-like token
```

This blocks false positives such as product labels, "More product details",
step instructions, connector text, and random OCR fragments. Examples that
should match:

```text
sunsky-online.com
S unsky-online.co
sky-online.com
~sunsky-oniine com
```

Examples that should not match:

```text
aline cer
Stable Bracket
More product details
onlin com
```

### 3. Text-Dense Layout Guard

Instruction sheets and multi-panel images often contain many real words and
layout lines. Weak text-band priors are unsafe there.

`image_layout_features()` computes:

- edge density;
- small connected-component count;
- strong horizontal/vertical panel-line score;
- text-dense and step-layout booleans.

When a layout is text-dense, weak prior detections are disabled. The image must
have direct OCR or strong template evidence before it is included in a
watermarked-only run.

### 4. ROI Classification

The most important routing decision is what sits underneath the watermark.
`estimate_product_overlap_v13()` classifies the mark box from interior pixels:

- `plain_white`
- `near_white`
- `low_texture_background`
- `simple_product_surface`
- `dark_product_surface`
- `thin_flex_cable`
- `complex_product_detail`
- `text_or_label_area`
- `unknown`

High-risk classes use glyph-only repair attempts. Broad white or rectangular
fills are blocked on flex cables, dark product surfaces, labels, and complex
product detail.

### 5. Mask Strategy

ClearMark prefers the smallest mask that can remove readable watermark text.

Mask variants include:

- glyph masks from local contrast around the detected text;
- canonical watermark ink masks for OCR detections;
- stronger glyph masks with slightly more dilation;
- tight/medium box masks only on safe, low-risk backgrounds;
- high-contrast fallback boxes only when the ROI is not product-risky.

The mask area is guarded by `MAX_MASK_AREA`, `PILOT_MASK_AREA`, and combined
mask limits. Oversized masks route to `needs_manual`.

### 6. Repair Strategy

The current repair engine is intentionally small:

- near-area background fill for plain/near-white background pixels;
- OpenCV Telea inpaint;
- OpenCV Navier-Stokes inpaint;
- optional LaMa/IOPaint escalation when installed and enabled;
- component-level residual cleanup before final rejection.

For pure background, near-area fill is tried before classical inpaint. It
samples the context ring around the watermark, estimates the local background
color and tiny texture/noise, and blends that into only the background portion
of the mask. Product-overlap pixels are handled separately with tiny masks, so
white background does not get pasted onto black cables or product surfaces.

This project does not yet implement the full 100-tool strategy bank. The next
big quality step would be better pixel reconstruction for
`thin_flex_cable`, `complex_product_detail`, and `text_or_label_area` cases.

### 7. Residual Micro-Cleanup

Some first-pass repairs remove most of the watermark but leave readable dot
chains, broken glyphs, or a trailing `.com`. ClearMark detects this inside a
horizontally expanded mark footprint.

If the only blockers are residual text signals, ClearMark attempts a second
small cleanup:

- ring-median fill over leftover components;
- small-radius Telea inpaint over only the residual component mask;
- tighter area limits for product-overlap regions.

This is not a full rectangular re-clean. It is designed to remove leftover
glyph fragments without damaging the product.

### 8. Final Publish Gate

The final gate decides whether an output is allowed into `cleaned/`.

It checks:

- required QA metrics are present;
- residual score is below threshold;
- template residual is below threshold;
- post-clean text components are minimal;
- post-clean detector count is zero;
- OCR on the cleaned mark-box crop does not still read Sunsky;
- dot-chain/broken-glyph detector passes;
- rectangular-band detector passes;
- product-damage detector passes.
- post-clean Sunsky double-check passes. If OCR sees any domain-like
  `sunsky-online.com` residue above the suspect threshold, the output fails
  even when OCR confidence is low.

If any gate fails, status is `needs_manual`. The output may still be written to
`attempts/` for review, but it is not publishable and is not placed in
`cleaned/`.

## Output Statuses

| Status | Meaning |
| --- | --- |
| `cleaned` | Publish gate passed. File is written to `cleaned/`. |
| `needs_manual` | Detection exists, but cleaning did not pass final QA. Best attempt may be written to `attempts/`. |
| `no_watermark` | No confirmed Sunsky watermark. Source is copied only for review if needed. |
| `skipped` | Known-clean iPhone 14+ image. |
| `cleaned_duplicate` | Duplicate reused a cleaned master result. |
| `duplicate_no_action` | Duplicate was skipped because the master was not cleaned. |

## Output Layout

Each run creates:

```text
outputs/<run-id>/ or custom --out/
  cleaned/          only publish-gate-passed cleaned image copies
  attempts/         failed best attempts for review only
  masks/            black/white masks
  originals/        copied originals for review only
  overlays/         originals with red mask overlay
  diffs/            absolute visual difference, amplified 3x
  manifest.jsonl    one JSON entry per selected source file
  summary.json      counts, settings, inventory stats, Telegram result
  review.html       side-by-side browser review
  compare.pdf       optional PDF compare for review and Telegram
```

Review columns:

```text
Original | Mask overlay | Result | Diff x3
```

For `needs_manual`, the result column is labeled `Attempt (failed QA)` and
points to `attempts/` if a failed attempt exists. This is for inspection only.

## Commands

### Install Dependencies

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 -m pip install -r requirements.txt
```

Optional OCR uses EasyOCR. Optional neural inpainting uses
`simple_lama_inpainting` or a local `iopaint` command.

### Build Inventory

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py inventory \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets
```

### Run A 50-Image Review Pilot

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

Use `--no-lama` for faster review pilots. Remove it when neural inpainting is
installed and quality matters more than runtime.

### Send PDF To Telegram

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."

python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 50 \
  --max-scan 700 \
  --watermarked-only \
  --preset review \
  --no-lama \
  --pdf \
  --telegram \
  --out /Users/alexkou/Downloads/clearmark-sunsky-50 \
  --rights-confirmed
```

In this environment the Telegram token can also be loaded from:

```bash
/Users/alexkou/.claude/channels/telegram/.env
```

### Run One-Pass Production Processing

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 scripts/watermark_pipeline.py process \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --preset review \
  --workers 8 \
  --ocr \
  --rights-confirmed
```

Workflow:

```text
for each non-iPhone-14+ master image:
    detect watermark using review preset
    confirm it is really sunsky-online.com
    if no confirmed detection -> record no_watermark and skip
    if confirmed detection    -> repair and validate
    if publish gate passes    -> write cleaned/
    otherwise                 -> write attempts/ for review, status needs_manual

duplicates:
    reuse master result only when the master status is cleaned
```

For speed-only processing, omit `--ocr`. The pipeline will then be more
conservative and skip weak prior detections unless canonical template evidence
is strong.

## Reading The Manifest

Each `manifest.jsonl` row contains fields such as:

- `file`
- `status`
- `presence_reason`
- `presence_score`
- `strategy`
- `mask_area_pct`
- `residual_score`
- `template_residual_score`
- `post_text_components`
- `roi_class`
- `product_overlap`
- `cleanup_attempted`
- `cleanup_strategy`
- `sunsky_check_pass`
- `post_clean_ocr_score`
- `reason`

Useful review patterns:

- `status=cleaned`: inspect a sample visually, but it passed the publish gate.
- `reason=residual_visible`: some watermark-like signal remains.
- `reason=dot_chain_residual`: broken glyph fragments remain.
- `reason=product_damage`: repair changed product structure too much.
- `warning=risky_roi:*`: mask was routed conservatively because the mark
  overlaps product detail.

## Project Structure

```text
clearmark/
  README.md
  requirements.txt
  scripts/
    watermark_pipeline.py   detection, repair, QA, review HTML/PDF, Telegram
  templates/
    watermark-template.png  canonical Sunsky text template
  outputs/                 ignored generated output
```

## Development Checklist

Before committing code:

```bash
cd /Users/alexkou/Documents/openai/clearmark
python3 -m py_compile scripts/watermark_pipeline.py
git diff --check
```

Before trusting a quality change:

```bash
python3 scripts/watermark_pipeline.py pilot \
  --assets /Users/alexkou/Documents/github/b2bweb/content/products/assets \
  --max-total 5 \
  --max-scan 100 \
  --watermarked-only \
  --preset fast \
  --no-lama \
  --pdf \
  --out /Users/alexkou/Downloads/clearmark-smoke \
  --rights-confirmed
```

Then inspect `compare.pdf` and `manifest.jsonl`.

## Troubleshooting

### Clean Images Are Included In A Watermarked Pilot

Use OCR-enabled `--watermarked-only` pilots. The OCR domain-structure gate is
the safest way to avoid product-text false positives.

### Sunsky Text Remains After Cleaning

Check:

- `residual_score`
- `template_residual_score`
- `dot_chain_score`
- `ocr_text`
- `cleanup_attempted`
- `cleanup_strategy`

If status is `needs_manual`, the file in `attempts/` is not publishable. It is
only a visual diagnostic.

### Product Detail Is Damaged

Look at:

- `roi_class`
- `product_overlap`
- `product_gate_pass`
- `product_color_delta`
- `product_edge_retention`
- `product_blob_score`

High-risk product classes should avoid broad masks. If damage still appears,
add a regression fixture and tighten product routing for that ROI class.

### The Run Is Slow

OCR is the main cost. For quick detector experiments:

```bash
--no-ocr --preset fast --max-total 5
```

For official review pilots, keep OCR enabled.

### Output Accidentally Goes Into The Source Repo

The script refuses output paths under:

```bash
/Users/alexkou/Documents/github/b2bweb
```

Use `/Users/alexkou/Downloads/...` or the project `outputs/` directory.

## Roadmap

The current pipeline is detection-safe and publish-gated, but the repair side
still needs better reconstruction for hard cases.

Next improvements:

1. Add regression fixtures for OCR false positives, step-layout pages, dark
   product surfaces, flex cables, and `.com` tail residuals.
2. Improve repair methods for `thin_flex_cable`, `complex_product_detail`, and
   `text_or_label_area`.
3. Add an honest `clean_covered` status only after cover output passes the same
   visual-fidelity gate as `cleaned`.
4. Add a dedicated report command that summarizes must-be-zero counters:
   residual OCR, dot-chain residue, visible band, and product damage.

Do not implement broad tiled-watermark removal unless fresh visual evidence
shows a real repeated watermark layer in the current dataset. Earlier review
found that premise was not supported by inspected images.

## Notes

The pipeline uses OpenCV DCT for perceptual hashes, so no `imagehash` package
is required. Generated outputs should remain local review artifacts and should
not be pushed to GitHub.
