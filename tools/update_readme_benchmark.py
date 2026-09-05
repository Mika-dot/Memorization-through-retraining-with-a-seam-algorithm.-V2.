from __future__ import annotations

import argparse
import json
from pathlib import Path

START = "<!-- BENCHMARK_EXAMPLE:START -->"
END = "<!-- BENCHMARK_EXAMPLE:END -->"
GUI_START = "<!-- GUI_APP:START -->"
GUI_END = "<!-- GUI_APP:END -->"


def human_size(n: int) -> str:
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if v < 1024 or unit == "GiB":
            return f"{v:.2f} {unit}"
        v /= 1024
    return f"{v:.2f} GiB"


def replace_or_insert(text: str, start: str, end: str, block: str, anchor: str) -> str:
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        return before + block + after
    if anchor not in text:
        return text.rstrip() + "\n\n" + block + "\n"
    return text.replace(anchor, block + "\n\n" + anchor, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--results", default="docs/benchmark_01/results.json")
    args = ap.parse_args()

    readme_path = Path(args.readme)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    text = readme_path.read_text(encoding="utf-8")

    gui_block = f"""{GUI_START}
## Desktop GUI

MTRSA now ships as a desktop application as well as a Python codec. The GUI is designed for the actual image -> `.mtr3` -> restored image workflow rather than as a demo wrapper.

**Windows x64:** download the current portable build from [GitHub Releases](https://github.com/Mika-dot/Memorization-through-retraining-with-a-seam-algorithm.-V2./releases/latest). Unzip it and run `MTRSA-GUI.exe`; no Python installation is required.

The application provides:

- drag-and-drop image loading;
- `fast`, `balanced`, and `max` encode presets;
- automatic CPU/CUDA selection when running from source;
- live seam/detail importance visualization;
- image -> self-contained `.mtr3` compression;
- `.mtr3` -> PNG restoration without retraining;
- progress reporting and cancellation;
- source size, encoded size, ratio, bpp, PSNR, encode time, and decode time;
- original / importance / reconstructed previews in one window.

The portable Windows release deliberately bundles the CPU PyTorch runtime for maximum compatibility. A source installation with a CUDA-enabled PyTorch build exposes GPU acceleration through `Device = auto`.

Source: [`v3_gui/`](v3_gui/). Reproducible Windows packaging and release publication: [`.github/workflows/gui-release.yml`](.github/workflows/gui-release.yml).
{GUI_END}"""
    text = replace_or_insert(text, GUI_START, GUI_END, gui_block, "## Quick start")

    block = f"""{START}
## Benchmark example: image 1, end to end

This example is generated automatically from the repository's historical **`NeuralNetworksV2/img/1.png`** using the same V3 codec code shipped with the GUI. It is not a cherry-picked claim against another codec; it shows exactly what MTRSA stores and what comes back out.

![MTRSA benchmark 01 pipeline](docs/benchmark_01/00_pipeline.png)

The stages are:

1. **Original** - input pixels from benchmark image 1.
2. **Importance map** - seam/detail energy used to spend more optimization budget on edges and texture without changing image geometry.
3. **`.mtr3` bitstream** - quantized per-image neural representation. The screenshot is a direct byte-grid visualization of the compressed file, not an image approximation.
4. **Restored image** - decoded by evaluating the compact neural field; no retraining is performed during decode.
5. **Difference x4** - absolute RGB error amplified four times so reconstruction failures are visible.

| Metric | Measured value |
|---|---:|
| Image | {results['width']} x {results['height']} |
| Preset / optimization | `{results['preset']}` / {results['steps']} steps |
| Source file | {human_size(results['source_bytes'])} |
| Raw RGB payload | {human_size(results['raw_rgb_bytes'])} |
| `.mtr3` bitstream | **{human_size(results['bitstream_bytes'])}** |
| Ratio vs source file | **{results['ratio_vs_source_file']:.2f}x** |
| Ratio vs raw RGB | **{results['ratio_vs_raw_rgb']:.2f}x** |
| Bits per pixel | **{results['bpp']:.4f} bpp** |
| PSNR after quantized round-trip | **{results['psnr_db']:.2f} dB** |
| Encode time on CI CPU | {results['encode_seconds']:.3f} s |
| Decode time on CI CPU | {results['decode_seconds']:.3f} s |

> **Important:** the source image file is already compressed, so `ratio vs source file` is not the same thing as compression relative to raw pixels. For codec research, **bpp + reconstruction quality** is the meaningful pair.

### Zoomed reconstruction check

![MTRSA benchmark crop comparison](docs/benchmark_01/06_crop_comparison.png)

Individual generated files: [original](docs/benchmark_01/01_original.png) · [importance](docs/benchmark_01/02_importance.png) · [compressed `.mtr3`](docs/benchmark_01/03_compressed.mtr3) · [bitstream visualization](docs/benchmark_01/03_bitstream_visual.png) · [restored](docs/benchmark_01/04_restored.png) · [difference x4](docs/benchmark_01/05_difference_x4.png) · [machine-readable results](docs/benchmark_01/results.json).
{END}"""
    text = replace_or_insert(text, START, END, block, "## Why the original idea is more relevant now than in 2022")
    readme_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
