from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mtr3.codec import decode, encode
from mtr3.seam import seam_importance


def human_size(n: int) -> str:
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if v < 1024 or unit == "GiB":
            return f"{v:.2f} {unit}"
        v /= 1024
    return f"{v:.2f} GiB"


def make_pipeline(images: list[tuple[str, Image.Image]], footer: str, out: Path) -> None:
    thumb_w, thumb_h = 340, 250
    header_h, footer_h = 42, 56
    gap = 12
    canvas = Image.new("RGB", (gap + len(images) * (thumb_w + gap), header_h + thumb_h + footer_h), (20, 23, 29))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (label, img) in enumerate(images):
        x = gap + i * (thumb_w + gap)
        draw.text((x, 14), label, fill=(235, 238, 244), font=font)
        fit = img.copy()
        fit.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        px = x + (thumb_w - fit.width) // 2
        py = header_h + (thumb_h - fit.height) // 2
        canvas.paste(fit, (px, py))
    draw.text((gap, header_h + thumb_h + 18), footer, fill=(190, 198, 210), font=font)
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", default="docs/benchmark_01")
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    source = Path(args.source)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    original = Image.open(source).convert("RGB")
    original_path = out / "01_original.png"
    original.save(original_path)

    rgb = np.asarray(original)
    imp = seam_importance(rgb)
    importance_rgb = np.repeat((imp[..., None] * 255.0).clip(0, 255).astype(np.uint8), 3, axis=2)
    importance_img = Image.fromarray(importance_rgb, mode="RGB")
    importance_path = out / "02_importance.png"
    importance_img.save(importance_path)

    bitstream_path = out / "03_compressed.mtr3"
    enc = encode(source, bitstream_path, preset=args.preset, steps=args.steps, device=args.device, quiet=True)

    # A byte-grid makes the otherwise opaque .mtr3 stage visible in README.
    bit_raw = np.frombuffer(bitstream_path.read_bytes(), dtype=np.uint8)
    side = int(np.ceil(np.sqrt(max(1, bit_raw.size))))
    byte_grid = np.zeros(side * side, dtype=np.uint8)
    byte_grid[: bit_raw.size] = bit_raw
    byte_grid = byte_grid.reshape(side, side)
    bitstream_img = Image.fromarray(byte_grid, mode="L").resize((512, 512), Image.Resampling.NEAREST).convert("RGB")
    bitstream_visual_path = out / "03_bitstream_visual.png"
    bitstream_img.save(bitstream_visual_path)

    restored_path = out / "04_restored.png"
    dec = decode(bitstream_path, restored_path, device=args.device)
    restored = Image.open(restored_path).convert("RGB")

    a = np.asarray(original).astype(np.int16)
    b = np.asarray(restored).astype(np.int16)
    diff = np.abs(a - b).clip(0, 255).astype(np.uint8)
    diff_x4 = np.clip(diff.astype(np.int16) * 4, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff_x4, mode="RGB")
    diff_path = out / "05_difference_x4.png"
    diff_img.save(diff_path)

    h, w = a.shape[:2]
    crop_w, crop_h = min(320, w), min(240, h)
    x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
    crop_box = (x0, y0, x0 + crop_w, y0 + crop_h)
    crop_original = original.crop(crop_box)
    crop_restored = restored.crop(crop_box)
    crop_diff = diff_img.crop(crop_box)
    make_pipeline(
        [("Original crop", crop_original), ("Restored crop", crop_restored), ("Difference x4", crop_diff)],
        f"Center crop {crop_w}x{crop_h}",
        out / "06_crop_comparison.png",
    )

    source_bytes = source.stat().st_size
    bitstream_bytes = bitstream_path.stat().st_size
    raw_bytes = w * h * 3
    ratio_file = source_bytes / bitstream_bytes if bitstream_bytes else 0.0
    ratio_raw = raw_bytes / bitstream_bytes if bitstream_bytes else 0.0
    footer = (
        f"{args.preset}, {int(enc['steps'])} steps | {enc['bpp']:.4f} bpp | "
        f"PSNR {enc['psnr_db']:.2f} dB | {human_size(source_bytes)} -> {human_size(bitstream_bytes)}"
    )
    make_pipeline(
        [("1. Original", original), ("2. Importance", importance_img), ("3. .mtr3 bytes", bitstream_img), ("4. Restored", restored), ("5. Error x4", diff_img)],
        footer,
        out / "00_pipeline.png",
    )

    results = {
        "source": str(source).replace("\\", "/"),
        "width": w,
        "height": h,
        "preset": args.preset,
        "steps": int(enc["steps"]),
        "source_bytes": source_bytes,
        "raw_rgb_bytes": raw_bytes,
        "bitstream_bytes": bitstream_bytes,
        "ratio_vs_source_file": ratio_file,
        "ratio_vs_raw_rgb": ratio_raw,
        "bpp": float(enc["bpp"]),
        "psnr_db": float(enc["psnr_db"]),
        "encode_seconds": float(enc["encode_seconds"]),
        "decode_seconds": float(dec["decode_seconds"]),
        "device": str(enc["device"]),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
