from __future__ import annotations

import argparse
import csv
import math
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mtr3.codec import decode, encode


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark MTR3 against Pillow JPEG/WebP baselines")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--preset", default="balanced", choices=["fast", "balanced", "max"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csv", default="benchmark.csv")
    args = ap.parse_args()

    rows = []
    for src_name in args.images:
        src = Path(src_name)
        original = np.asarray(Image.open(src).convert("RGB"))
        h, w = original.shape[:2]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bit = td / "out.mtr3"
            rec = td / "out.png"
            er = encode(src, bit, preset=args.preset, device=args.device, quiet=True)
            dr = decode(bit, rec, device=args.device)
            reconstructed = np.asarray(Image.open(rec).convert("RGB"))
            rows.append({
                "image": src.name,
                "codec": f"MTR3-{args.preset}",
                "bytes": bit.stat().st_size,
                "bpp": bit.stat().st_size * 8.0 / (w * h),
                "psnr_db": psnr(original, reconstructed),
                "encode_s": er["encode_seconds"],
                "decode_s": dr["decode_seconds"],
            })

            for fmt, ext, quality in [("JPEG", ".jpg", 90), ("WEBP", ".webp", 90)]:
                out = td / f"baseline{ext}"
                t0 = time.perf_counter()
                Image.fromarray(original).save(out, format=fmt, quality=quality)
                enc_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                dec = np.asarray(Image.open(out).convert("RGB"))
                dec_s = time.perf_counter() - t0
                rows.append({
                    "image": src.name,
                    "codec": f"{fmt}-q{quality}",
                    "bytes": out.stat().st_size,
                    "bpp": out.stat().st_size * 8.0 / (w * h),
                    "psnr_db": psnr(original, dec),
                    "encode_s": enc_s,
                    "decode_s": dec_s,
                })

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
