# MTRSA V3 — experimental codec

This directory contains the first executable V3 prototype. It keeps the original repository's idea — **an image is represented by parameters learned specifically for that image** — but replaces the 2022 coordinate MLP with a multi-resolution latent neural field and a compact quantized bitstream.

## What changed

- Fourier coordinate features make the tiny decoder learn high frequencies faster.
- Multi-resolution latent grids store local image information with far fewer parameters than a dense pixel table.
- Seam/edge energy is no longer used to throw pixels away. It becomes an importance distribution so edges and thin structures receive more optimization budget.
- Coarse-to-fine optimization improves encode convergence.
- CUDA AMP is enabled automatically when available.
- Every floating tensor is symmetric INT8 quantized and the byte stream is compressed with zlib.
- Decoding is fully vectorized and requires **no retraining**.
- The `.mtr3` container is self-contained: model configuration, dimensions, quantization scales, and image-specific parameters are stored in the file.

## Install

```bash
cd v3
python -m pip install -e .
```

## Encode / decode

```bash
mtr3 encode input.png output.mtr3 --preset balanced --device auto
mtr3 decode output.mtr3 restored.png --device auto
```

Presets: `fast`, `balanced`, `max`.

## Benchmark

```bash
python benchmark.py ../NeuralNetworksV2/img/1.png --preset balanced --csv benchmark.csv
```

The benchmark writes size, bpp, PSNR, encode time, and decode time for MTR3 plus Pillow JPEG/WebP reference points. Do not treat results from one image as a codec ranking; use Kodak, Tecnick and CLIC and compare full rate-distortion curves.

## Important status note

This is an **alpha research codec**, not a claim that the current prototype beats JPEG XL, VVC, JPEG AI, C3 or Cool-Chic. V3 establishes a clean implementation and bitstream on which the next experiments can be measured. The next performance step should replace zlib with an rANS entropy model and add learned entropy parameters / quantization-aware optimization.
