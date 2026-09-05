from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from mtr3.codec import EncodeCancelled, encode


def test_encode_progress_callback(tmp_path):
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(16, dtype=np.uint8)[None, :] * 12
    src = tmp_path / "src.png"
    bit = tmp_path / "src.mtr3"
    Image.fromarray(rgb).save(src)

    progress: list[tuple[int, int, float]] = []
    encode(
        src,
        bit,
        preset="fast",
        device="cpu",
        steps=2,
        quiet=True,
        progress_callback=lambda done, total, loss: progress.append((done, total, loss)),
    )

    assert progress
    assert progress[-1][0] == progress[-1][1]
    assert bit.exists()


def test_encode_can_be_cancelled(tmp_path):
    src = tmp_path / "src.png"
    bit = tmp_path / "src.mtr3"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(src)

    with pytest.raises(EncodeCancelled):
        encode(
            src,
            bit,
            preset="fast",
            device="cpu",
            steps=3,
            quiet=True,
            cancel_check=lambda: True,
        )
