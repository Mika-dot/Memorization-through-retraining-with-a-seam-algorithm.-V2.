# MTRSA GUI

Desktop frontend for the MTRSA V3 research image codec.

## Run from source

```bash
python -m pip install -e v3
python -m pip install -r v3_gui/requirements.txt
python -m v3_gui.app
```

The app supports drag-and-drop, image → `.mtr3` compression, `.mtr3` → PNG restoration, seam/detail importance preview, preset/device selection, progress reporting, cancellation, and codec metrics.

## Windows portable build

The repository's GitHub Actions release workflow builds an `onedir` PyInstaller package and publishes it as `MTRSA-GUI-Windows-x64.zip`. The portable package intentionally uses the CPU PyTorch wheel so it runs on Windows PCs without a CUDA runtime. Running from source with a CUDA-enabled PyTorch build enables GPU encoding automatically when `Device = auto`.
