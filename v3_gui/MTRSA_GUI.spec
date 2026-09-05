# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_submodules

# PyInstaller resolves script paths relative to the .spec directory, while CI
# invokes this spec from the repository root. Build absolute paths explicitly
# so the package works from either location.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
SCRIPT = os.path.join(SPECPATH, 'app.py')
V3_ROOT = os.path.join(ROOT, 'v3')

hiddenimports = collect_submodules('mtr3')

a = Analysis(
    [SCRIPT],
    pathex=[ROOT, V3_ROOT],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MTRSA-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MTRSA-GUI',
)
