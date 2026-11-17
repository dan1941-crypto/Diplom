# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = [
    "onnxruntime",
    "dpkt",
    "pyqtgraph",
]
hiddenimports += collect_submodules("scapy.arch.windows")
hiddenimports += collect_submodules("scapy.layers")

a = Analysis(
    ["src\\main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("configs\\default.yaml", "configs"),
        ("models\\nids_resnet.onnx", "models"),
        ("models\\preprocessor.joblib", "models"),
        ("models\\metadata.json", "models"),
        ("legacy_reference\\test.pcapng", "legacy_reference"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "onnxscript",
        "sklearn",
        "scipy",
        "pandas",
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtPdf",
        "tkinter",
        "matplotlib",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="NIDS_Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
