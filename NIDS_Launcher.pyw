"""Windowless launcher for NIDS Traffic Analyzer (double-click in Explorer)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHONW = PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"


def _in_project_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == (PROJECT_ROOT / ".venv").resolve()
    except OSError:
        return False


if __name__ == "__main__":
    if not _in_project_venv() and VENV_PYTHONW.exists():
        subprocess.Popen(
            [str(VENV_PYTHONW), "-m", "src.main"],
            cwd=str(PROJECT_ROOT),
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
        )
        sys.exit(0)

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.main import main

    sys.exit(main())
