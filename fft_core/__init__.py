"""Core FFT implementation package."""

import importlib
from pathlib import Path

_pkg = __name__
_dir = Path(__file__).parent
skip_files = ["__init__.py", "__pycache__", "selection.py"]

def import_files(dir: Path, base_pkg: str):
    for path in dir.iterdir():
        if path.is_file() and path.suffix == ".py" and path.name not in skip_files:
            # module name: e.g. fft_core.fft_numba or fft_core.extra.fft_foo
            module_name = f"{base_pkg}.{path.stem}"
            importlib.import_module(module_name)
        elif path.is_dir() and (path / "__init__.py").exists():
            # Only treat subfolders with __init__.py as packages
            import_files(path, f"{base_pkg}.{path.name}")

import_files(_dir, _pkg)

from .selection import fft_functions