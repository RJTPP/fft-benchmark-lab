# fft_core/__init__.py
import os, importlib

# import every module so decorators run
_pkg = __name__
_dir = os.path.dirname(__file__)
for fname in os.listdir(_dir):
    if fname.endswith(".py") and fname != "__init__.py":
        importlib.import_module(f"{_pkg}.{fname[:-3]}")

# expose the registry at package level
from .selection import fft_functions