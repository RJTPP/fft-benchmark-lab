from numpy.fft import fft as numpy_fft
from scipy.fft import fft as scipy_fft
from fft_core import (
    naiveDFT,
    fft_recursive,
    fft_iterative,
    fft_iterative_numba,
    fft_iterative_numba_64,
)

fft_functions = {
    "numpy_fft (Reference)": numpy_fft,
    "scipy_fft (Reference)": scipy_fft,
    # "naiveDFT": naiveDFT,
    # "fft_recursive": fft_recursive,
    # "fft_iterative": fft_iterative,
    "fft_iterative (numba)": fft_iterative_numba,
    # "fft_iterative (numba, 64bits)": fft_iterative_numba_64,
}