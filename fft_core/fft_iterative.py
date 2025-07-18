"""Iterative FFT implementations."""

from .selection import register_fft
import numpy as np

# @register_fft(name="iterative")
def fft_iterative(x: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform (FFT) using the iterative Radix-2 Cooley-Tukey algorithm with bit-reversal permutation.
    The bit-reversed order mimics the order of the base-case subproblems in recursion.
    """
    N = x.shape[0]
    # if N & (N - 1) != 0:
    #     raise ValueError("Input size must be a power of 2")

    # Bit-reversal permutation
    bits = int(np.log2(N))
    indices = np.arange(N)
    rev_indices = np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in indices])
    x = x[rev_indices].astype(np.complex128)  # ✅ This avoids the warning

    # Iterative FFT
    size = 2
    while size <= N:
        half = size // 2
        w_m = np.exp(-2j * np.pi * np.arange(half) / size)

        for start in range(0, N, size):
            for k in range(half):
                i = start + k
                j = i + half
                t = w_m[k] * x[j]
                x[j] = x[i] - t
                x[i] = x[i] + t
        size *= 2

    return x


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(f"Expected: {np.fft.fft(x)}")
    print(f"Got     : {fft_iterative(x)}")