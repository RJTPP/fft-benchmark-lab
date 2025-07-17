from .selection import register_fft
import numpy as np
from numba import njit

@njit(cache=True)
def bit_reverse_indices(n: int) -> np.ndarray:
    """
    Compute bit-reversed indices for an array of size n.
    """
    bits = int(np.log2(n))
    result = np.empty(n, dtype=np.int32)
    for i in range(n):
        rev = 0
        for j in range(bits):
            rev |= ((i >> j) & 1) << (bits - 1 - j)
        result[i] = rev
    return result

@register_fft(name="iterative_numba")
@njit(fastmath=True, cache=True)
def fft_iterative_numba(x: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform (FFT) using the iterative Radix-2 Cooley-Tukey algorithm with bit-reversal permutation.
    The bit-reversed order mimics the order of the base-case subproblems in recursion.
    Compiled with Numba for performance optimization.
    """
    N = x.shape[0]
    # if N & (N - 1) != 0:
    #     raise ValueError("Input size must be a power of 2")

    indices = bit_reverse_indices(N)
    x_copy = np.empty(N, dtype=np.complex128)
    for i in range(N):
        x_copy[i] = x[indices[i]]

    size = 2
    while size <= N:
        half = size // 2
        angle = -2j * np.pi / size
        for start in range(0, N, size):
            for k in range(half):
                w = np.exp(angle * k)
                i = start + k
                j = i + half
                t = w * x_copy[j]
                x_copy[j] = x_copy[i] - t
                x_copy[i] = x_copy[i] + t
        size *= 2

    return x_copy


@njit(fastmath=True, cache=True)
def fft_iterative_numba_64(x: np.ndarray) -> np.ndarray:
    """
    iterative Fast Fourier Transform (FFT) on an input array using Numba for optimization.
    Using 64-bit complex numbers for faster calculations.
    """
    N = x.shape[0]
    # if N & (N - 1) != 0:
    #     raise ValueError("Input size must be a power of 2")

    indices = bit_reverse_indices(N)
    x_copy = np.empty(N, dtype=np.complex64)
    for i in range(N):
        x_copy[i] = x[indices[i]]

    size = 2
    while size <= N:
        half = size // 2
        angle = -2j * np.pi / size
        for start in range(0, N, size):
            for k in range(half):
                w = np.exp(angle * k)
                i = start + k
                j = i + half
                t = w * x_copy[j]
                x_copy[j] = x_copy[i] - t
                x_copy[i] = x_copy[i] + t
        size *= 2

    return x_copy


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(f"Expected: {np.fft.fft(x)}")
    print(f"Got     : {fft_iterative_numba(x)}")