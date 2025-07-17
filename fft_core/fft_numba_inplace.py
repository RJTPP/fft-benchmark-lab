from .selection import register_fft
import numpy as np
from numba import njit, jit


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

@njit(fastmath=True, cache=True)
def _fft_iterative_numba_inplace_helper(x: np.ndarray) -> np.ndarray:
    """
    Performs an in-place Fast Fourier Transform (FFT) using the iterative 
    Radix-2 Cooley-Tukey algorithm.
    
    NOTE: This function modifies the input array 'x'.
    """
    N = x.shape[0]
    # if N & (N - 1) != 0:
    #     raise ValueError("Input size must be a power of 2")

    # 1. Perform in-place bit-reversal permutation
    indices = bit_reverse_indices(N)
    for i in range(N):
        j = indices[i]
        if i < j:
            x[i], x[j] = x[j], x[i]

    # 2. Iterative butterfly computations, now on 'x'
    size = 2
    while size <= N:
        half = size // 2
        angle = -2j * np.pi / size
        for start in range(0, N, size):
            for k in range(half):
                w = np.exp(angle * k)
                i = start + k
                j = i + half
                t = w * x[j]
                x[j] = x[i] - t
                x[i] = x[i] + t
        size *= 2

    return x


@register_fft(name="iterative_numba_inplace")
def fft_iterative_numba_inplace_wrapper(x: np.ndarray) -> np.ndarray:
    # Ensure input is complex for in-place operations
    if x.dtype != np.complex128:
        x = x.astype(np.complex128)
    return _fft_iterative_numba_inplace_helper(x)


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(f"Expected: {np.fft.fft(x)}")
    print(f"Got     : {fft_iterative_numba_inplace_wrapper(x)}")