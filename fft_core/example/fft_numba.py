"""Numba-optimized FFT implementations."""


import numpy as np
from numba import njit

from fft_core.selection import register_fft


@njit(cache=True)
def bit_reverse_indices(n: int) -> np.ndarray:
    """
    Compute bit-reversed indices for an array of size n.

    The bit-reversed indices are used to reorder the array in a way that
    mimics the order of the base-case subproblems in recursion. This
    helps in optimizing the iterative FFT algorithm.
    """
    # Calculate the number of bits required to represent the size
    bits = int(np.log2(n))

    # Initialize the result array
    result = np.empty(n, dtype=np.int32)

    # Iterate over the array and compute the bit-reversed indices
    for i in range(n):
        rev = 0

        # Iterate over the bits of the index
        for j in range(bits):
            # Right shift the index by j bits and take the least significant bit
            # Left shift the least significant bit by (bits - 1 - j) bits and add to rev
            rev |= ((i >> j) & 1) << (bits - 1 - j)

        # Store the bit-reversed index
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
    
    # (commented out for speed)
    # Check if the input size is a power of 2
    if N & (N - 1) != 0:
        raise ValueError("Input size must be a power of 2")

    # Compute the bit-reversed indices
    indices = bit_reverse_indices(N)

    # Create a copy of the input array with the bit-reversed order -> mimics the order of recursion
    x_copy = np.empty(N, dtype=np.complex128)
    for i in range(N):
        x_copy[i] = x[indices[i]]

    # Initialize the size of the blocks -> base case of recursion
    size = 2

    # Iterate over the array in blocks of size 'size'
    while size <= N:
        # Compute the half-size of the block
        half = size // 2

        # Compute the twiddle factor
        angle = -2j * np.pi / size

        # Iterate over the blocks
        for start in range(0, N, size):
            # Iterate over the elements of the block
            for k in range(half):
                # Compute the twiddle factor
                w = np.exp(angle * k)

                # Compute the indices of the elements of the block
                i = start + k
                j = i + half

                # Apply the Radix-2 butterfly operation
                t = w * x_copy[j]
                x_copy[j] = x_copy[i] - t
                x_copy[i] = x_copy[i] + t

        # Double the size of the blocks -> next level of recursion
        size *= 2

    return x_copy


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(f"Expected: {np.fft.fft(x)}")
    print(f"Got     : {fft_iterative_numba(x)}")