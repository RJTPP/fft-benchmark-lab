import numpy as np

from .selection import register_fft


def radix2_bit_reversal(x: np.ndarray) -> np.ndarray:
    """
    Performs the standard radix-2 bit-reversal permutation.
    This is required for any power-of-2 iterative FFT.
    """
    N = len(x)
    log2_N = int(np.log2(N))
    
    reversed_indices = np.zeros(N, dtype=int)
    for i in range(N):
        binary_i = i
        reversed_i = 0
        for _ in range(log2_N):
            reversed_i <<= 1
            reversed_i |= (binary_i & 1)
            binary_i >>= 1
        reversed_indices[i] = reversed_i
        
    return x[reversed_indices]

def radix4_bit_reversal(x: np.ndarray) -> np.ndarray:
    """
    Performs the radix-4 bit-reversal permutation on the input array.
    This reorders the input for the iterative FFT algorithm.
    """
    N = len(x)
    # Calculate the number of stages, which is log4(N)
    num_stages = int(np.log(N) / np.log(4))
    
    # Generate the reversed indices
    reversed_indices = np.zeros(N, dtype=int)
    for i in range(N):
        binary_i = i
        reversed_i = 0
        for _ in range(num_stages):
            # Shift reversed index left by 2 bits (multiply by 4)
            reversed_i <<= 2
            # Take the last two bits of the original index and add them
            reversed_i |= (binary_i & 3)
            # Shift original index right by 2 bits (divide by 4)
            binary_i >>= 2
        reversed_indices[i] = reversed_i
        
    # Return the array with elements in the bit-reversed order
    return x[reversed_indices]


# @register_fft(name="radix4_iterative")
def fft_radix4_iterative(x: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform (FFT) using the iterative Radix-4 Cooley-Tukey algorithm.
    Length N must be a power of 4.
    """
    N = len(x)
    if not (N > 0 and (N & (N - 1) == 0) and int(np.log(N) / np.log(4)) == (np.log(N) / np.log(4))):
        raise ValueError("Input length must be a power of 4")

    # 1. Perform Radix-4 Bit-Reversal
    X = radix4_bit_reversal(x).astype(np.complex128)

    num_stages = int(np.log(N) / np.log(4))

    # 2. Perform iterative butterfly computations for each stage
    for stage in range(1, num_stages + 1):
        M = 4**stage      # Size of the FFT at the current stage
        M4 = M // 4       # Size of the sub-problem
        
        # Pre-compute twiddle factors for the current stage
        k_vals = np.arange(M4)
        angle = -2j * np.pi * k_vals / M
        W_m = np.exp(angle)
        W_2m = W_m**2
        W_3m = W_m**3
        
        # Loop through the butterfly groups
        for j in range(0, N, M):
            # Select indices for the 4 sub-groups within the butterfly
            indices = j + k_vals
            
            # Get data for the butterfly computation
            x0 = X[indices]
            x1 = X[indices + M4]
            x2 = X[indices + 2*M4]
            x3 = X[indices + 3*M4]
            
            # Apply the Radix-4 butterfly operation
            T0 = x0 + W_m * x1 + W_2m * x2 + W_3m * x3
            T1 = x0 - 1j * W_m * x1 - W_2m * x2 + 1j * W_3m * x3
            T2 = x0 - W_m * x1 + W_2m * x2 - W_3m * x3
            T3 = x0 + 1j * W_m * x1 - W_2m * x2 - 1j * W_3m * x3
            
            # Place the results back into the array
            X[indices]        = T0
            X[indices + M4]   = T1
            X[indices + 2*M4] = T2
            X[indices + 3*M4] = T3
            
    return X


@register_fft(name="mixed_radix_iterative")
def fft_mixed_radix_iterative(x: np.ndarray) -> np.ndarray:
    """
    FFT using a mixed-radix (4 and 2) iterative Cooley-Tukey algorithm.
    Length N must be a power of 2. It prioritizes Radix-4 for efficiency.
    """
    N = len(x)
    if not (N > 0 and (N & (N - 1) == 0)):
        raise ValueError("Input length must be a power of 2")

    # 1. Perform standard Radix-2 Bit-Reversal
    X = radix2_bit_reversal(x).astype(np.complex128)

    # 2. Perform Radix-4 butterfly stages
    M = 4 # Start with a 4-point transform
    while M <= N:
        M4 = M // 4
        k_vals = np.arange(M4)
        angle = -2j * np.pi * k_vals / M
        W_m, W_2m, W_3m = np.exp(angle), np.exp(2*angle), np.exp(3*angle)
        
        for j in range(0, N, M):
            indices = j + k_vals
            x0, x1, x2, x3 = X[indices], X[indices + M4], X[indices + 2*M4], X[indices + 3*M4]
            
            X[indices]        = x0 + W_m * x1 + W_2m * x2 + W_3m * x3
            X[indices + M4]   = x0 - 1j*W_m*x1 - W_2m*x2 + 1j*W_3m*x3
            X[indices + 2*M4] = x0 -   W_m*x1 + W_2m*x2 -   W_3m*x3
            X[indices + 3*M4] = x0 + 1j*W_m*x1 - W_2m*x2 - 1j*W_3m*x3
        
        # If we just completed the full transform, we are done
        if M == N:
            return X
        
        # Move to the next size for the radix-4 stage
        M *= 4

    # 3. If N was not a power of 4, a final Radix-2 stage is needed.
    # At this point, M has been multiplied by 4 one too many times.
    # The size of the completed transform is M/4. We need to do a
    # radix-2 pass to get to M/2 = N.
    M = N
    M2 = M // 2
    k_vals = np.arange(M2)
    angle = -2j * np.pi * k_vals / M
    W_m = np.exp(angle)

    for j in range(0, N, M):
        indices = j + k_vals
        x0, x1 = X[indices], X[indices + M2]
        
        X[indices]      = x0 + W_m * x1
        X[indices + M2] = x0 - W_m * x1

    return X


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(f"Expected: {np.fft.fft(x)}")
    print(fft_radix4_iterative(x))