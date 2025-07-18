from .selection import register_fft
import numpy as np


def fft_radix4_recursive(x: np.ndarray):
    """
    Fast Fourier Transform (FFT) using the recursive Radix-4 Cooley-Tukey algorithm.
    Length must be a power of 4
    """
    N = len(x)
    if N == 1:
        return x
    if N % 4 != 0:
        raise ValueError("Input length must be divisible by 4 (i.e., power of 4)")
    
    N = len(x)
    
    x_0 = x[0::4]
    x_1 = x[1::4]
    x_2 = x[2::4]
    x_3 = x[3::4]
    
    x_0 = fft_radix4_recursive(x_0)
    x_1 = fft_radix4_recursive(x_1)
    x_2 = fft_radix4_recursive(x_2)
    x_3 = fft_radix4_recursive(x_3)
    
    k = np.arange(N // 4)
    angle = -2j * np.pi * k / N
    w   = np.exp(angle)
    w_2 = np.exp(2 * angle)
    w_3 = np.exp(3 * angle)
    
    T0 = x_0 +       w * x_1 +     w_2 * x_2 +      w_3 * x_3
    T1 = x_0 - 1j *  w * x_1 -     w_2 * x_2 + 1j * w_3 * x_3
    T2 = x_0 -       w * x_1 +     w_2 * x_2 -      w_3 * x_3
    T3 = x_0 + 1j *  w * x_1 -     w_2 * x_2 - 1j * w_3 * x_3

    # Combine all 4 outputs into full result
    return np.concatenate([T0, T1, T2, T3])
    
    
# @register_fft(name="split_radix_recursive")
def fft_split_radix_recursive(x: np.ndarray):
    """
    Fast Fourier Transform (FFT) using the recursive Radix-2 and Radix-4 Cooley-Tukey algorithm.
    """
    N = len(x)
    if N == 1:
        return x
    if N % 4 == 0:
        x_0 = x[0::4]
        x_1 = x[1::4]
        x_2 = x[2::4]
        x_3 = x[3::4]
        
        x_0 = fft_split_radix_recursive(x_0)
        x_1 = fft_split_radix_recursive(x_1)
        x_2 = fft_split_radix_recursive(x_2)
        x_3 = fft_split_radix_recursive(x_3)
        
        k = np.arange(N // 4)
        angle = -2j * np.pi * k / N
        w   = np.exp(angle)
        w_2 = np.exp(2 * angle)
        w_3 = np.exp(3 * angle)
        
        T0 = x_0 +       w * x_1 +     w_2 * x_2 +      w_3 * x_3
        T1 = x_0 - 1j *  w * x_1 -     w_2 * x_2 + 1j * w_3 * x_3
        T2 = x_0 -       w * x_1 +     w_2 * x_2 -      w_3 * x_3
        T3 = x_0 + 1j *  w * x_1 -     w_2 * x_2 - 1j * w_3 * x_3
        return np.concatenate([T0, T1, T2, T3])
    elif N % 2 == 0:
        even = x[::2]
        odd = x[1::2]
        even = fft_split_radix_recursive(even)
        odd = fft_split_radix_recursive(odd)
        
        k = np.arange(N // 2)
        w = np.exp(-2j*np.pi*k/N)
        
        first_half = even + w*odd
        second_half = even - w*odd
        
        return np.concatenate((first_half, second_half))
    else:
        raise ValueError("Input length must be a power of 2")


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(f"Expected: {np.fft.fft(x)}")
    print(fft_split_radix_recursive(x))