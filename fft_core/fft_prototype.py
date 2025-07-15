import numpy as np


def naiveDFT(x):
    """The naive implementation for comparison (O(N^2))"""
    N = x.size
    X = np.ones(N)*(0+0j)

    for k in range(N):
        A = np.ones(N)*(0+0j)
        for n in range(N):
            A[n] = x[n]*np.exp(-(complex(0, 2*np.pi*k*n/N)))
        X[k] = sum(A)

    return X

def fft_prototype(x: np.ndarray):
    if len(x) == 1:
        return x
    
    N = len(x)
    
    even = x[::2]
    odd = x[1::2]
    even = fft_prototype(even)
    odd = fft_prototype(odd)
    
    k = np.arange(N // 2)
    w = np.exp(-2j*np.pi*k/N)
    
    first_half = even + w*odd
    second_half = even - w*odd
    
    return np.concatenate((first_half, second_half))


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(f"Expected: {np.fft.fft(x)}")
    print(naiveDFT(x))
    print(fft_prototype(x))