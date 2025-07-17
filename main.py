from numpy.fft import fft as numpy_fft
from scipy.fft import fft as scipy_fft
from utils import test, test_case
from fft_core import (
    naiveDFT,
    fft_recursive,
    fft_iterative,
    fft_iterative_numba,
    fft_iterative_numba_64,
)

functions = {
    "numpy_fft (Reference)": numpy_fft,
    "scipy_fft (Reference)": scipy_fft,
    # "naiveDFT": naiveDFT,
    # "fft_recursive": fft_recursive,
    # "fft_iterative": fft_iterative,
    "fft_iterative (numba)": fft_iterative_numba,
    # "fft_iterative (numba, 64bits)": fft_iterative_numba_64,
}

def test_fft_correctness(testcase, verbose=True):
    for name, func in functions.items():
        test.test_correctness(
            func, 
            testcase, 
            reference_func=scipy_fft, 
            name=name,
            verbose=verbose,
        )

def test_fft_speed(testcase, verbose=True):
    for name, func in functions.items():
        test.test_speed(
            func, 
            testcase, 
            name=name,
            verbose=verbose,
        )


if __name__ == "__main__":
    print("Warming up...")
    test_fft_correctness(test_case.get_simple_test_cases(), verbose=False)
    print()
    print("Testing correctness...")
    print()
    test_fft_correctness(test_case.get_combined_test_cases(), verbose=True)
    print()
    print("Testing speed...")
    print()
    test_fft_speed(test_case.get_large_test_cases_extended(), verbose=True)
    print()
    # print("Testing speed (2)...")
    # test_fft_speed(test_case.large_npy_one_case, verbose=True)