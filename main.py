from numpy.fft import fft as numpy_fft
from scipy.fft import fft as scipy_fft
from utils import test, test_case
from fft_core.fft_prototype import naiveDFT, fft_prototype

functions = {
    "numpy_fft (Reference)": numpy_fft,
    "scipy_fft (Reference)": scipy_fft,
    "naiveDFT": naiveDFT,
    "fft_prototype": fft_prototype
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
    test_fft_correctness(test_case.simple_test_cases, verbose=False)
    print()
    print("Testing correctness...")
    print()
    test_fft_correctness(test_case.simple_test_cases, verbose=True)
    print()
    print("Testing speed...")
    print()
    test_fft_speed(test_case.mid_size_test_cases, verbose=True)