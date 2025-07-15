from utils import test, test_case
from fft_core.fft_prototype import naiveDFT, fft_prototype
from scipy.fft import fft as scipy_fft

functions = {
    "scipy_fft (Reference)": scipy_fft,
    "naiveDFT": naiveDFT,
    "fft_prototype": fft_prototype
}

def test_fft_correctness(verbose=True):
    for name, func in functions.items():
        test.test_correctness(
            func, 
            test_case.simple_test_cases, 
            reference_func=scipy_fft, 
            name=name,
            verbose=verbose,
        )

def test_fft_speed(verbose=True):
    for name, func in functions.items():
        test.test_speed(
            func, 
            test_case.mid_size_test_cases, 
            name=name,
            verbose=verbose,
        )


if __name__ == "__main__":
    print("Testing correctness...")
    print()
    test_fft_correctness(verbose=True)
    print()
    print("Testing speed...")
    print()
    test_fft_speed(verbose=True)