from numpy.fft import fft as numpy_fft
from scipy.fft import fft as scipy_fft
from utils import test, test_case
from fft_selection import fft_functions

def test_fft_correctness(testcase, verbose=True):
    for name, func in fft_functions.items():
        test.test_correctness(
            func, 
            testcase, 
            reference_func=scipy_fft, 
            name=name,
            verbose=verbose,
        )

def test_fft_speed(testcase, verbose=True):
    for name, func in fft_functions.items():
        test.test_speed(
            func, 
            testcase, 
            name=name,
            verbose=verbose,
        )


if __name__ == "__main__":
    # TODO: Update README
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