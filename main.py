import logging

# 1. Configure the root logger:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)

# 2. (Optional) If you want all modules to inherit this level:
logging.getLogger().setLevel(logging.INFO)


from numpy.fft import fft as numpy_fft
from scipy.fft import fft as scipy_fft
from utils import test, test_case
from fft_core import fft_functions

fft_functions = {
    "numpy": numpy_fft,
    "scipy": scipy_fft,
    **fft_functions
}

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
    # print(fft_functions)
    print()
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