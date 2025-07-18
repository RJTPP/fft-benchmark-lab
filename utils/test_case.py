"""Functions for generating test cases for FFT implementations."""

import numpy as np

_simple_test_cases = [
    np.array([1], dtype=complex),
    np.array([1, 2, 3, 4], dtype=complex),
    np.array([0, 1, 0, -1], dtype=complex),
    np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=complex),
]
def get_simple_test_cases():
    global _simple_test_cases
    return _simple_test_cases


_real_test_cases = None
def get_real_test_cases():
    global _real_test_cases
    if _real_test_cases is None:
        _real_test_cases = [
            np.random.rand(2**x) for x in range(0, 10)
        ]
    return _real_test_cases


_complex_test_cases = None
def get_complex_test_cases():
    global _complex_test_cases
    if _complex_test_cases is None:
        _complex_test_cases = [
            np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(0, 10)
        ]
    return _complex_test_cases


_combined_test_cases = None
def get_combined_test_cases():
    global _combined_test_cases
    if _combined_test_cases is None:
        _combined_test_cases = get_real_test_cases() + get_complex_test_cases()
    return _combined_test_cases


_mid_size_test_cases = None
def get_mid_size_test_cases():
    global _mid_size_test_cases
    if _mid_size_test_cases is None:
        _mid_size_test_cases = [
            np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(7, 12)
        ]
    return _mid_size_test_cases


_large_test_cases = None
def get_large_test_cases():
    global _large_test_cases
    if _large_test_cases is None:
        _large_test_cases = [
            np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(10, 20)
        ]
    return _large_test_cases


_large_test_cases_extended = None
def get_large_test_cases_extended():
    global _large_test_cases_extended
    if _large_test_cases_extended is None:
        _large_test_cases_extended = [
            np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(1, 20)
        ]
    return _large_test_cases_extended


_large_test_cases_extended_base4 = None
def get_large_power_of_four_test_cases():
    global _large_test_cases_extended_base4
    if _large_test_cases_extended_base4 is None:
        _large_test_cases_extended_base4 = [
            np.random.rand(4**x) + 1j * np.random.rand(4**x) for x in range(1, 10)
        ]
    return _large_test_cases_extended_base4


_massive_test_cases = None
def get_massive_test_cases():
    global _massive_test_cases
    if _massive_test_cases is None:
        _massive_test_cases = [
            np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(1, 28)
        ]
    return _massive_test_cases


_large_npy_one_case = None
def get_large_npy_one_case():
    global _large_npy_one_case
    if _large_npy_one_case is None:
        _large_npy_one_case = [
            np.array([1]*(2**x), dtype=complex) for x in range(25, 28)
        ]
    return _large_npy_one_case


def print_test_case(test_case: list[np.ndarray]):
    for i, test in enumerate(test_case):
        print(f"test case {i+1}:\n {test}\n")


if __name__ == "__main__":
    print_test_case(get_simple_test_cases())
    print([x.shape for x in get_large_test_cases()])
