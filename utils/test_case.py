import numpy as np

simple_test_cases = [
    np.array([1], dtype=complex),
    np.array([1, 2, 3, 4], dtype=complex),
    np.array([0, 1, 0, -1], dtype=complex),
    np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=complex),
]

real_test_cases = [
    np.random.rand(2**x) for x in range(0, 10)
]

complex_test_cases = [
    np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(0, 10)
]

combined_test_cases = real_test_cases + complex_test_cases

mid_size_test_cases = [
    np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(7, 12)
]

large_test_cases = [
    np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(10, 20)
]

# large_npy_one_case = [
#     np.array([1]*(2**x), dtype=complex) for x in range(25, 28)
# ]


def print_test_case(test_case: list[np.ndarray]):
    for i, test in enumerate(test_case):
        print(f"test case {i+1}:\n {test}\n")


if __name__ == "__main__":
    print_test_case(simple_test_cases)
    print([x.shape for x in large_test_cases])