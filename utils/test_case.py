import numpy as np

simple_test_cases = [
    np.array([1], dtype=complex),
    np.array([1, 2, 3, 4], dtype=complex),
    np.array([0, 1, 0, -1], dtype=complex),
    np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=complex),
]

large_test_cases = [
    np.random.rand(2**x) + 1j * np.random.rand(2**x) for x in range(20, 25)
]


def print_test_case(test_case: list[np.ndarray]):
    for i, test in enumerate(test_case):
        print(f"test case {i+1}:\n {test}\n")


if __name__ == "__main__":
    print_test_case(simple_test_cases)
    print([x.shape for x in large_test_cases])