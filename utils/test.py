import numpy as np
from scipy.fft import fft as scipy_fft
from time import perf_counter

import test_case


def get_func_name(func: callable):
    try:
        return func.__name__
    except AttributeError:
        return "Function"


def test_correctness(func: callable, test_cases: list[np.ndarray], reference_func: callable=scipy_fft, name: str = None):
    if name is None:
        name = get_func_name(func)
        
    print(f"🔍 Correctness Testing: {name}...")
    for i, x in enumerate(test_cases):
        try:
            output = func(x)
            expected = reference_func(x)
            mae = np.mean(np.abs(output - expected))
            mse = np.mean(np.square(output - expected))
            assert np.allclose(output, expected, rtol=1e-5, atol=1e-8)
            print(f"  ✅ Test case {i + 1}: PASS (MAE: {mae:.2g}, MSE: {mse:.2g})")
        except AssertionError:
            print(f"  ❌ Test case {i + 1}: FAIL (MAE: {mae:.2g}, MSE: {mse:.2g})")
        except Exception as e:
            print(f"  💥 Test case {i + 1}: ERROR ({e})")


def test_speed(func: callable, test_cases: list[np.ndarray], name: str = None):
    if name is None:
        name = get_func_name(func)
        
    print(f"🔍 Speed Testing: {name}...")
    start_time = perf_counter()
    for x in test_cases:
        func(x)
    end_time = perf_counter()
    time_used_ms = (end_time - start_time) * 1e6
    avg_time_ms = time_used_ms / len(test_cases)
    print(f"  ✅ Time: {time_used_ms:.2f} ms (avg: {avg_time_ms:.2f} ms)")
            

if __name__ == "__main__":
    test_correctness(np.fft.fft, test_case.simple_test_cases)
    test_speed(np.fft.fft, test_case.simple_test_cases)