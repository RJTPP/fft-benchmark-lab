import numpy as np
from scipy.fft import fft as scipy_fft
from time import perf_counter

import test_case


def get_func_name(func: callable):
    try:
        return func.__name__
    except AttributeError:
        return "Function"


def test_correctness(func: callable, test_cases: list[np.ndarray], reference_func: callable=scipy_fft, name: str = None, verbose: bool = False):
    results = []
    
    if name is None:
        name = get_func_name(func)
    
    if verbose:
        print(f"🔍 Correctness Testing: {name}...")
    for i, x in enumerate(test_cases):
        verbose_output = None
        res = {
            "no": i + 1,
            "input": x,
            "expected": None,
            "output": None,
            "mae": None,
            "mse": None,
            "is_pass": False,
            "is_error": False,
        }
        
        try:
            output = func(x)
            expected = reference_func(x)
            mae = np.mean(np.abs(output - expected))
            mse = np.mean(np.square(output - expected))
            
            
            res["expected"] = expected
            res["output"] = output
            res["mae"] = mae
            res["mse"] = mse
            
            assert np.allclose(output, expected, rtol=1e-5, atol=1e-8)
            
            verbose_output = f"  ✅ Test case {i + 1}: PASS (MAE: {mae:.2g}, MSE: {mse:.2g})"
            res["is_pass"] = True
            res["is_error"] = False
        except AssertionError:
            verbose_output = f"  ❌ Test case {i + 1}: FAIL (MAE: {mae:.2g}, MSE: {mse:.2g})"
            res["is_pass"] = False
            res["is_error"] = False
        except Exception as e:
            verbose_output = f"  💥 Test case {i + 1}: ERROR ({e})"
            res["is_pass"] = False
            res["is_error"] = True
        
        if verbose and verbose_output is not None:
            print(verbose_output)
        results.append(res)
        
    return results


def test_speed(func: callable, test_cases: list[np.ndarray], name: str = None, verbose: bool = False):
    results = []
    
    if name is None:
        name = get_func_name(func)
        
    if verbose:
        print(f"🔍 Speed Testing: {name}...")
    
    # Warmup
    warmup_input = np.random.rand(256) + 1j * np.random.rand(256)
    try:
        for _ in range(10):
            func(warmup_input)
    except Exception as e:
        print(f"  ⚠️ Warmup failed: {e}")
    
    # Run
    for i, test in enumerate(test_cases):
        verbose_output = None
        res = {
            "no": i + 1,
            "input": test,
            "time_used_us": None,
            "time_per_bin_us": None,
            "is_error": False
        }
        try:
            # Measure time
            start_time = perf_counter()
            func(test)
            end_time = perf_counter()
            
            time_used_us = (end_time - start_time) * 1e6
            avg_time_us = time_used_us / len(test)
            
            # Output
            verbose_output = f"  ✅ Time: {time_used_us:.2f} µs (avg per bin: {avg_time_us:.2f} µs)"
            res["time_used_us"] = time_used_us
            res["time_per_bin_us"] = avg_time_us
            res["is_error"] = False
        except Exception as e:
            verbose_output = f"  💥 Time: ERROR ({e})"
            res["is_error"] = True
            
        if verbose and verbose_output is not None:
            print(verbose_output)
        results.append(res)
        
    return results
            

if __name__ == "__main__":
    a = test_correctness(np.fft.fft, test_case.simple_test_cases, verbose=True)
    b = test_speed(np.fft.fft, test_case.large_test_cases, verbose=True)
    c = test_speed(scipy_fft, test_case.large_test_cases, verbose=True)
    
    # for test in a:
    #     print(test)
        
    # for test in b:
    #     print(test)