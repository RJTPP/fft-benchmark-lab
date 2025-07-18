"""Functions for testing FFT implementations."""

import numpy as np
from scipy.fft import fft as scipy_fft
from time import perf_counter
from .io_utils import qprint, colored_print


def get_func_name(func: callable):
    try:
        return func.__name__
    except AttributeError:
        return "Function"


def test_correctness(func: callable, test_cases: list[np.ndarray], reference_func: callable=scipy_fft, name: str = None, verbose: bool = False):
    is_quiet = not verbose
    results = []
    
    if name is None:
        name = get_func_name(func)
    
    
    qprint(f"🔍 Correctness Testing: {name}...", is_quiet)
    for i, test in enumerate(test_cases):
        res = {
            "func": name,
            "test_no": i + 1,
            "input": test,
            "input_size": len(test),
            "expected": None,
            "output": None,
            "mae": None,
            "mse": None,
            "is_pass": False,
            "is_error": False,
        }
        
        try:
            output = func(test)
            expected = reference_func(test)
            mae = np.mean(np.abs(output - expected))
            mse = np.mean(np.abs(output - expected) ** 2)
            
            
            res["expected"] = expected
            res["output"] = output
            res["mae"] = mae
            res["mse"] = mse
            
            assert np.allclose(output, expected, rtol=1e-5, atol=1e-8)
            
            colored_print(f"  ✅ Test case {i + 1:>2} (size: {len(test):>8}): PASS -> MAE: {mae:<8.2g}, MSE: {mse:>8.2g}", color="GREEN",quiet=is_quiet)
            res["is_pass"] = True
            res["is_error"] = False
        except AssertionError:
            colored_print(f"  ❌ Test case {i + 1:>2} (size: {len(test):>8}): FAIL -> MAE: {mae:<8.2g}, MSE: {mse:>8.2g}", color="RED", quiet=is_quiet)
            res["is_pass"] = False
            res["is_error"] = False
        except Exception as e:
            colored_print(f"  💥 Test case {i + 1:>2} (size: {len(test):>8}): ERROR ({e})", color="YELLOW",quiet=is_quiet)
            res["is_pass"] = False
            res["is_error"] = True
        
        results.append(res)
        
    return results


def test_speed(func: callable, test_cases: list[np.ndarray], name: str = None, verbose: bool = False):
    is_quiet = not verbose
    results = []
    
    if name is None:
        name = get_func_name(func)
        
    qprint(f"🕐 Speed Testing: {name}...", is_quiet)
    
    # Warmup
    warmup_input = np.random.rand(256) + 1j * np.random.rand(256)
    try:
        for _ in range(10):
            func(warmup_input)
    except Exception as e:
        print(f"  ⚠️ Warmup failed: {e}")
    
    # Run
    for i, test in enumerate(test_cases):
        res = {
            "func": name,
            "test_no": i + 1,
            "input": test,
            "input_size": len(test),
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
            is_exceed_thousands = time_used_us > 1000
            unit_str = "ms" if is_exceed_thousands else "µs"
            colored_print(f"  ✅ Time (size: {len(test):>8}): {time_used_us if not is_exceed_thousands else time_used_us/1000:>8.2f} {unit_str} (avg per bin: {avg_time_us:.3f} µs)", color="GREEN", quiet=is_quiet)
            res["time_used_us"] = time_used_us
            res["time_per_bin_us"] = avg_time_us
            res["is_error"] = False
        except Exception as e:
            colored_print(f"  💥 Time: ERROR ({e})", color="YELLOW", quiet=is_quiet)
            res["is_error"] = True
            
        results.append(res)
        
    return results
            

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    import test_case
    
    a = test_correctness(np.fft.fft, test_case.get_simple_test_cases(), verbose=True)
    b = test_speed(np.fft.fft, test_case.get_large_test_cases(), verbose=True)
    c = test_speed(scipy_fft, test_case.get_large_test_cases(), verbose=True)
    
    data = []
    for x in [a, a]:
        data.extend(x)
    
    columns = ["name", "test_no", "input_size", "mae", "mse", "is_pass", "is_error"]
    data = [
        [
            x[y] for y in columns
        ]
        for x in sorted(data, key=lambda x: (x["name"], x["test_no"]))
        ]
    import polars as pl
    print(pl.DataFrame(data, schema=columns))
    
    # for test in a:
    #     print(test)
        
    # for test in b:
    #     print(test)