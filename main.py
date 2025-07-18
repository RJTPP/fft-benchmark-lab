"""Main script for testing FFT implementations."""

import logging

# 1. Configure the root logger:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)

# 2. (Optional) If you want all modules to inherit this level:
logging.getLogger().setLevel(logging.INFO)


import argparse
import polars as pl
from numpy.fft import fft as numpy_fft
from scipy.fft import fft as scipy_fft
from utils import test, test_case, csv_utils
from fft_core import fft_functions

fft_functions = {
    "numpy": numpy_fft,
    "scipy": scipy_fft,
    **fft_functions
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", help="quiet mode", action="store_true")
    parser.add_argument("-m", "--mode", help="test mode: all, check(correctness), speed", choices=["all", "check", "speed"], default="all")
    parser.add_argument("-t", "--table", help="output as table", action="store_true")
    return parser.parse_args()


def qprint(output="", quiet=False, **kwargs):
    if not quiet:
        print(output, **kwargs)
    

def test_fft_correctness(testcase, verbose=True) -> pl.DataFrame:
    columns = ["func", "test_no", "input_size", "mae", "mse", "is_pass", "is_error"]
    results = []
    for name, func in fft_functions.items():
        res = test.test_correctness(
            func, 
            testcase, 
            reference_func=scipy_fft, 
            name=name,
            verbose=verbose,
        )
        results.extend(res)
        
    results = [
        [x[y] for y in columns]
        for x in sorted(results, key=lambda x: (x["func"], x["test_no"]))
    ]
    return pl.DataFrame(results, schema=columns, orient="row")
    


def test_fft_speed(testcase, verbose=True) -> pl.DataFrame:
    columns = ["func", "test_no", "input_size", "time_used_us", "time_per_bin_us", "is_error"]
    results = []
    for name, func in fft_functions.items():
        res = test.test_speed(
            func, 
            testcase, 
            name=name,
            verbose=verbose,
        )
        results.extend(res)
        
    results = [
        [x[y] for y in columns]
        for x in sorted(results, key=lambda x: (x["func"], x["test_no"]))
    ]
    return pl.DataFrame(results, schema=columns, orient="row")


if __name__ == "__main__":
    # Handle args
    args = get_args()
    is_quiet = args.quiet
    
    if is_quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Silent normal output if use --table mode
    if args.table:
        is_quiet = True
        
    is_verbose = not is_quiet
    

    
    # Warm up
    qprint(quiet=is_quiet)
    qprint("Warming up...", quiet=is_quiet)
    test_fft_correctness(test_case.get_simple_test_cases(), verbose=False)
    
    # Test Correctness
    if args.mode in ["check", "all"]:
        qprint(quiet=is_quiet)
        qprint("Testing correctness...", quiet=is_quiet)
        qprint(quiet=is_quiet)
        
        correctness_df = test_fft_correctness(test_case.get_combined_test_cases(), verbose=is_verbose)
        if args.table:
            with pl.Config(tbl_rows=-1):
                qprint("Correctness", quiet=args.quiet)
                qprint(correctness_df, quiet=args.quiet)
    
    # Test Speed
    if args.mode in ["speed", "all"]:
        qprint(quiet=is_quiet)
        qprint("Testing speed...", quiet=is_quiet)
        qprint(quiet=is_quiet)
        speed_df = test_fft_speed(test_case.get_large_test_cases_extended(), verbose=is_verbose)
        if args.table:
            with pl.Config(tbl_rows=-1):
                qprint("Speed", quiet=args.quiet)
                qprint(speed_df, quiet=args.quiet)