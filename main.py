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
from datetime import datetime
from pathlib import Path

import polars as pl
from numpy.fft import fft as numpy_fft
from scipy.fft import fft as scipy_fft

from fft_core import fft_functions
from utils import csv_utils, test, test_case
from utils.io_utils import colored_print, qprint

RESULT_DIR = "results"

fft_functions = {
    "scipy": scipy_fft,
    # "numpy": numpy_fft,
    **fft_functions
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="test mode: all, metrics, speed", choices=["all", "metrics", "speed"], default="all")
    parser.add_argument("-t", "--table", help="output as table", action="store_true")
    parser.add_argument(
        "-s", "--save-csv",
        metavar="FILENAME",
        nargs="?",
        const=True,  # Temporary placeholder to detect usage without value
        help="Optionally save results to a CSV file. If no filename is provided, uses results_YYYYMMDD_HHMMSS.csv"
    )
    parser.add_argument("--minimal", help="Reduce output verbosity during tests", action="store_true")
    return parser.parse_args()
    

def test_fft_metrics(testcase, verbose=True) -> pl.DataFrame:
    columns = ["func", "test_no", "input_size", "mae", "mse", "is_pass", "is_error"]
    results = []
    for name, func in fft_functions.items():
        res = test.test_metrics(
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
    is_quiet = args.minimal
    
    if is_quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Silent normal output if use --table mode
    if args.table:
        is_quiet = True
        
    is_verbose = not is_quiet
    
    metrics_df = None
    speed_df = None

    
    # Warm up
    qprint(quiet=is_quiet)
    qprint("Warming up...", quiet=is_quiet)
    test_fft_metrics(test_case.get_simple_test_cases(), verbose=False)
    
    # Test metrics
    if args.mode in ["metrics", "all"]:
        qprint(quiet=is_quiet)
        qprint("Testing metrics...", quiet=is_quiet)
        qprint(quiet=is_quiet)
        
        metrics_df = test_fft_metrics(test_case.get_combined_test_cases(), verbose=is_verbose)
        if args.table:
            with pl.Config(tbl_rows=-1):
                qprint("Metrics", quiet=args.quiet)
                qprint(metrics_df, quiet=args.quiet)
    
    # Test Speed
    if args.mode in ["speed", "all"]:
        qprint(quiet=is_quiet)
        qprint("Testing speed...", quiet=is_quiet)
        qprint(quiet=is_quiet)
        speed_df = test_fft_speed(test_case.get_massive_test_cases(), verbose=is_verbose)
        if args.table:
            with pl.Config(tbl_rows=-1):
                qprint("Speed", quiet=args.quiet)
                qprint(speed_df, quiet=args.quiet)
                
    # Save to CSV
    if args.save_csv:
        print("\n🗂️  Saving results…")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine base output directory
        base_dir = Path(RESULT_DIR) / (args.save_csv is True and f"results_{timestamp}" or args.save_csv)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create sub-dirs
        metrics_dir = base_dir / "metrics"
        speed_dir   = base_dir / "speed"
        metrics_dir.mkdir(exist_ok=True)
        speed_dir.mkdir(exist_ok=True)

        # Save combined CSVs
        if metrics_df is not None:
            combined_metrics = base_dir / "metrics.csv"
            csv_utils.df_to_csv(metrics_df, combined_metrics)
            colored_print(f"  💾  Saved {'combined':<20} metrics to {combined_metrics}", color="CYAN")

            # Per-function metrics
            for func in metrics_df["func"].unique():
                func_df  = metrics_df.filter(pl.col("func") == func)
                func_path = metrics_dir / f"{func}_metrics.csv"
                csv_utils.df_to_csv(func_df, func_path)
                colored_print(f"  💾  Saved {func:<20} metrics to {func_path}", color="CYAN")

        if speed_df is not None:
            combined_speed = base_dir / "speed.csv"
            csv_utils.df_to_csv(speed_df, combined_speed)
            colored_print(f"  💾  Saved {'combined':<20} speed to {combined_speed}", color="CYAN")

            # Per-function speed
            for func in speed_df["func"].unique():
                func_df  = speed_df.filter(pl.col("func") == func)
                func_path = speed_dir / f"{func}_speed.csv"
                csv_utils.df_to_csv(func_df, func_path)
                colored_print(f"  💾  Saved {func:<20} speed to {func_path}", color="CYAN")