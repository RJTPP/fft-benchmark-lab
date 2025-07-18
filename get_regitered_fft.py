from fft_core import fft_functions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--as-list", "-l", action="store_true")
args = parser.parse_args()

if args.as_list and args.verbose:
    print("WARNING: --as-list and --verbose are mutually exclusive. Using --as-list.")
    print()

if not fft_functions:
    print("No FFT implementations registered.")
    exit(1)

print(f"Found {len(fft_functions)} registered FFT implementations:\n")

if args.as_list:
    print(f"{list(fft_functions.keys())}".replace("\'", "\""))
    exit(0)
    
for i, (name, func) in enumerate(fft_functions.items()):
    if args.verbose:
        print(f"{i+1}. {name} ({func.__module__}.{func.__name__})")
        # print docstring
        print(func.__doc__)
    else:
        print(f"{i+1}. {name}")