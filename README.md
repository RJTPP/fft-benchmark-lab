# ⚡ FFT Benchmark Lab

A Python benchmarking toolkit (originally developed for the 01204496-65 DSP course at Kasetsart University) for evaluating custom FFT implementations against SciPy.  

**Key capabilities:**
- Hand-written FFT algorithms (recursive, iterative, radix-4, etc.)  
- Automated error analysis (MAE, MSE, pass/fail) and speed benchmarking  
- CLI controls: `--mode`, `--minimal`, `--save-csv`  
- Colorized terminal output and organized, timestamped CSV results  

## 📜 Table of Contents

- [⚡ FFT Benchmark Lab](#-fft-benchmark-lab)
  - [📜 Table of Contents](#-table-of-contents)
  - [📂 Project Structure](#-project-structure)
  - [📥 Getting Started](#-getting-started)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Running Benchmark](#running-benchmark)
    - [Optional flags](#optional-flags)
    - [Custom Implementations](#custom-implementations)
    - [Listing Registered FFT Implementations](#listing-registered-fft-implementations)
  - [📊 Example Output](#-example-output)
    - [Console Output](#console-output)
    - [CSV Output Format](#csv-output-format)
  - [📄 License](#-license)
  - [🧑‍💻 Contributors](#-contributors)


## 📂 Project Structure

```
project/
├── fft_core/
│   ├── __init__.py
│   ├── selection.py       # Helper file for importing FFT implementations
│   └── ...                # FFT implementations
│
├── util/
│   ├── __init__.py
│   ├── csv_utils.py       # CSV utilities for saving results
│   ├── io_utils.py        # I/O utilities for colored and silent output
│   ├── test_case.py       # Predefined test signals
│   └── test.py            # Benchmark and correctness wrapper
│
├── get_registered_fft.py  # CLI for listing registered FFT implementations
├── main.py                # CLI entry point for benchmarking
├── README.md              # Project overview (this file)
├── uv.lock                # uv package lock file
├── requirements.txt       # Python dependencies (if not using uv)
└── pyproject.toml         # Project metadata
```

## 📥 Getting Started


### Installation

**Clone this repository:**

```bash
git clone https://github.com/RJTPP/fft-benchmark-lab.git
cd fft-benchmark-lab
```

**Install Dependencies**

This project use [`uv`](https://github.com/astral-sh/uv) for package management and installation.

To install the project dependencies, run:

```bash
uv sync
```

However, you can also use `pip` if preferred:

```bash
pip install -r requirements.txt
```


### Usage

### Running Benchmark

To run the benchmark tests, execute the main script:

```bash
uv run main.py
```

Or if you prefer using `python`:

```bash
python main.py
```

### Optional flags

- `--mode [all|check|speed]` — Run only metrics tests, speed tests, or both (default: all)
- `--save-csv [DIR]` — Save results to a timestamped directory (e.g., `results_20250101_000000/`)
  - If no directory is provided, a default folder will be created
- `--minimal` — Reduce test output to minimal


### Custom Implementations

You can benchmark any function that takes a 1D `np.ndarray` and returns a `np.ndarray` (e.g., DFT, FFT, or other spectral transforms). 

By default, results are compared against `scipy.fft.fft`.

To add a new FFT implementation:

1. Create a new file in `fft_core/`, e.g. `fft_myalgo.py`.
2. Decorate your FFT function with `@register_fft` from `fft_core.selection`:
   ```python
    import numpy as np
    from fft_core.selection import register_fft

    @register_fft
    # Or use a custom name: @register_fft(name="myalgo")
    def fft(x: np.ndarray) -> np.ndarray:
        """Your FFT algorithm here"""
        return ...
   ```
    - Each function will be auto-registered by name.
    - If multiple functions share the same name, they will be renamed automatically (e.g., `fft`, `fft_1`, `fft_2`, …).
    - The function must accept a 1D np.ndarray of complex values and return a transformed np.ndarray of the same shape and type.

3. The main script will automatically detect `fft_myalgo.fft` and include it in benchmarks.

> [!TIP]
> To organize your implementations, you can use subfolders in fft_core/ (with `__init__.py`), e.g. `fft_core/mygroup/fft_cool.py`

### Listing Registered FFT Implementations

To list all registered FFT implementations, run:

```bash
uv run get_registered_fft.py [-l | --list] [-v | --verbose]

# Or `python get_registered_fft.py`
```

## 📊 Example Output

### Console Output

```
🔍 Correctness Testing: simple_fft...
  ✅ Test case 1: PASS (MAE: 0, MSE: 0+0j)
  ✅ Test case 2: PASS (MAE: 0, MSE: 0+0j)
  ...

🕐 Speed Testing: simple_fft...
  ✅ Time (size: 1024): 17.92 µs (avg per bin: 0.018 µs)
  ✅ Time (size: 2048): 42.68 µs (avg per bin: 0.021 µs)
  ...

🗂️  Saving results...
  💾  Saved simple_fft metrics to ...
  ...
```


### CSV Output Format

Benchmark results are saved under the `/results/` folder in a timestamped subdirectory, e.g., `results/results_20250101_000000/`, with the following structure:

```
results/
 ┗ results_YYYYMMDD_HHMMSS/  # Or the provided directory name
   ┣ metrics/
   ┃ ┣ FUNC_NAME.csv         # Individual metric results per function
   ┃ ┗ ...
   ┣ speed/
   ┃ ┣ FUNC_NAME.csv         # Individual speed results per function
   ┃ ┗ ...
   ┣ metrics.csv             # Combined metrics for all functions
   ┗ speed.csv               # Combined speed for all functions
```

**metrics.csv** and metrics/FUNC_NAME.csv share the same format:

```csv
func,test_no,input_size,mae,mse,is_pass,is_error
```

- **func**: FFT function name  
- **test_no**: index of the test case  
- **input_size**: signal length  
- **mae**: mean absolute error  
- **mse**: mean squared error  
- **is_pass**: whether the result matched tolerance (`true`/`false`)  
- **is_error**: whether an exception occurred (`true`/`false`)  

**speed.csv** and speed/FUNC_NAME.csv share the same format:

```csv
func,test_no,input_size,time_used_us,time_per_bin_us,is_error
```

- **func**: FFT function name  
- **test_no**: index of the test case  
- **input_size**: signal length  
- **time_used_us**: total execution time in microseconds  
- **time_per_bin_us**: average time per FFT bin  
- **is_error**: whether an exception occurred during timing (`true`/`false`)  


## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🧑‍💻 Contributors

Rajata Thamcharoensatit ([@RJTPP](https://github.com/RJTPP))
