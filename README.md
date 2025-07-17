# ⚡ FFT Project

**This project was developed as part of the 01204496-65 Algorithmic-Oriented Digital Signal Processing for Computer Engineers course at Kasetsart University.**

 It demonstrates how the Fast Fourier Transform (FFT) works through practical Python implementations, comparing performance and accuracy against `numpy.fft` and `scipy.fft` functions.

## 📜 Table of Contents

- [⚡ FFT Project](#-fft-project)
  - [📜 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [📂 Project Structure](#-project-structure)
  - [📥 Getting Started](#-getting-started)
  - [🔧 Installation](#-installation)
  - [🚀 Usage](#-usage)
    - [📈 Running Benchmark](#-running-benchmark)
    - [✏️ Custom FFT Implementations](#️-custom-fft-implementations)
  - [📊 Example Output](#-example-output)
  - [📄 License](#-license)
  - [🧑‍💻 Contributors](#-contributors)


## ✨ Features

* Recursive and iterative FFT implementations in `fft_core/`
* Performance benchmarking with configurable tolerance and timing in `util/test.py`
* Test signals defined in `util/test_case.py`
* Simple CLI entry point in `main.py`



## 📂 Project Structure

```
project/
├── fft_core/
│   ├── __init__.py
│   └── ...            # FFT implementations
│
├── util/
│   ├── __init__.py
│   ├── test_case.py   # Predefined test signals
│   └── test.py        # Benchmark and correctness wrapper
│
├── main.py            # CLI entry point for benchmarking
├── README.md          # Project overview (this file)
├── uv.lock            # uv package lock file
├── requirements.txt   # Python dependencies (if not using uv)
└── pyproject.toml     # Project metadata
```

## 📥 Getting Started

Clone this repository:

```bash
git clone https://github.com/your-username/custom-fft-lab.git
cd custom-fft-lab
```


## 🔧 Installation

This project use [`uv`](https://github.com/astral-sh/uv) for package management and installation.

To install the project dependencies, run

```bash
uv sync
```

However, you can also use `pip` if preferred

```bash
pip install -r requirements.txt
```


## 🚀 Usage

### 📈 Running Benchmark

To run the benchmark tests, execute the main script:

```bash
uv run main.py
```

Or if you prefer using `python`:

```bash
python main.py
```


### ✏️ Custom FFT Implementations

To add a new FFT implementation:

1. Create a new file in `fft_core/`, e.g. `fft_myalgo.py`.
2. Decorate your FFT function with `@register_fft` from `fft_core.selection`:
   ```python
    from .selection import register_fft
    import numpy as np

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


## 📊 Example Output

```
🔍 Correctness Testing: simple_fft...
  ✅ Test case 1: PASS (MAE: 0, MSE: 0+0j)
  ✅ Test case 2: PASS (MAE: 0, MSE: 0+0j)

🕐 Speed Testing: simple_fft...
  ✅ Time (size: 1024): 17.92 µs (avg per bin: 0.018 µs)
  ✅ Time (size: 2048): 42.68 µs (avg per bin: 0.021 µs)
```


## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🧑‍💻 Contributors

Rajata Thamcharoensatit ([@RJTPP](https://github.com/RJTPP))
