# SBL prior normalization — Python companion

Python implementations of **prior-normalizing (KR) transport maps** for sparse Bayesian learning (SBL), packaged as a small library plus **worked examples** that plug into [TRIPS-Py](https://github.com/mpasha3/trips-py) deblurring test problems.

This repository is meant as a **public companion** to:

- **[TRIPS-Py](https://github.com/mpasha3/trips-py)** — regularization and test problems for inverse problems (here: 1D/2D deblurring).
- The paper *Efficient sampling for sparse Bayesian learning using hierarchical prior normalization* (Glaubitz & Marzouk, 2025) and its **[reference Julia code](https://github.com/jglaubitz/paper-2025-SBL-priorNormalization)**.

The Julia repository remains the authoritative source for the full numerical study; this repo focuses on a **minimal, reusable Python core** (`prior_normalization`) and **two scripts** that combine TRIPS-Py forward models with Tikhonov/GCV and prior-normalizing MAP estimation.

## Contents

| Path | Description |
|------|-------------|
| [`src/prior_normalization/`](src/prior_normalization/) | Chebyshev-built Φ and Knothe–Rosenblatt inverse map utilities used by the examples. |
| [`examples/deblurring_1d/run_deblurring_1d.py`](examples/deblurring_1d/run_deblurring_1d.py) | 1D deblurring (`Deblurring1D`): whitening via first differences, four SBL hyperparameter sets from the paper. |
| [`examples/deblurring_2d/run_deblurring_2d.py`](examples/deblurring_2d/run_deblurring_2d.py) | 2D deblurring (`Deblurring2D`): pixel-domain MAP; requires TRIPS-Py demo image `.mat` files (see below). |
| [`examples/bootstrap.py`](examples/bootstrap.py) | Resolves `prior_normalization` on `sys.path` and locates TRIPS-Py (`TRIPS_PY_ROOT` or common clone locations). |

Reference figures shipped with the project live under each example’s `figures/reference_*.png`. Running an example writes `figures/results_deblurring_*.png` (ignored by git unless you choose to commit them).

## Citation

If you use this code or the underlying method, cite the article:

```bibtex
@article{glaubitz2025efficient,
  title   = {Efficient sampling for sparse Bayesian learning using hierarchical prior normalization},
  author  = {Glaubitz, Jan and Marzouk, Youssef},
  journal = {arXiv preprint arXiv:2505.23753},
  year    = {2025},
  month   = {05}
}
```

Also cite TRIPS-Py if you use its operators and test problems; see that repository for their preferred reference.

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib, PyLops (declared in [`pyproject.toml`](pyproject.toml))
- A checkout of **TRIPS-Py** that includes `trips.test_problems` (clone from GitHub; PyPI packages may not match this layout).

## Installation

From the repository root:

```bash
git clone https://github.com/<your-org>/SBL-PriorNormalization-Python.git
cd SBL-PriorNormalization-Python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt   # optional lock-style install without extras
```

### Point the examples at TRIPS-Py

Clone TRIPS-Py somewhere on your machine, then either:

```bash
export TRIPS_PY_ROOT=/absolute/path/to/trips-py
```

or place the clone at `~/Documents/trips-py` or `~/trips-py`, which [`examples/bootstrap.py`](examples/bootstrap.py) discovers automatically.

The 2D script temporarily sets the process working directory to `<trips-py>/demos` so `Deblurring2D` can load `./data/image_data/*.mat`; the prior directory is restored afterward.

## Running the examples

Run from any working directory; use the repo’s Python so editable `prior_normalization` resolves:

```bash
export TRIPS_PY_ROOT=/path/to/trips-py   # if not auto-discovered
export MPLBACKEND=Agg                    # optional: non-interactive plotting

python examples/deblurring_1d/run_deblurring_1d.py
python examples/deblurring_2d/run_deblurring_2d.py
```

**2D runtime:** The full default settings (`24×24` unknowns, four SBL models) involve large-scale optimization and can take a long time. For a **smaller dry run** (e.g. CI or laptops):

```bash
export SBL_EXAMPLE_QUICK=1
python examples/deblurring_2d/run_deblurring_2d.py
```

This uses a coarser grid and a single SBL model with a reduced iteration budget (see the script). Omit `SBL_EXAMPLE_QUICK` for results comparable to the shipped reference PNGs.

## Scope and disclaimer

This repository does **not** reproduce every experiment from the Julia paper; it provides the **prior-normalization machinery in Python** and **two TRIPS-Py deblurring demonstrations** (Tikhonov/GCV vs prior-normalizing MAP). The code is provided as-is for research and education.

## License

MIT — see [LICENSE](LICENSE).

## CMDA capstone (Virginia Tech)

This work was carried out as a **Computational Modeling and Data Analytics (CMDA)** capstone project at Virginia Tech.

| Role | People |
|------|--------|
| **Project team** | Daniel Miles, Ash Vadicherla, Jun Young Kwon |
| **Project sponsor** | Dr. Mirjeta Pasha |

## Acknowledgments

The project builds on research by Jan Glaubitz and Youssef Marzouk and on the [TRIPS-Py](https://github.com/mpasha3/trips-py) software ecosystem.
