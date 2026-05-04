"""
Bootstrap imports for the deblurring examples.

Why this module exists
----------------------
The examples need two separate codebases on ``sys.path``:

1. **This repository** — the ``prior_normalization`` package under ``src/``, which
   implements Chebyshev-discretized Φ and the Knothe–Rosenblatt inverse map used
   in prior-normalizing MAP (see Glaubitz & Marzouk, 2025).

2. **TRIPS-Py** — a *different* Git checkout that provides ``trips.test_problems``
   (e.g. ``Deblurring1D``, ``Deblurring2D``) and utilities such as GCV. The PyPI
   package name may not match a full clone layout, so we resolve a filesystem path
   and prepend it to ``sys.path`` explicitly.

Neither layout is guaranteed to be on ``PYTHONPATH`` when someone runs a script
from an arbitrary working directory, so ``repo_root()`` walks upward from this file
until it finds ``src/prior_normalization/``, and ``find_trips_py_root()`` locates
TRIPS-Py using ``TRIPS_PY_ROOT`` or heuristics documented in ``find_trips_py_root``.
"""

from __future__ import annotations

import os
import pathlib
import sys


def repo_root() -> pathlib.Path:
    """
    Absolute path to the root of *this* repository (the folder that contains
    ``src/prior_normalization``).

    Algorithm: start at ``examples/bootstrap.py``'s directory (``examples/``),
    then walk to parents until ``src/prior_normalization`` exists. That works
    whether you run scripts from the repo root or from inside ``examples/``.
    """
    here = pathlib.Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if (p / "src" / "prior_normalization").is_dir():
            return p
    raise RuntimeError(
        "Could not locate repository root (expected src/prior_normalization/). "
        "Run examples from the cloned repo."
    )


def ensure_prior_normalization_on_path() -> pathlib.Path:
    """
    Prepend ``<repo>/src`` to ``sys.path`` so ``import prior_normalization`` succeeds.

    Returns
    -------
    pathlib.Path
        Repository root (same as ``repo_root()``).
    """
    root = repo_root()
    src = root / "src"
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)
    return root


def _is_trips_py(path: pathlib.Path) -> bool:
    """True if ``path`` looks like the root of a TRIPS-Py clone with test problems."""
    return (path / "trips" / "test_problems").is_dir()


def find_trips_py_root() -> pathlib.Path:
    """
    Locate the root directory of a TRIPS-Py installation (folder containing ``trips/``).

    Resolution order
    ----------------
    1. **``TRIPS_PY_ROOT``** — if set, must point at a directory that contains
       ``trips/test_problems/``. Raises if set but invalid (fail fast).
    2. **Fixed guesses** — ``~/Documents/trips-py``, ``~/trips-py``, and a sibling
       ``../trips-py`` next to this repo (common student layout).
    3. **Breadth-first-ish walk** — under ``~/Documents`` then ``~``, look for a
       subdirectory named ``trips-py`` that passes ``_is_trips_py``. This can be
       slow on large home directories; prefer ``TRIPS_PY_ROOT`` in production.

    Returns
    -------
    pathlib.Path
        Resolved absolute path to the TRIPS-Py repo root.

    Raises
    ------
    RuntimeError
        If no valid TRIPS-Py root is found, or ``TRIPS_PY_ROOT`` is wrong.
    """
    env = os.environ.get("TRIPS_PY_ROOT", "").strip()
    if env:
        candidate = pathlib.Path(env).expanduser().resolve()
        if _is_trips_py(candidate):
            return candidate
        raise RuntimeError(
            f"TRIPS_PY_ROOT={env!r} is set but does not contain trips/test_problems/"
        )

    root = repo_root()
    for candidate in (
        pathlib.Path.home() / "Documents" / "trips-py",
        pathlib.Path.home() / "trips-py",
        root.parent / "trips-py",
    ):
        if _is_trips_py(candidate):
            return candidate.resolve()

    for search_root in (pathlib.Path.home() / "Documents", pathlib.Path.home()):
        if not search_root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(str(search_root)):
            if "trips-py" in dirnames:
                cand = pathlib.Path(dirpath) / "trips-py"
                if _is_trips_py(cand):
                    return cand.resolve()

    raise RuntimeError(
        "trips-py not found. Clone https://github.com/mpasha3/trips-py and set "
        "TRIPS_PY_ROOT to that directory, or place it at ~/Documents/trips-py."
    )


def ensure_trips_py_on_path() -> pathlib.Path:
    """
    Prepend the TRIPS-Py repository root to ``sys.path`` so ``import trips...`` works.

    Returns
    -------
    pathlib.Path
        The TRIPS-Py root directory (same as ``find_trips_py_root()``).
    """
    trips_root = find_trips_py_root()
    p = str(trips_root)
    if p not in sys.path:
        sys.path.insert(0, p)
    return trips_root
