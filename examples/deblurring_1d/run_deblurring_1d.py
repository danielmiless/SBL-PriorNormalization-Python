#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1D deblurring: TRIPS-Py forward model + Tikhonov/GCV + prior-normalizing MAP
=============================================================================

Problem setup (inverse problem)
-------------------------------
We observe a blurred, noisy vector ``b`` on a 1D grid of length ``N``:

    b = A x + e,

where ``A`` is a convolution-type blur (built by TRIPS-Py's ``Deblurring1D``),
``x`` is the unknown signal, and ``e`` is Gaussian noise. The goal is to recover
``x`` from ``b``.

Why "whitening" with a first-difference matrix ``L``
-----------------------------------------------------
The piecewise-constant truth used here is **not** sparse in the pixel basis, but
its **first differences** are sparse (jumps only at edges). Writing

    z = L x,   hence   x = L^{-1} z

with ``L`` a first-difference operator turns the problem into recovering a sparse
``z``. Substituting into the forward model,

    b = A x = A L^{-1} z = F z,

so we optimize in ``z``-space with **whitened forward** ``F = A @ L_inv``. The
SBL / prior-normalizing machinery is applied to ``z``; we map back with
``x = L_inv @ z`` for plotting and RMSE.

Baseline: Tikhonov + generalized cross-validation (GCV)
--------------------------------------------------------
We solve ``min_z ||F z - b||^2 + λ ||z||^2`` and choose ``λ`` by GCV using the
same TRIPS-Py helper used inside their Tikhonov solvers. This is a standard
quadratic regularizer (smoothness in ``z``-space).

Prior-normalizing MAP
-----------------------
Instead of a simple Gaussian prior on ``z``, we use a **hierarchical sparse
Bayesian learning (SBL)** prior controlled by hyperparameters ``(r, β, ϑ)``.
Following Glaubitz & Marzouk (2025), a **prior-normalizing Knothe–Rosenblatt (KR)
transport** maps a reference standard normal vector ``ξ`` (stacked auxiliary
variables ``(τ, u)`` per component) to ``(z, θ)`` in the physical/hyperparameter
space. MAP estimation is done by minimizing ``-log p(ξ | b)`` (implemented as a
least-squares data term plus a quadratic penalty on ``ξ`` in the code path
below—see the ``neg_log_posterior`` definition).

TRIPS-Py naming convention
--------------------------
In GCV calls, ``F`` plays the role of ``R_A`` (forward after any reparameterization),
and ``b_noisy`` is ``b_vec``.

References
----------
Glaubitz & Marzouk, "Efficient sampling for sparse Bayesian learning using
hierarchical prior normalization", arXiv:2505.23753 (2025). Hyperparameter sets
``(r, β, ϑ)`` in ``models`` match their Table 1 labeling as "Model 1"–"Model 4".
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import path: sibling ``examples/bootstrap.py`` must be importable as ``bootstrap``
# ---------------------------------------------------------------------------
_EXAMPLES = Path(__file__).resolve().parent.parent  # .../examples
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))
from bootstrap import ensure_prior_normalization_on_path, ensure_trips_py_on_path

ensure_prior_normalization_on_path()
ensure_trips_py_on_path()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime

# TRIPS-Py's Deblurring1D still references np.int0, removed in NumPy 2.x.
if not hasattr(np, 'int0'):
    np.int0 = np.intp  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Heavy TRIPS-Py / pylops import order and optional-dependency stubs
# ---------------------------------------------------------------------------
# pylops probes for an "astra" CUDA package at import time. Import pylops first
# so it can disable astra cleanly; then stub optional modules Deblurring1D might
# import (tomography / imaging extras we do not need for this 1D demo).
import pylops  # noqa: F401

from unittest.mock import MagicMock
for _mod in ('astra', 'PIL', 'PIL.Image', 'resizeimage', 'resizeimage.resizeimage', 'requests'):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from trips.test_problems.Deblurring1D import Deblurring1D
from trips.utilities.reg_param.gcv import generalized_crossvalidation

from prior_normalization import (
    make_phi_chebyshev,
    priorNormalizing_KR_inv,
    priorNormalizing_KR_inv_tu,
)

print('Environment ready.')

# =============================================================================
# 1. Build discrete forward model ``A`` and noisy data ``b_noisy``
# =============================================================================
np.random.seed(42)

N           = 64       # grid size
noise_level = 0.03     # relative noise: ||e|| / ||b_true||
blur_param  = 2.0      # Gaussian PSF standard deviation (in pixel units)

# ``CommitCrime=True`` uses the same random draw for the noise instance (TRIPS-Py flag).
Deblur = Deblurring1D(CommitCrime=True)

# Built-in piecewise-constant phantom (sparse gradient).
x_true = Deblur.gen_xtrue(N, test='piecewise').flatten()
t      = np.linspace(0, 1, N)  # unit interval coordinates for plots

# ``A_op`` is a pylops linear operator representing blur; we materialize dense ``A``
# so we can reuse the same ``F`` in SciPy's ``minimize`` and dense GCV algebra.
A_op = Deblur.forward_Op_1D(parameter=blur_param, nx=N, boundary_condition='reflect')

A_dense = np.zeros((N, N))
for j in range(N):
    ej = np.zeros(N)
    ej[j] = 1.0
    A_dense[:, j] = A_op.matvec(ej)

# Noiseless blurred signal and additive Gaussian noise (TRIPS-Py helper).
b_true         = Deblur.gen_data(x_true).flatten()
b_noisy, delta = Deblur.add_noise(b_true.reshape(-1, 1), 'Gaussian', noise_level)
b_noisy        = b_noisy.flatten()

# ``delta`` is returned as the noise norm; for i.i.d. standard normal in R^N,
# E[||e||^2] = N, so we estimate a scalar noise variance per component.
sigma_sq = (delta ** 2) / N

print(f'N={N} | blur_param={blur_param} | noise_level={noise_level}')
print(f'delta={delta:.4f} | sigma_sq={sigma_sq:.2e}')

# =============================================================================
# 2. Whitening: z = L x,  x = L_inv z,  F = A L_inv
# =============================================================================
# L is (I - shift-down): (Lx)_i ≈ x_i - x_{i-1}.  L_inv is cumulative sum so that
# L_inv @ (L @ x) ≈ x up to boundary conventions encoded in the dense matrices here.
L     = np.eye(N) - np.diag(np.ones(N - 1), k=-1)
L_inv = np.tril(np.ones((N, N)))

z_true = L @ x_true
F      = A_dense @ L_inv

assert np.allclose(F @ z_true, b_true, atol=1e-8), 'whitening consistency check failed'
print(f'F shape: {F.shape} | z_true nonzeros: {np.sum(np.abs(z_true) > 1e-10)}')

# =============================================================================
# 3. Tikhonov baseline: min_z ||Fz - b||^2 + λ||z||^2,  λ from GCV
# =============================================================================
# Arguments (Q_A, R_A, R_L, b) match TRIPS-Py's Tikhonov/GCV calling convention:
#   Q_A = I  → no additional projection on the data side
#   R_A = F  → whitened forward
#   R_L = I  → identity penalty on z (standard Tikhonov in z-space)
lam_tikh = generalized_crossvalidation(
    np.eye(N),
    F,
    np.eye(N),
    b_noisy.reshape(-1, 1),
)

# Normal equations: (F^T F + λ I) z = F^T b
z_tikh    = np.linalg.solve(F.T @ F + lam_tikh * np.eye(N), F.T @ b_noisy)
x_tikh    = L_inv @ z_tikh
rmse_tikh = np.sqrt(np.mean((x_tikh - x_true) ** 2))
print(f'\nTikhonov GCV:  lambda={lam_tikh:.4e}  RMSE={rmse_tikh:.4f}')


def prior_normalizing_MAP(F, b, sigma_sq, r, beta, vartheta,
                          n_cheb=64, B=5.0, jac_eps=1e-5, verbose=True):
    """
    Compute a MAP estimate of ``z`` (whitened signal) using the prior-normalizing KR map.

    Optimization vector ``xi`` (length ``2N``) interleaves auxiliary variables::

        xi[0::2] → τ (half the KR coordinates)
        xi[1::2] → u (other half)

    ``priorNormalizing_KR_inv_tu`` maps ``(u, τ)`` to ``z`` for fixed hyperparameters
    ``(r, β, ϑ)``; ``Phi`` is a Chebyshev-based discretization of the map's Φ functional.

    The objective combines:
      * Gaussian likelihood: ``(1/(2σ²)) ||F z(ξ) - b||²``
      * Reference prior on ``ξ``: ``½ ||ξ||²``  (standard normal in ξ-space)

    Parameters
    ----------
    F : (N, N) ndarray
        Whitened forward operator.
    b : (N,) ndarray
        Noisy observations.
    sigma_sq : float
        Scalar observation variance (estimated above).
    r, beta, vartheta : float
        SBL hyperparameters (see paper Table 1).
    n_cheb : int
        Number of Chebyshev nodes for Φ.
    B : float
        Half-width of bounded domain for Chebyshev grid in map construction.
    jac_eps : float
        Finite-difference step for ``approx_fprime`` (L-BFGS-B wants a gradient).
    verbose : bool
        Print optimizer progress.

    Returns
    -------
    z_MAP : (N,) ndarray
        MAP estimate in ``z``-space.
    theta_MAP : (N,) ndarray
        MAP local variance / scale parameters from the KR inverse map.
    result : scipy.optimize.OptimizeResult
        Full optimizer diagnostic object.
    """
    N   = F.shape[1]
    Phi = make_phi_chebyshev(beta, r, n_points=n_cheb)

    def neg_log_posterior(xi):
        # Unpack interleaved τ, u and push through KR map to physical z.
        tau = xi[0::2]
        u   = xi[1::2]
        z   = priorNormalizing_KR_inv_tu(u, tau, r, beta, vartheta, Phi, B=B)
        residual = F @ z - b
        return (0.5 * np.dot(residual, residual) / sigma_sq
                + 0.5 * np.dot(xi, xi))

    jac = lambda xi: approx_fprime(xi, neg_log_posterior, jac_eps)

    if verbose:
        print(f'  Optimizing [N={N}, params={2*N}] ...', end=' ', flush=True)

    result = minimize(
        neg_log_posterior,
        x0=np.zeros(2 * N),
        method='L-BFGS-B',
        jac=jac,
        options=dict(maxiter=50_000, ftol=1e-10, gtol=1e-8),
    )

    tau_MAP = result.x[0::2]
    u_MAP   = result.x[1::2]
    z_MAP, theta_MAP = priorNormalizing_KR_inv(
        u_MAP, tau_MAP, r, beta, vartheta, Phi, B=B
    )

    if verbose:
        print(f'iters={result.nit} | converged={result.success}')

    return z_MAP, theta_MAP, result


# Four hierarchical SBL models from Glaubitz & Marzouk (2025), Table 1.
models = [
    ('Model 1  r=1',    dict(r=1.0,   beta=1.501,   vartheta=5e-2)),
    ('Model 2  r=0.5',  dict(r=0.5,   beta=3.0918,  vartheta=5.9323e-3)),
    ('Model 3  r=-0.5', dict(r=-0.5,  beta=2.0165,  vartheta=1.2583e-3)),
    ('Model 4  r=-1',   dict(r=-1.0,  beta=1.0017,  vartheta=1.2308e-4)),
]

results_pn = {}
for name, hp in models:
    print(f'\n=== {name} ===')
    z_map, theta_map, opt = prior_normalizing_MAP(
        F, b_noisy, sigma_sq,
        r=hp['r'], beta=hp['beta'], vartheta=hp['vartheta'],
    )
    x_map = L_inv @ z_map
    rmse  = np.sqrt(np.mean((x_map - x_true) ** 2))
    print(f'  RMSE = {rmse:.4f}')
    results_pn[name] = dict(z=z_map, theta=theta_map, x=x_map, rmse=rmse)

# =============================================================================
# 4. Figures: one subplot per SBL model, truth + Tikhonov + MAP overlay
# =============================================================================
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (name, res), color in zip(axes.ravel(), results_pn.items(), colors):
    ax.step(t, x_true,  where='post', color='k',    lw=2,   alpha=0.6,
            label='Truth')
    ax.step(t, x_tikh,  where='post', color='gray', lw=1,   ls='--',
            label=f'Tikhonov GCV (RMSE={rmse_tikh:.3f})')
    ax.step(t, res['x'], where='post', color=color,  lw=1.5,
            label=f'Prior-norm MAP (RMSE={res["rmse"]:.3f})')
    ax.set(title=name, xlabel='$t$', xlim=(0, 1))
    ax.legend(fontsize=8)

plt.suptitle(
    '1D Deblurring (TRIPS-Py Deblurring1D)\n'
    'Tikhonov GCV vs Prior-Normalizing MAP',
    fontsize=13
)
plt.tight_layout()
_out_dir = Path(__file__).resolve().parent / 'figures'
_out_dir.mkdir(parents=True, exist_ok=True)
_out = _out_dir / 'results_deblurring_1d.png'
plt.savefig(_out, dpi=150, bbox_inches='tight')
print(f'\nPlot saved to {_out}')
# Headless / CI: skip plt.show() when there is no interactive canvas.
if matplotlib.get_backend().lower() not in ('agg', 'template'):
    plt.show()

print('\n┌─────────────────────────┬──────────┐')
print('│ Method                  │   RMSE   │')
print('├─────────────────────────┼──────────┤')
print(f'│ Tikhonov + GCV          │ {rmse_tikh:.4f}   │')
for name, res in results_pn.items():
    label = f'Prior-norm  {name}'
    print(f'│ {label:<23} │ {res["rmse"]:.4f}   │')
print('└─────────────────────────┴──────────┘')
