#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D deblurring: TRIPS-Py forward model + Tikhonov/GCV + prior-normalizing MAP
=============================================================================

Problem setup
-------------
We observe a vectorized image ``b`` of length ``N = nx * ny``:

    b = A x + e,

where ``A`` is a 2D convolution (PSF) from TRIPS-Py's ``Deblurring2D``, ``x`` is the
unknown image (flattened column-major or as returned by TRIPS-Py), and ``e`` is
noise.

Why ``L = I`` here (no first-difference whitening)
---------------------------------------------------
The built-in phantoms (``satellite``, ``hubble``, …) are **naturally sparse or
structured in the pixel domain** (dark background, compact support). Unlike the
1D piecewise-constant demo, we do **not** apply a derivative-based ``L``: we set
``L = L^{-1} = I`` so ``z = x`` and ``F = A``. The same prior-normalizing MAP
machinery then runs directly on pixel coefficients.

Working directory quirk (``chdir`` into TRIPS-Py ``demos/``)
------------------------------------------------------------
``Deblurring2D.gen_true`` loads MATLAB archives via **relative paths** such as
``./data/image_data/<phantom>.mat``. Those files ship under ``<trips-py>/demos/``.
We ``os.chdir`` into that folder for the TRIPS-Py calls, then **restore** the
previous cwd in a ``finally`` block so the rest of your session is unaffected.

Quick mode (``SBL_EXAMPLE_QUICK``)
---------------------------------
The full default grid is ``24×24`` (``N = 576``) with **four** SBL models. Each MAP
solve optimizes ``2N`` variables and can take a very long time. Setting
``SBL_EXAMPLE_QUICK=1`` reduces ``nx, ny``, runs **one** model, lowers the L-BFGS-B
iteration cap, and uses a smaller figure layout for fast smoke tests / CI.

TRIPS-Py / GCV convention
---------------------------
``generalized_crossvalidation(Q_A, R_A, R_L, b_vec)`` — here ``R_A = F = A`` and
``b_vec`` is an ``(N, 1)`` column (TRIPS-Py convention).
"""

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))
from bootstrap import ensure_prior_normalization_on_path, ensure_trips_py_on_path

ensure_prior_normalization_on_path()
_trips_root = ensure_trips_py_on_path()

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime

if not hasattr(np, 'int0'):
    np.int0 = np.intp  # type: ignore[attr-defined]

# Directory where TRIPS-Py ships ``data/image_data/*.mat`` for phantoms.
_demos_dir = _trips_root / 'demos'
if not (_demos_dir / 'data' / 'image_data').is_dir():
    raise RuntimeError(f'Expected image data at {_demos_dir}/data/image_data')

_prev_cwd = os.getcwd()
os.chdir(_demos_dir)
try:
    # Import pylops before optional stubs (same rationale as the 1D script).
    import pylops  # noqa: F401

    # Stub optional heavy deps if missing; try real PIL first (matplotlib may need it).
    from unittest.mock import MagicMock
    import importlib
    for _mod in ('astra', 'PIL', 'PIL.Image', 'resizeimage', 'resizeimage.resizeimage', 'requests'):
        if _mod in sys.modules:
            continue
        try:
            importlib.import_module(_mod)
        except Exception:
            sys.modules[_mod] = MagicMock()

    from trips.test_problems.Deblurring2D import Deblurring2D
    from trips.utilities.reg_param.gcv import generalized_crossvalidation

    from prior_normalization import (
        make_phi_chebyshev,
        priorNormalizing_KR_inv,
        priorNormalizing_KR_inv_tu,
    )

    print('Environment ready.')

    # -----------------------------------------------------------------------
    # Optional fast path: smaller problem + fewer models + lower maxiter
    # -----------------------------------------------------------------------
    _quick = os.environ.get('SBL_EXAMPLE_QUICK', '').lower() in ('1', 'true', 'yes')
    if _quick:
        print('SBL_EXAMPLE_QUICK: using a smaller image and one SBL model (for CI / dry runs).')

    np.random.seed(42)

    nx = ny = 12 if _quick else 24
    psf_dim    = (3, 3)       # PSF support size in pixels (per axis)
    psf_spread = (1.0, 1.0)   # Gaussian PSF std (per axis)
    noise_level = 0.01       # relative noise level for TRIPS-Py's add_noise
    image_name  = 'satellite'  # phantom key understood by Deblurring2D.gen_true
    _map_maxiter = 2_000 if _quick else 10_000

    N = nx * ny
    Deblur = Deblurring2D(CommitCrime=True)

    # Forward operator: applies blur to flattened length-N vectors.
    A_op = Deblur.forward_Op(psf_dim, psf_spread, nx, ny)

    x_true_im = Deblur.gen_true(image_name)
    x_true    = x_true_im.reshape(-1)

    b_true = Deblur.gen_data(x_true).reshape(-1)
    b_noisy_im, delta = Deblur.add_noise(b_true.reshape(-1, 1), opt='Gaussian',
                                         noise_level=noise_level)
    b_noisy = b_noisy_im.reshape(-1)
    b_vec   = b_noisy.reshape(-1, 1)  # column vector for GCV helper

    sigma_sq = (delta ** 2) / N

    print(f'nx={nx} ny={ny} | psf_dim={psf_dim} | psf_spread={psf_spread}')
    print(f'noise_level={noise_level} | delta={delta:.4f} | sigma_sq={sigma_sq:.2e}')

    # Dense ``A`` for small experiments (same pattern as 1D script).
    A_dense = np.zeros((N, N))
    for j in range(N):
        ej = np.zeros(N)
        ej[j] = 1.0
        A_dense[:, j] = np.asarray(A_op.matvec(ej)).reshape(-1)

    # Pixel-domain whitening: F = A @ I = A.
    L     = np.eye(N)
    L_inv = np.eye(N)
    F     = A_dense @ L_inv
    z_true = L @ x_true

    assert np.allclose(F @ z_true, A_dense @ x_true, atol=1e-10)
    print(f'F shape: {F.shape}')

    lam_tikh = generalized_crossvalidation(
        np.eye(N),
        F,
        np.eye(N),
        b_vec,
    )

    z_tikh    = np.linalg.solve(F.T @ F + lam_tikh * np.eye(N), F.T @ b_noisy)
    x_tikh    = L_inv @ z_tikh
    rmse_tikh = np.sqrt(np.mean((x_tikh - x_true) ** 2))
    print(f'\nTikhonov GCV:  lambda={lam_tikh:.4e}  RMSE={rmse_tikh:.4f}')


    def prior_normalizing_MAP(F, b, sigma_sq, r, beta, vartheta,
                              n_cheb=64, B=5.0, jac_eps=1e-5,
                              maxiter=None, verbose=True):
        """
        MAP in ``z``-space (here ``z`` equals pixels) via prior-normalizing KR map.

        See the 1D script's docstring for the meaning of ``xi``, ``Phi``, and the
        ``neg_log_posterior`` structure; the mathematics is identical, only ``F``
        and ``N`` change.

        Parameters
        ----------
        maxiter : int or None
            L-BFGS-B ``maxiter``. If ``None``, uses the outer scope ``_map_maxiter``
            (smaller when ``SBL_EXAMPLE_QUICK`` is set).
        """
        if maxiter is None:
            maxiter = _map_maxiter
        N   = F.shape[1]
        Phi = make_phi_chebyshev(beta, r, n_points=n_cheb)

        def neg_log_posterior(xi):
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
            options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-6),
        )

        tau_MAP = result.x[0::2]
        u_MAP   = result.x[1::2]
        z_MAP, theta_MAP = priorNormalizing_KR_inv(
            u_MAP, tau_MAP, r, beta, vartheta, Phi, B=B
        )

        if verbose:
            print(f'iters={result.nit} | converged={result.success}')

        return z_MAP, theta_MAP, result


    _model_defs = [
        ('Model 1  r=1',    dict(r=1.0,   beta=1.501,   vartheta=5e-2)),
        ('Model 2  r=0.5',  dict(r=0.5,   beta=3.0918,  vartheta=5.9323e-3)),
        ('Model 3  r=-0.5', dict(r=-0.5,  beta=2.0165,  vartheta=1.2583e-3)),
        ('Model 4  r=-1',   dict(r=-1.0,  beta=1.0017,  vartheta=1.2308e-4)),
    ]
    models = _model_defs[:1] if _quick else _model_defs

    results_pn = {}
    for name, hp in models:
        print(f'\n=== {name} ===')
        z_map, theta_map, opt = prior_normalizing_MAP(
            F, b_noisy, sigma_sq,
            r=hp['r'], beta=hp['beta'], vartheta=hp['vartheta'],
            maxiter=_map_maxiter,
        )
        x_map = L_inv @ z_map
        rmse  = np.sqrt(np.mean((x_map - x_true) ** 2))
        print(f'  RMSE = {rmse:.4f}')
        results_pn[name] = dict(z=z_map, theta=theta_map, x=x_map, rmse=rmse)

    # Shared color scale: phantoms are nonnegative; use truth max for fair comparison.
    vmax = float(x_true.max())
    vmin = 0.0

    plt.set_cmap('inferno')
    if _quick:
        # 2×2 summary: truth, data, Tikhonov, first (only) MAP model
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(x_true.reshape(nx, ny), vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Truth')
        axes[0, 1].imshow(b_noisy.reshape(nx, ny))
        axes[0, 1].set_title('Blurred + noise')
        axes[1, 0].imshow(x_tikh.reshape(nx, ny), vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'Tikhonov GCV\nRMSE={rmse_tikh:.3f}')
        name0, res0 = next(iter(results_pn.items()))
        axes[1, 1].imshow(res0['x'].reshape(nx, ny), vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f'{name0}\nRMSE={res0["rmse"]:.3f}')
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
    else:
        # Full layout: top row = truth, data, Tikhonov, spacer; bottom = four MAP models
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes[0, 0].imshow(x_true.reshape(nx, ny), vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Truth')
        axes[0, 1].imshow(b_noisy.reshape(nx, ny))
        axes[0, 1].set_title('Blurred + noise')
        axes[0, 2].imshow(x_tikh.reshape(nx, ny), vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f'Tikhonov GCV\nRMSE={rmse_tikh:.3f}')
        axes[0, 3].axis('off')

        for ax, (name, res) in zip(axes[1, :], results_pn.items()):
            ax.imshow(res['x'].reshape(nx, ny), vmin=vmin, vmax=vmax)
            ax.set_title(f'{name}\nRMSE={res["rmse"]:.3f}')

        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        '2D Deblurring (TRIPS-Py Deblurring2D)\n'
        'Tikhonov GCV vs Prior-Normalizing MAP',
        fontsize=13,
    )
    plt.tight_layout()
    # Save next to this script (not relative to TRIPS-Py cwd); __file__ stays absolute.
    _out_dir = Path(__file__).resolve().parent / 'figures'
    _out_dir.mkdir(parents=True, exist_ok=True)
    _out = _out_dir / 'results_deblurring_2d.png'
    plt.savefig(_out, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {_out}')
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
finally:
    os.chdir(_prev_cwd)
