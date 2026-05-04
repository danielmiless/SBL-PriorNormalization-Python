"""
Prior-normalizing transport maps and Calvetti–Somersalo 2024 transform.

Mirrors the Julia PriorNormalization package from:
  https://github.com/jglaubitz/paper-2025-SBL-priorNormalization

These maps transform between the original SBL variables (z, θ) and standardized
variables (u, τ) so that the prior becomes standard normal, enabling more
efficient optimization and MCMC.
"""

import numpy as np
from scipy import special
from numpy.polynomial import chebyshev


def transform_CalvettiSomersalo2024(v, omega, r, beta, vartheta):
    """
    Transform (v, ω) -> (z, θ) from Calvetti & Somersalo (2024).

    Used for the alternative reparameterization in the transformed posterior.
    Formula (from Julia source):
      θ = 2^(-1/r) * ϑ * |ω|^(2/r)
      z = v * sqrt(θ)

    Parameters
    ----------
    v, omega : float or array
        Standard-normal-like variables.
    r, beta, vartheta : float
        Generalized gamma hyperprior parameters (power, shape, scale).

    Returns
    -------
    z, theta : same shape as inputs
        Increment and variance variables in the original space.
    """
    theta = (2.0 ** (-1.0 / r)) * vartheta * (np.abs(omega) ** (2.0 / r))
    z = v * np.sqrt(theta)
    return z, theta


def make_phi(beta, r):
    """
    Build Φ(τ): the (1-p)-quantile of Gamma(β, 1) as a function of τ.

    In Julia, gammainvccdf(β, 1, p) returns the inverse complementary CDF of
    Gamma(β, 1), i.e. the value x such that P(X > x) = p, so cdf(x) = 1-p.
    Equivalently, x is the (1-p)-quantile. In Python we use gammaincinv(β, 1-p).

    The argument p(τ) depends on r:
      r > 0:  p(τ) = 0.5*erfc(τ/√2)  =>  1-p = 0.5 + 0.5*erf(τ/√2)
      r ≤ 0:  p(τ) = 0.5 + 0.5*erf(τ/√2)  =>  1-p = 0.5*erfc(τ/√2)

    Returns
    -------
    phi : callable
        Function tau -> gamma_quantile (used in prior-normalizing KR inverse).
    """

    def phi(tau):
        tau = np.asarray(tau)
        if r > 0:
            p = 0.5 * special.erfc(tau / np.sqrt(2.0))
            q = 1.0 - p  # 0.5 + 0.5*erf(tau/sqrt(2))
        else:
            p = 0.5 + 0.5 * special.erf(tau / np.sqrt(2.0))
            q = 1.0 - p  # 0.5*erfc(tau/sqrt(2))
        # Clip to avoid gammaincinv domain errors at 0 or 1
        return special.gammaincinv(beta, np.clip(q, 1e-16, 1 - 1e-16))

    return phi


def make_phi_chebyshev(beta, r, n_points=64):
    """
    Build Φ(τ) as a Chebyshev polynomial approximation on [-5, 5] (matches Julia ApproxFun).

    Smoother and differentiable for optimization; use this for better gradient flow
    when matching Julia's prior-normalized MAP.
    """
    tau_grid = np.linspace(-5.0, 5.0, n_points)
    vals = np.array([_phi_exact(t, beta, r) for t in tau_grid])
    # Chebyshev basis is on [-1, 1]; map tau in [-5, 5] -> x in [-1, 1] via x = tau/5
    x_grid = tau_grid / 5.0
    deg = min(31, n_points - 2)  # avoid poorly conditioned fit
    coef = chebyshev.chebfit(x_grid, vals, deg=deg)
    def phi(tau):
        tau = np.asarray(tau)
        x = np.clip(tau / 5.0, -1.0, 1.0)
        return chebyshev.chebval(x, coef)
    return phi


def _phi_exact(tau, beta, r):
    """Exact Φ(τ) for a scalar tau (used to build Chebyshev fit)."""
    if r > 0:
        p = 0.5 * special.erfc(tau / np.sqrt(2.0))
        q = 1.0 - p
    else:
        p = 0.5 + 0.5 * special.erf(tau / np.sqrt(2.0))
        q = 1.0 - p
    q = np.clip(q, 1e-16, 1 - 1e-16)
    return float(special.gammaincinv(beta, q))


def priorNormalizing_KR_inv_tτ_aux(tau, r, beta, vartheta, Phi):
    """
    First component of inverse KR map (core): τ -> θ.

    Maps standard-normal τ to variance θ via the generalized-gamma quantile:
    gamma_quantile = Φ(τ), then θ = ϑ * gamma_quantile^(1/r).
    """
    gamma_quantile = np.abs(Phi(tau))
    theta = vartheta * (gamma_quantile ** (1.0 / r))
    return theta


def priorNormalizing_KR_inv_tτ(tau, r, beta, vartheta, Phi, B=5.0):
    """
    First component of inverse KR map with linear extension outside [-B, B].

    For |τ| ≤ B we use _aux. For τ < -B or τ > B we linearly extend using
    a finite-difference approximation of the derivative at ±B, matching the
    Julia implementation. Output is clamped to be nonnegative.
    """
    tau = np.asarray(tau)
    scalar = np.isscalar(tau) or tau.ndim == 0
    tau = np.atleast_1d(tau)
    theta = np.empty_like(tau, dtype=float)
    delta = 0.1 * B  # step for FD derivative

    for i, t in enumerate(tau):
        if np.abs(t) <= B:
            theta[i] = priorNormalizing_KR_inv_tτ_aux(t, r, beta, vartheta, Phi)
        elif t < -B:
            # Linear extension from τ = -B
            theta_B = priorNormalizing_KR_inv_tτ_aux(-B, r, beta, vartheta, Phi)
            theta_delta = priorNormalizing_KR_inv_tτ_aux(-B + delta, r, beta, vartheta, Phi)
            theta_deriv = (theta_delta - theta_B) / delta
            theta[i] = theta_B + theta_deriv * (t + B)
        else:
            # Linear extension from τ = B
            theta_B = priorNormalizing_KR_inv_tτ_aux(B, r, beta, vartheta, Phi)
            theta_delta = priorNormalizing_KR_inv_tτ_aux(B - delta, r, beta, vartheta, Phi)
            theta_deriv = (theta_B - theta_delta) / delta
            theta[i] = theta_B + theta_deriv * (t - B)

        if theta[i] <= 0:
            theta[i] = 0.0

    return theta[0] if scalar else theta


def priorNormalizing_KR_inv_tu(u, tau, r, beta, vartheta, Phi, B=5.0):
    """
    Second component of inverse KR map: (u, τ) -> z.

    Given θ = _tτ(τ), we have z = u * sqrt(θ) (conditional inverse map).
    """
    theta = priorNormalizing_KR_inv_tτ(tau, r, beta, vartheta, Phi, B=B)
    return u * np.sqrt(theta)


def priorNormalizing_KR_inv(u, tau, r, beta, vartheta, Phi, B=5.0):
    """
    Full inverse of the prior-normalizing Knothe–Rosenblatt map: (u, τ) -> (z, θ).

    Pulls back from the (u, τ) space (standard normal prior) to (z, θ).
    """
    theta = priorNormalizing_KR_inv_tτ(tau, r, beta, vartheta, Phi, B=B)
    z = u * np.sqrt(theta)
    return z, theta
