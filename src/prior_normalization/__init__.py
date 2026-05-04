"""
Prior normalization transforms for SBL (sparse Bayesian learning).

Python port of the Julia PriorNormalization package from:
  https://github.com/jglaubitz/paper-2025-SBL-priorNormalization

Provides:
  - transform_CalvettiSomersalo2024: (v, ω) -> (z, θ) reparameterization
  - make_phi: build Φ(τ) for the prior-normalizing map
  - priorNormalizing_KR_inv, priorNormalizing_KR_inv_tu, etc.: inverse KR map components
"""

from .transforms import (
    transform_CalvettiSomersalo2024,
    make_phi,
    make_phi_chebyshev,
    priorNormalizing_KR_inv,
    priorNormalizing_KR_inv_tτ,
    priorNormalizing_KR_inv_tτ_aux,
    priorNormalizing_KR_inv_tu,
)

__all__ = [
    "transform_CalvettiSomersalo2024",
    "make_phi",
    "make_phi_chebyshev",
    "priorNormalizing_KR_inv",
    "priorNormalizing_KR_inv_tτ",
    "priorNormalizing_KR_inv_tτ_aux",
    "priorNormalizing_KR_inv_tu",
]
