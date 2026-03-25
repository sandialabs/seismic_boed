import os

import numpy as np


DEFAULT_PRIOR_MODE = "student_t_iso_centered"
DEFAULT_PRIOR_SAMPLES = 50
DEFAULT_PRIOR_DF = 5.0
DEFAULT_PRIOR_DEV_SCALE = 0.25
_FALSE_STRINGS = {"0", "false", "no", "off"}


def normalize_mt(mt6):
    """
    Normalize MT vectors using the repository's Voigt-space convention:

        ||m|| = sqrt(0.5 * sum_i m_i^2)
    """
    mt = np.asarray(mt6, dtype=float)
    if mt.shape[-1] != 6:
        raise ValueError("Moment tensor input must end in 6 Voigt components.")

    norms = np.sqrt(0.5 * np.sum(mt**2, axis=-1, keepdims=True))
    if np.any(norms <= 0.0):
        raise ValueError("Moment tensor norm must be positive.")
    return mt / norms


def isotropic_mt_unit():
    """Return the normalized isotropic MT direction in Voigt order."""
    return normalize_mt(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float))


def mt_marginalization_enabled():
    raw = os.environ.get("SEISMIC_OED_MT_MARGINALIZE", "0").strip().lower()
    return raw not in _FALSE_STRINGS


def get_mt_prior_config():
    mode = os.environ.get("SEISMIC_OED_MT_PRIOR_MODE", DEFAULT_PRIOR_MODE).strip()
    nsamples = max(
        1, int(os.environ.get("SEISMIC_OED_MT_PRIOR_SAMPLES", DEFAULT_PRIOR_SAMPLES))
    )
    df = max(1.0, float(os.environ.get("SEISMIC_OED_MT_PRIOR_DF", DEFAULT_PRIOR_DF)))
    dev_scale = max(
        0.0,
        float(os.environ.get("SEISMIC_OED_MT_PRIOR_DEV_SCALE", DEFAULT_PRIOR_DEV_SCALE)),
    )
    seed_raw = os.environ.get("SEISMIC_OED_MT_PRIOR_SEED")
    seed = None if seed_raw is None or seed_raw == "" else int(seed_raw)
    return {
        "mode": mode,
        "samples": nsamples,
        "df": df,
        "dev_scale": dev_scale,
        "seed": seed,
    }


def sample_mt_prior(
    size,
    mode=DEFAULT_PRIOR_MODE,
    df=DEFAULT_PRIOR_DF,
    dev_scale=DEFAULT_PRIOR_DEV_SCALE,
    seed=None,
    rng=None,
):
    """
    Sample normalized MT directions from the configured prior.

    The v1 prior is a Student-t perturbation around the isotropic direction,
    followed by renormalization onto the fixed-norm MT manifold.
    """
    if mode != DEFAULT_PRIOR_MODE:
        raise ValueError(f"Unsupported MT prior mode: {mode}")

    if rng is None:
        rng = np.random.default_rng(seed)

    base = isotropic_mt_unit()
    perturb = rng.standard_t(df=df, size=(int(size), 6))
    samples = base[None, :] + float(dev_scale) * perturb
    return normalize_mt(samples)
