import numpy as np
import pandas as pd
from scipy.stats import qmc 

def sample_heston_lhs(
    n_samples: int,
    seed: int = 42,
    moneyness_bounds=(0.6, 1.4),
    T_bounds=(0.09, 1.40),
    r_bounds=(-0.01, 0.10),
    S0: int = 105,
    q: int = 0
) -> pd.DataFrame:
    """
    Generate n_samples plausible Heston parameters via Latin Hypercube Sampling (LHS),
    and also sample option strikes through moneyness m = K / S0.

    Constraints (as implemented by bounds below):
      - v0     in [0.01, 0.09]
      - kappa  in [0.50, 5.00] # not sampled --> m
      - theta  in [0.01, 0.09]
      - sigma_v in [0.10, 1.00]
      - rho    in [-0.95, -0.10]
      - T      in [0.09, 1.40]
      - r      in [-0.01, 0.10]
      - moneyness m in [m_low, m_high]
      - Feller: 2*kappa*theta >= sigma_v^2

    Returns
    -------
    pd.DataFrame
        Columns: ['S0','K','m','T','r','q','v0','kappa','theta','sigma_v','rho'].
    """
    # bounds
    m_low, m_high = moneyness_bounds
    T_low, T_high = T_bounds
    r_low, r_high = r_bounds

    # order:              v0  kappa theta sigma_v rho   m      T      r
    l_bounds = np.array([0.01, 0.5, 0.01, 0.10, -0.95, m_low, T_low, r_low], dtype=float)
    u_bounds = np.array([0.09, 5.0, 0.09, 1.00, -0.10, m_high, T_high, r_high], dtype=float)

    collected = []
    attempt = 0

    # Repeat in batches until we have enough valid samples
    while len(collected) < n_samples and attempt < 20:
        # Oversample to compensate for filtering
        batch_size = int(np.ceil((n_samples - len(collected)) * 2.0))
        sampler = qmc.LatinHypercube(d=8, seed=seed + attempt)
        U = sampler.random(batch_size)
        X = qmc.scale(U, l_bounds, u_bounds)  # shape [batch_size, 8]

        v0, kappa, theta, sigma_v, rho, m, T_samp, r_samp = (
            X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
        )

        # Feller condition to ensure variance stays strictly positive
        feller_ok = (2.0 * kappa * theta) >= (sigma_v ** 2)

        X_ok = X[feller_ok]

        # Append valid rows until reaching n_samples
        for row in X_ok:
            collected.append(row)
            if len(collected) >= n_samples:
                break

        attempt += 1

    if len(collected) < n_samples:
        raise RuntimeError(
            "Could not generate enough samples satisfying constraints. "
            "Relax bounds slightly or increase attempts."
        )

    params = np.vstack(collected)[:n_samples]
    df = pd.DataFrame(params, columns=["v0", "kappa", "theta", "sigma_v", "rho", "m", "T", "r"])

 
    df["S0"] = S0
    df["K"]  = (df["m"] * S0).astype(float)
    df["q"]  = q
   
    df = df[["S0","K","m","T","r","q","v0","kappa","theta","sigma_v","rho"]]

    return df