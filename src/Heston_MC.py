import numpy as np

def heston_mc_qe_price(
    S0, K, T, r, q,
    v0, kappa, theta, sigma_v, rho,
    steps=200, paths=100_000, antithetic=True, seed=None,
    option_type="call"
):

    rng = np.random.default_rng(seed)
    dt = T / steps
    ed = np.exp(-kappa * dt)
    
    # Adjust for antithetic variates
    n = paths // 2 if antithetic else paths
    
    # Initialize arrays
    S = np.full(n, S0, dtype=np.float64)
    v = np.full(n, v0, dtype=np.float64)
    
    if antithetic:
        S_a = np.full(n, S0, dtype=np.float64)
        v_a = np.full(n, v0, dtype=np.float64)
    
    # Time stepping
    for _ in range(steps):
        # Random variates
        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        U = rng.random(n)
        
        # === Normal paths ===
        v_next = _qe_step(v, Z1, U, theta, kappa, sigma_v, ed)
        S = _stock_step(S, v, v_next, Z1, Z2, r, q, rho, dt)
        v = v_next
        
        # === Antithetic paths ===
        if antithetic:
            v_next_a = _qe_step(v_a, -Z1, 1.0 - U, theta, kappa, sigma_v, ed)
            S_a = _stock_step(S_a, v_a, v_next_a, -Z1, -Z2, r, q, rho, dt)
            v_a = v_next_a
    
    # Compute payoffs
    disc = np.exp(-r * T)
    if option_type.lower() == "call":
        payoff = np.maximum(S - K, 0.0)
        if antithetic:
            payoff_a = np.maximum(S_a - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)
        if antithetic:
            payoff_a = np.maximum(K - S_a, 0.0)
    
    # Combine antithetic paths
    if antithetic:
        combined_payoff = 0.5 * (payoff + payoff_a)
        m_pay = np.mean(combined_payoff)
        std_pay = np.std(combined_payoff, ddof=1)
        n_eff = 2 * n
    else:
        m_pay = np.mean(payoff)
        std_pay = np.std(payoff, ddof=1)
        n_eff = n
    
    price = disc * m_pay
    stderr = disc * std_pay / np.sqrt(n_eff)
    
    return price, stderr


def _qe_step(v_curr, Z1, U, theta, kappa, sigma_v, ed):
    """
    QE scheme for variance evolution (Andersen 2008).
    """
    # Conditional moments
    m = theta + (v_curr - theta) * ed
    s2 = (v_curr * sigma_v**2 * ed * (1.0 - ed) / kappa +
          theta * sigma_v**2 * (1.0 - ed)**2 / (2.0 * kappa))
    
    # Numerical safeguard
    m = np.maximum(m, 1e-14)
    psi = s2 / (m**2)
    
    # Threshold
    psi_c = 1.5
    
    # CASE 1: Noncentral chi-square approximation (low psi)
    idx1 = psi <= psi_c
    sqrt_term = np.sqrt(2.0 / psi[idx1])
    b2 = 2.0 / psi[idx1] - 1.0 + sqrt_term * np.sqrt(sqrt_term**2 - 1.0)
    a = m[idx1] / (1.0 + b2)
    v_next_1 = a * (np.sqrt(b2) + Z1[idx1])**2
    
    # CASE 2: Exponential mixture (high psi)
    idx2 = ~idx1
    p = (psi[idx2] - 1.0) / (psi[idx2] + 1.0)
    p = np.clip(p, 0.0, 1.0 - 1e-14)
    beta = (1.0 - p) / m[idx2]
    v_next_2 = np.where(U[idx2] <= p, 0.0, 
                        -np.log((1.0 - p) / (1.0 - U[idx2])) / beta)
    
    # Combine cases
    v_next = np.empty_like(v_curr)
    v_next[idx1] = v_next_1
    v_next[idx2] = v_next_2
    v_next = np.maximum(v_next, 0.0)
    
    return v_next


def _stock_step(S_curr, v_curr, v_next, Z1, Z2, r, q, rho, dt):
    """
    Stock price evolution with correlated Brownian motion.
    """
    # Average variance over step
    v_bar = 0.5 * (v_curr + v_next)
    
    # Correlated random variate
    Zs = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2
    
    # Log-normal step
    S_next = S_curr * np.exp(
        (r - q) * dt - 0.5 * v_bar * dt + 
        np.sqrt(np.maximum(v_bar, 0.0) * dt) * Zs
    )
    
    return S_next