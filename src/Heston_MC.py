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
    n = paths // 2 if antithetic else paths

    S = np.full(n, S0, dtype=np.float64)
    v = np.full(n, v0, dtype=np.float64)

    for _ in range(steps):
        # casual driver 
        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        U  = rng.random(n)

        # momenti condizionali per QE
        m  = theta + (v - theta) * ed
        s2 = (v * sigma_v**2 * ed * (1.0 - ed) / kappa
              + theta * sigma_v**2 * (1.0 - ed)**2 / (2.0 * kappa))
        # evita divisioni patologiche
        m  = np.maximum(m, 1e-14)
        psi = s2 / (m**2)

        # soglia
        psi_c = 1.5

        # CASE 1: noncentral chi-square approx
        idx1 = psi <= psi_c
        b2 = 2.0/psi[idx1] - 1.0 + np.sqrt(2.0/psi[idx1]) * np.sqrt(2.0/psi[idx1] - 1.0)
        a  = m[idx1] / (1.0 + b2)
        Zv = Z1[idx1]
        v_next_1 = a * (np.sqrt(b2) + Zv)**2

        # CASE 2: exponential mixture
        idx2 = ~idx1
        p = (psi[idx2] - 1.0) / (psi[idx2] + 1.0)
        p = np.clip(p, 0.0, 1.0 - 1e-14)
        beta = (1.0 - p) / m[idx2]
        U2 = U[idx2]
        v_next_2 = np.where(U2 <= p, 0.0, -np.log((1.0 - p) / (1.0 - U2)) / beta)

        v_next = np.empty_like(v)
        v_next[idx1] = v_next_1
        v_next[idx2] = v_next_2
        v_next = np.maximum(v_next, 0.0)

        # media del var per step
        v_bar = 0.5 * (v + v_next)

        # incremento log-S con correlazione
        Zs = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2
        S *= np.exp((r - q) * dt - 0.5 * v_bar * dt + np.sqrt(np.maximum(v_bar, 0.0) * dt) * Zs)

        v = v_next

    disc = np.exp(-r * T)
    payoff = np.maximum(S - K, 0.0) if option_type.lower() == "call" else np.maximum(K - S, 0.0)

    if antithetic:
        # traiettorie antitetiche
        S_a = np.full(n, S0, dtype=np.float64)
        v_a = np.full(n, v0, dtype=np.float64)
        rng_a = np.random.default_rng(seed)
        for _ in range(steps):
            Z1 = rng_a.standard_normal(n)
            Z2 = rng_a.standard_normal(n)
            U  = rng_a.random(n)

            # usa antitetici
            Z1a, Z2a, Ua = -Z1, -Z2, 1.0 - U

            m  = theta + (v_a - theta) * ed
            s2 = (v_a * sigma_v**2 * ed * (1.0 - ed) / kappa
                  + theta * sigma_v**2 * (1.0 - ed)**2 / (2.0 * kappa))
            m  = np.maximum(m, 1e-14)
            psi = s2 / (m**2)
            psi_c = 1.5

            idx1 = psi <= psi_c
            b2 = 2.0/psi[idx1] - 1.0 + np.sqrt(2.0/psi[idx1]) * np.sqrt(2.0/psi[idx1] - 1.0)
            a  = m[idx1] / (1.0 + b2)
            v_next_1 = a * (np.sqrt(b2) + Z1a[idx1])**2

            idx2 = ~idx1
            p = (psi[idx2] - 1.0) / (psi[idx2] + 1.0)
            p = np.clip(p, 0.0, 1.0 - 1e-14)
            beta = (1.0 - p) / m[idx2]
            v_next_2 = np.where(Ua[idx2] <= p, 0.0, -np.log((1.0 - p) / (1.0 - Ua[idx2])) / beta)

            v_next = np.empty_like(v_a)
            v_next[idx1] = v_next_1
            v_next[idx2] = v_next_2
            v_next = np.maximum(v_next, 0.0)

            v_bar = 0.5 * (v_a + v_next)
            Zs = rho * Z1a + np.sqrt(1.0 - rho**2) * Z2a
            S_a *= np.exp((r - q) * dt - 0.5 * v_bar * dt + np.sqrt(np.maximum(v_bar, 0.0) * dt) * Zs)

            v_a = v_next

        payoff_a = np.maximum(S_a - K, 0.0) if option_type.lower() == "call" else np.maximum(K - S_a, 0.0)
        pay = 0.5 * (payoff + payoff_a)
        m_pay = np.mean(pay)
        std_pay = np.std(pay, ddof=1)
        n_eff = 2 * n  # antitetici contano come 2N
    else:
        m_pay = np.mean(payoff)
        std_pay = np.std(payoff, ddof=1)
        n_eff = n

    price = disc * m_pay
    stderr = disc * std_pay / np.sqrt(n_eff)
    return price, stderr