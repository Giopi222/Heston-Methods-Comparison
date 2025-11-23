import numpy as np


def heston_price_fft(S, K, T, r, v0, kappa, theta, sigma_v, rho,
                     option_type="call", q=0.0,
                     alpha=1.5, N=4096, eta=0.05):
    i = 1j

    # risk-neutral char funct of ln(S_T)
    def phi(u):
        a = kappa * theta
        b_h = kappa
        d = np.sqrt((rho*sigma_v*i*u - b_h)**2 + (sigma_v**2)*(i*u + u**2))
        g = (b_h - rho*sigma_v*i*u + d) / (b_h - rho*sigma_v*i*u - d)

        C = (r - q)*i*u*T + (a/(sigma_v**2)) * (
            (b_h - rho*sigma_v*i*u + d)*T - 2*np.log((1 - g*np.exp(d*T))/(1 - g))
        )
        D = ((b_h - rho*sigma_v*i*u + d)/(sigma_v**2)) * (
            (1 - np.exp(d*T))/(1 - g*np.exp(d*T))
        )
        return np.exp(C + D*v0 + i*u*np.log(S))

    # grid
    u = np.arange(N) * eta

    # weights
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    w *= eta

    # psi(u)
    uj = u - i*(alpha + 1.0)
    numerator = np.exp(-r*T) * phi(uj)
    denominator = (alpha**2 + alpha - u**2) + i*(2*alpha + 1.0)*u
    denominator = np.where(np.abs(denominator) < 1e-14, 1e-14, denominator)  # guardia numerica
    psi = numerator / denominator

    lam = 2.0*np.pi / (N * eta)                
    k_min = np.log(S) - (N * lam) / 2.0         
    k_grid = k_min + np.arange(N) * lam       

    # Input FFT 
    fft_input = np.exp(-1j * k_min * u) * psi * w
    fft_vals = np.fft.fft(fft_input)
    fft_real = np.real(fft_vals)

    # Call prices on the entire grid k
    C_k = (np.exp(-alpha * k_grid) / np.pi) * fft_real

    # Interpolation
    k_target = np.log(K)
    if k_target <= k_grid[0]:
        price_call = C_k[0]
    elif k_target >= k_grid[-1]:
        price_call = C_k[-1]
    else:
        idx = np.searchsorted(k_grid, k_target) - 1
        t = (k_target - k_grid[idx]) / (k_grid[idx+1] - k_grid[idx])
        price_call = C_k[idx] * (1 - t) + C_k[idx+1] * t

    if option_type.lower() == "call":
        return float(price_call)
    else:
        # put by parity
        return float(price_call - S*np.exp(-q*T) + K*np.exp(-r*T))



