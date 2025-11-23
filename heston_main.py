import numpy as np
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import pi, sqrt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import least_squares
from scipy.integrate import quad


df = pd.read_csv('/Users/gio/Library/Mobile Documents/com~apple~CloudDocs/INFO/Progetti/Heston_model/df_call_AAPL.csv')


def black_scholes(S, K, T, r, sigma, q, option_type="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    
def greeks(S, K, T, r, sigma, q, option_type="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)

    Delta = np.exp(-q*T) * norm.cdf(d1)
    Gamma = np.exp(-q*T) * pdf / (S * sigma * np.sqrt(T))
    Vega = S * np.exp(-q*T) * pdf * np.sqrt(T)

    return Delta, Gamma, Vega

def heston_fft(S, K, T, r, v0, kappa, theta, sigma_v, rho,
                     option_type="call", q=0.0,
                     alpha=1.5, N=4096, eta=0.05):
    i = 1j

    # Caratteristica risk-neutral di ln(S_T)
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

    # Griglia in frequenza
    u = np.arange(N) * eta

    # Pesi trapezoidali
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    w *= eta

    # psi(u) di Carr-Madan
    uj = u - i*(alpha + 1.0)
    numerator = np.exp(-r*T) * phi(uj)
    denominator = (alpha**2 + alpha - u**2) + i*(2*alpha + 1.0)*u
    denominator = np.where(np.abs(denominator) < 1e-14, 1e-14, denominator)  # guardia numerica
    psi = numerator / denominator

    # Spaziatura in log-strike
    lam = 2.0*np.pi / (N * eta)                 # passo in k
    k_min = np.log(S) - (N * lam) / 2.0         # shift (inizio griglia)
    k_grid = k_min + np.arange(N) * lam         # griglia di log-strike

    # Input FFT 
    fft_input = np.exp(-1j * k_min * u) * psi * w
    fft_vals = np.fft.fft(fft_input)
    fft_real = np.real(fft_vals)

    # Prezzi delle call su tutta la griglia k
    C_k = (np.exp(-alpha * k_grid) / np.pi) * fft_real

    # Interpola al log-strike desiderato
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
    


# OTTIMIZZAZIONE HESTON
def residuals_price(p, df, option_type="call"):
    kappa, theta, sigma_v, rho, v0 = p
    res = []
    for row in df.itertuples(index=False):
        S, K, T, r, q, price_mkt = row.S, row.strike, row.T_years, row.risk_free, row.q_t, row.lastPrice
        price_mod = heston_fft(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type, q)
        price_mod = max(float(price_mod), 0.0)
        v = row.Vega
        w = v if np.isfinite(v) and v>1e-10 else 1e-10
        res.append((price_mod - price_mkt) * np.sqrt(w))
    # penalità Feller dura
    feller_violation = max(0.0, sigma_v**2 - 2.0*kappa*theta)
    res.append(1e4 * feller_violation)
    return np.array(res, float)

def calibrate_heston_price(df, **kwargs):
    return least_squares(residuals_price,
                         kwargs.get("x0",(1.0,0.04,0.30,-0.6,0.04)),
                         bounds=kwargs.get("bounds",((0.01,0.01,0.05,-0.95,0.01),
                                                     (5.00,0.20,1.00, 0.00,0.50))),
                         args=(df, kwargs.get("option_type","call")),
                         loss=kwargs.get("loss","soft_l1"),
                         f_scale=kwargs.get("f_scale",0.01),
                         max_nfev=kwargs.get("max_nfev",150),
                         xtol=1e-8, ftol=1e-8, gtol=1e-8)

res_price = calibrate_heston_price(df)
params = res_price.x


