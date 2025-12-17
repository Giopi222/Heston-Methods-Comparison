# THE HESTON MODEL


The Heston model can be viewed as a nonlinear generalisation of the Ornstein--Uhlenbeck process, where the variance $v_t$ follows a Cox-Ingersoll-Ross (CIR) diffusion:

$$
dv_t = \kappa (\theta - v_t)\, dt + \eta \sqrt{v_t}\, dW_t^{(2)}.
$$

Its structure resembles the mean-reverting OU process, except that the volatility coefficient $\eta$ is not constant (as $\sigma$ would be in an OU model), but proportional to $\sqrt{v_t}$.  
If we informally approximate $\sqrt{v_t} \approx 1$, the dynamics reduce to an OU process with additive noise:
$$
dv_t \approx \kappa (\theta - v_t)\, dt + \eta\, dW_t.
$$

The square-root term ensures $v_t \ge 0$, although it does not always prevent the process from hitting zero. For this reason, one often imposes the Feller condition:
$$
2\kappa \theta \;\ge\; \eta^2
$$

The full Heston model is given by:
$$
\begin{cases}
dS_t = \sqrt{v_t}\, S_t\, dW_t^{(1)} + (r-q) S_t\, dt, \\[6pt]
dv_t = \eta \sqrt{v_t}\, dW_t^{(2)} + \kappa(\theta - v_t)\, dt, \\[6pt]
d\langle W^{(1)}, W^{(2)}\rangle_t = \rho\, dt.
\end{cases}
$$

The first equation describes the asset-price dynamics;  
The second governs the variance (a CIR process);  
The third specifies the instantaneous correlation between the Brownian motions $W_t^{(1)}$ and $W_t^{(2)}$.  
When $\rho < 0$, the price tends to fall when volatility rises---the well-known leverage effect.

A crucial difference from the Black--Scholes model is that, with stochastic volatility, $\ln S_t$ is no longer normally distributed. Conditional on a given path of $v_t$ it is Gaussian, but marginally it becomes a mixture of Gaussians, capable of producing fat tails and skewness observed in financial markets. The magnitude of this effect is controlled by $\eta$, known as the *volatility of volatility*.

Although the probability density of $\ln S_t$ is not known in closed form, its characteristic function is. This enables Fourier-based pricing techniques (FFT, COS), as well as Monte Carlo simulation. 

By the martingale approach, we arrive at the following multi-dimensional Heston option pricing PDE:
$$\frac{\partial V}{\partial t}
+\frac{1}{2}vS^2 \frac{\partial^2 V}{\partial S^2}
+\rho\eta v S\frac{\partial^2 V}{\partial S \partial v}
+\frac{1}{2}\eta^2 v\frac{\partial^2 V}{\partial v^2}
+r S \frac{\partial V}{\partial S}
+\kappa(\theta - v)\frac{\partial V}{\partial v}
-r V
= 0$$

with terminal condition
$$
V(S,v,T) = (S-K)^+.
$$


### The rho parameter
Rho defines the skew:

- $\rho < 0$  → negative skew (equity markets, leverage effect)
- $\rho ≈ 0$ → nearly symmetric smile
- $\rho > 0$ → positive skew (rare in equities)



### The eta parameter
Eta defines the smile:

- small $\eta$  → flat smile (close to Black–Scholes)
- moderate $\eta$  → visible smile curvature
- large $\eta$  → strong smile and heavy tails

So, the parameters $\kappa$ and $\theta$ control the speed and level of mean reversion of volatility, while $\eta$ governs the magnitude of volatility fluctuations and $\rho$ determines the skew of the implied volatility surface.

### Greeks under the Heston Model

The benefits of the model become clear when looking at Greeks, where stochastic volatility introduces richer behaviour:


- **Delta**: 
    In Black-Scholes it depends only on price and constant volatility.  
    In Heston it is influenced by the dynamics of $v_t$, producing a more flexible Delta that reacts better to strong market skew.

- **Vega**: In Black--Scholes it is a monotonic function of $S$ and time to maturity.  
    In Heston, volatility is not an exogenous input but a stochastic mean-reverting variable, so Vega reflects fluctuations in the variance itself (*volatility-of-volatility*).

- **Gamma**: Black-Scholes provides simple closed-form expressions, but they can be rigid.  
    Under Heston, sensitivities arise from a more complex pricing structure and better capture the curvature of smile and skew.


Despite its strengths, the model is not perfect: it can reproduce smile and skew effects, but not uniformly across maturities, and it often struggles with very steep short-term skews.