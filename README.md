# Heston-Methods-Comparison

This project compares three methods for pricing European call options under the Heston model: the Carr–Madan FFT approach, Monte Carlo simulation using the Quadratic Exponential (QE) scheme, and a neural-network-based approximation.
The datasets are synthetic and generated via Latin Hypercube Sampling (LHT), ensuring efficient coverage of the model’s parameter space. This choice reflects the purpose of the project, which is to explore the behavior, strengths, and limitations of the different pricing approaches rather than to declare a definitive “winner”.

The three techniques are evaluated in terms of execution time and pricing accuracy, using the FFT solution as the reference benchmark.
The results show that the neural network is by far the fastest method (around four times faster than FFT), at the cost of a higher mean error and larger error variance.
Monte Carlo, while significantly slower, produces prices that are very close to the Carr–Madan benchmark and exhibits greater stability in its error distribution.

Moreover, the three approaches each exhibit a distinct strength. The Carr–Madan FFT method is extremely fast and accurate, but only when the model and payoff admit the analytical structure it relies on, it's "elegant" but definitely not elastic. Monte Carlo, in contrast, is the most general, flexible and robust technique: it works for any payoff and model specification, though at the cost of substantially higher computational time. Neural networks offer unmatched speed once trained, making them ideal for real-time or high-frequency applications, but their reliability is limited to the domain covered during training and they provide no strict numerical guarantees outside it.
Naturally, all of these methods remain considerably slower than Black–Scholes, which benefits from a closed-form solution.
