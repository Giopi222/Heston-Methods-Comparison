def heston_integral(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call", q=0.0,
                 int_upper=150.0):
    
    i = 1j
    a = kappa * theta
    x0 = np.log(S)

    def char_func(phi, Pnum):
        # little trap
        u = 0.5 if Pnum == 1 else -0.5 # P_1 e P_2
        b = kappa - rho * sigma_v if Pnum == 1 else kappa

        # d and g 
        d = np.sqrt((rho*sigma_v*i*phi - b)**2 - (sigma_v**2)*(2*u*i*phi - phi**2))
        gp = (b - rho*sigma_v*i*phi + d) / (b - rho*sigma_v*i*phi - d)

        # avoid |gp| ~ 1
        exp_dT = np.exp(d * T)
        one_minus_gp = 1.0 - gp
        one_minus_gp_exp = 1.0 - gp * exp_dT

        eps = 1e-14
        one_minus_gp = np.where(np.abs(one_minus_gp) < eps, eps, one_minus_gp)
        one_minus_gp_exp = np.where(np.abs(one_minus_gp_exp) < eps, eps, one_minus_gp_exp)

        C = (r - q) * i * phi * T + (a / (sigma_v**2)) * (
            (b - rho*sigma_v*i*phi + d) * T - 2.0 * np.log(one_minus_gp_exp / one_minus_gp)
        )
        D = ((b - rho*sigma_v*i*phi + d) / (sigma_v**2)) * (
            (1.0 - exp_dT) / one_minus_gp_exp
        )
        return np.exp(C + D * v0 + i * phi * x0)

    def Pj(j):
        def integrand(phi):
            return np.real(np.exp(-i*phi*np.log(K)) * char_func(phi, j) / (i*phi))
        val, _ = quad(integrand, 0.0, int_upper, limit=500, epsabs=1e-9, epsrel=1e-7)
        return 0.5 + val/np.pi

    P1 = Pj(1)
    P2 = Pj(2)

    call = S*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2
    if option_type.lower() == "call":
        return float(call)
    else:
        return float(call - S*np.exp(-q*T) + K*np.exp(-r*T))