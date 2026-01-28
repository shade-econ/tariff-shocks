"""Nonlinear small open economy model"""

import numpy as np
from scipy import optimize, interpolate, integrate
from types import SimpleNamespace

from baseline.baseline_linear import get_CE_NFA

"""Core short-run (beta -> 1) model without extensions"""

def solve_shortrun_inner(tau, tau_r, tau_X, ichg, alpha, eta, gamma, sigma, E_LR=1, P_LR=1, C_LR=None):
    # parallels model_linear.py, but nonlinear and in levels
    # around initial steady state with Y = 1, X = M = alpha, C = N = 1-alpha
    # allow for distinct long-run forward-looking values for transition-to-permanent-shock case
    C_LR = C_LR if C_LR is not None else 1-alpha
    E = E_LR/ichg
    PF = (1+tau)*E
    P = ces_prices(PF, 1, alpha, eta)
    X = alpha * (P * (1+tau_r) * (1+tau_X) / E)**(-gamma)
    C = (P * ichg / P_LR)**(-sigma) * C_LR
    Y = C + X
    M = alpha * (PF / P)**(-eta) * Y
    N = invert_ces(M, Y, 1-alpha, eta)
    assert np.isclose(ces_quantities(N, M, 1-alpha, eta), Y), "production function not satisfied"
    TB = P*(1+tau_X)/E*X - M
    W = (utility(C, sigma) - utility(1-alpha, sigma))/(1-alpha)**(-1/sigma) + get_CE_NFA(alpha, eta, gamma) * TB
    GDPn = P * Y - E * M
    assert np.isclose(GDPn, N + tau*E*M), "GDP identity not satisfied"
    return SimpleNamespace(E=E, PF=PF, P=P, X=X, C=C, Y=Y, M=M, N=N,
                        TB=TB, W=W, ichg=ichg, tau=tau, tau_r=tau_r, tau_X=tau_X, GDPn=GDPn)


def solve_shortrun(tau, alpha, eta, gamma, sigma, retaliation=False, natural=False, export=False):
    tau_r = tau*retaliation
    tau_X = tau*export
    tau = tau*(1-export)

    if not natural:
        return solve_shortrun_inner(tau, tau_r, tau_X, 1, alpha, eta, gamma, sigma)
    else:
        result = optimize.root_scalar(
            lambda ichg: solve_shortrun_inner(tau, tau_r, tau_X, ichg, alpha, eta, gamma, sigma).N - (1-alpha),
            x0=1, method='brentq', xtol=1E-15, bracket=(0.2, 5),
        )
        assert result.converged, 'could not solve for natural rate ichg'
        return solve_shortrun_inner(tau, tau_r, tau_X, result.root, alpha, eta, gamma, sigma)


"""Core long-run model"""

def solve_longrun_inner(tau, tau_r, tau_X, E, alpha, eta, gamma):
    N, GDP = 1-alpha, 1-alpha
    PF = (1+tau)*E
    P = ces_prices(PF, 1, alpha, eta)
    X = alpha * (P * (1+tau_r) * (1+tau_X) / E)**(-gamma)
    Y = N / (1-alpha) / P**eta
    C = Y - X
    M = alpha * Y * (PF / P)**(-eta)
    assert np.isclose(ces_quantities(N, M, 1-alpha, eta), Y), "production function not satisfied"
    GDPn = P * Y - E * M
    assert np.isclose(GDPn, N + tau*E*M), "GDP identity not satisfied"
    TB = P*(1+tau_X)/E*X - M
    return SimpleNamespace(E=E, PF=PF, P=P, X=X, C=C, Y=Y, M=M, N=N,
                        GDP=GDP, TB=TB, tau=tau, tau_r=tau_r, tau_X=tau_X, GDPn=GDPn)


def solve_longrun(tau, alpha, eta, gamma, retaliation=False, export=False):
    tau_r = tau*retaliation
    tau_X = tau*export
    tau = tau*(1-export)

    result = optimize.root_scalar(
            lambda E: solve_longrun_inner(tau, tau_r, tau_X, E, alpha, eta, gamma).TB,
            x0=0, method='brentq', xtol=1E-15, bracket=(0.4, 2.5))
    assert result.converged, 'could not solve for long-run exchange rate dlogE'
    return solve_longrun_inner(tau, tau_r, tau_X, result.root, alpha, eta, gamma)


"""Decomposition of welfare effects"""
def welfare_decomposition(max_tau, alpha, eta, gamma, sigma, retaliation=False, n_tau=100):
    taus = np.linspace(0, max_tau, n_tau)

    # calculate welfare in standard and retaliation cases 
    total = np.array([solve_shortrun(tau, alpha, eta, gamma, sigma, retaliation=retaliation).W for tau in taus])
    results_stab = [solve_shortrun(tau, alpha, eta, gamma, sigma, retaliation=retaliation, natural=True) for tau in taus]
    stab = np.array([r.W for r in results_stab])

    # normalize by steady-state consumption
    total /= (1-alpha)
    stab /= (1-alpha)

    # express in gaps
    gap = total - stab
    terms_of_trade = terms_of_trade_correction(taus, results_stab, get_CE_NFA(alpha, eta, gamma)) / (1-alpha)
    distortion = stab - terms_of_trade
    return taus, (total, gap, terms_of_trade, distortion)


def terms_of_trade_correction(taus, results, CE_NFA):
    """Given equilibrium for different taus, compute "terms of trade" correction:
    effect of varying export price"""
    # get functions for export quantity and change in international price
    PX_func = interpolate.CubicSpline(taus, [r.P/r.E for r in results]) # price of exports abroad
    X_func = interpolate.CubicSpline(taus, [r.X for r in results])
    PX_der = PX_func.derivative()

    # cumulative benefit of change in international price at date 0 is integral of change * quantity
    TOT_0 = np.array([integrate.quad(lambda tau: PX_der(tau) * X_func(tau), 0, taup)[0] for taup in taus])

    # also change in export price at later dates induced by NFA, corresponds to difference
    # between consumption-scale welfare effect of NFA and '1'
    TOT_TB = (CE_NFA - 1) * np.array([r.TB for r in results])
    return TOT_0 + TOT_TB


def chained_GDP(taus, results):
    """Obtain chained GDP from results for various taus"""
    # need enough taus that cubic splines are accurate
    # key formula: dlogGDP = P*Y/GDPn*dlogY - E*M/GDPn*dlogM
    # or cancelling factors, dlogGDP = (P*dY - E*dM)/GDPn
    assert taus[0] == 0

    # get derivatives of Y, M, and N at each point
    dYs = interpolate.CubicSpline(taus, [r.Y for r in results]).derivative()(taus)
    dMs = interpolate.CubicSpline(taus, [r.M for r in results]).derivative()(taus)
    dNs = interpolate.CubicSpline(taus, [r.N for r in results]).derivative()(taus)
    
    # compute logGDP'(tau) at each point
    logGDP_ders = np.array([(r.P * Yd - r.E * Md) / r.GDPn for r, Yd, Md in zip(results, dYs, dMs)])

    # alternative formula: dlogGDP = (w*dN + tau*E*dM)/GDPn (note w=1)
    logGDP_ders_alt = np.array([(dN + r.tau * r.E * dM) / r.GDPn for r, dN, dM in zip(results, dNs, dMs)])
    assert np.allclose(logGDP_ders, logGDP_ders_alt, atol=5E-6), f"two formulas for dlogGDP do not match"

    # integrate to get chained GDP, starting from GDP at tau=0
    logGDPs = interpolate.CubicSpline(taus, logGDP_ders).antiderivative()(taus)
    GDPs = np.exp(logGDPs) * results[0].GDPn
    return GDPs


def add_chained_GDP(results):
    """adds chained GDP to each result in results list"""
    taus = np.array([r.tau for r in results])
    GDPs = chained_GDP(taus, results)
    for r, gdp in zip(results, GDPs):
        r.GDP = gdp


"""Utilities"""

def ces_prices(p_x, p_y, alpha, eta):
    if eta == 1:
        return p_x**alpha * p_y**(1-alpha)
    else:
        return (alpha*p_x**(1-eta) + (1-alpha)*p_y**(1-eta))**(1/(1-eta))

def ces_quantities(x, y, alpha, eta):
    if eta == 1:
        return x**alpha * y**(1-alpha) / (alpha**alpha * (1-alpha)**(1-alpha))
    else:
        return (alpha**(1/eta)*x**(1-1/eta) + (1-alpha)**(1/eta)*y**(1-1/eta))**(eta/(eta-1))
    
def invert_ces(y, z, alpha, eta):
    """For CES production function with shares alpha and 1-alpha, where second input is y
    and output is z, solve to figure out what first input x must be."""
    if eta == 1:
        return (z * alpha**alpha * (1-alpha)**(1-alpha) / y**(1-alpha))**(1/alpha)
    else:
        return ((z**(1-1/eta) - (1-alpha)**(1/eta)*y**(1-1/eta))/alpha**(1/eta))**(eta / (eta-1))
    
def utility(c, sigma):
    if sigma == 1:
        return np.log(c)
    else:
        return (c**(1-1/sigma) - 1)/(1-1/sigma)