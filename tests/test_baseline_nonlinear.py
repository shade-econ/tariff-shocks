import numpy as np
from baseline.baseline_nonlinear import (solve_shortrun as solve_shortrun_nonlinear,
                                solve_longrun as solve_longrun_nonlinear,
                                welfare_decomposition)
from baseline.baseline_linear import (solve_shortrun as solve_shortrun_linear,
                                solve_longrun as solve_longrun_linear)
from baseline import baseline_formulas as bf
from test_baseline_linear import random_parameters


def numerical_differentiation_consistency(f_nonlinear, f_linear, *args, **kwargs):
    """numerically differentiate f_nonlinear given some args (first arg size of shock),
    see if agrees with f_linear"""
    h = 1E-4
    r_ss = f_nonlinear(0, *args, **kwargs)
    r_up = f_nonlinear(h, *args, **kwargs)
    r_dn = f_nonlinear(-h, *args, **kwargs)
    r_lin = f_linear(1, *args, **kwargs)

    # first do comparison for standard variables
    ks = ['E', 'PF', 'P', 'X', 'C', 'Y', 'M', 'N']
    for k in ks:
        dr = (getattr(r_up, k) - getattr(r_dn, k)) / (2 * h)
        assert np.isclose(dr / getattr(r_ss, k), getattr(r_lin, f'dlog{k}'), 
                atol=1E-4), f"numerical derivative of {k} does not match linear model"

    # now three special cases where comparison slightly trickier 
    # ihat not scaled, has different names, only in short-run
    if hasattr(r_up, 'ihat'):
        assert np.isclose((r_up.ichg - r_dn.ichg) / (2 * h), r_lin.ihat,
                atol=1E-4), "numerical derivative of ichg does not match linear model"
        
    # welfare scaled by C, has different names, only in short-run
    if hasattr(r_up, 'W'):
        assert np.isclose((r_up.W - r_dn.W) / (2 * h) / r_ss.C, r_lin.dW,
                atol=1E-4), "numerical derivative of W does not match linear model"
    
    # dTB_Tr is scaled by trade X=M, not by steady state TB = 0
    assert np.isclose((r_up.TB - r_dn.TB) / (2 * h) / r_ss.X, r_lin.dTB_Tr,
                atol=1E-4), "numerical derivative of TB does not match linear model"


def test_shortrun_consistency():
    *params, _ = random_parameters()
    fs = [solve_shortrun_nonlinear, solve_shortrun_linear]

    # try for each combination of retaliation/export tax and natural rate
    numerical_differentiation_consistency(*fs, *params)
    numerical_differentiation_consistency(*fs, *params, retaliation=True)
    numerical_differentiation_consistency(*fs, *params, export=True)
    numerical_differentiation_consistency(*fs, *params, retaliation=True, natural=True)
    numerical_differentiation_consistency(*fs, *params, export=True, natural=True)


def test_longrun_consistency():
    *params, _, _ = random_parameters()
    fs = [solve_longrun_nonlinear, solve_longrun_linear]

    # try for each combination of retaliation/export tax
    numerical_differentiation_consistency(*fs, *params)
    numerical_differentiation_consistency(*fs, *params, retaliation=True)
    numerical_differentiation_consistency(*fs, *params, export=True)


def test_proposition_11_firstorder():
    alpha, eta, gamma, sigma, _ = random_parameters()

    # non-retaliation case
    taus, (_, gap, tot, _) = welfare_decomposition(0.01, alpha, eta, gamma, sigma, n_tau=50)
    _, tot_formula, gap_formula = bf.Welfare(alpha, eta, gamma, sigma)
    assert np.isclose(gap[1]/taus[1], gap_formula, atol=1E-3)
    assert np.isclose(tot[1]/taus[1], tot_formula, atol=5E-3)

    # retaliation case
    taus, (_, gap, tot, _) = welfare_decomposition(0.01, alpha, eta, gamma, sigma, n_tau=50, retaliation=True)
    _, tot_formula, gap_formula = bf.Welfare(alpha, eta, gamma, sigma, retaliation=True)
    assert np.isclose(gap[1]/taus[1], gap_formula, atol=5E-3)
    assert np.isclose(tot[1]/taus[1], tot_formula, atol=5E-3)


def test_proposition_11_secondorder():
    # one-sided second-order numerical differentiation (combined with splines, etc) too inaccurate
    # so for some random parameters will fail; just test for our fixed standard calibration here
    alpha, eta, gamma, sigma = 1/9, 1.15, 1.5, 1.89
    
    # non-retaliation case
    taus, (_, _, _, distortion) = welfare_decomposition(0.1, alpha, eta, gamma, sigma, n_tau=50)
    dM_dtau = solve_shortrun_linear(1, alpha, eta, gamma, sigma).dlogM
    distortion_coeff = 1/2*alpha/(1-alpha)*dM_dtau
    assert np.isclose(distortion[1]/taus[1]**2, distortion_coeff, rtol=0.03)

    # retaliation case
    taus, (_, _, _, distortion) = welfare_decomposition(0.1, alpha, eta, gamma, sigma, n_tau=50, retaliation=True)
    dM_dtau = solve_shortrun_linear(1, alpha, eta, gamma, sigma, retaliation=True).dlogM
    distortion_coeff = 1/2*alpha/(1-alpha)*dM_dtau
    assert np.isclose(distortion[1]/taus[1]**2, distortion_coeff, rtol=0.07)
