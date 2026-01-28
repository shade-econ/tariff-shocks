import numpy as np
import pytest

from baseline.baseline_linear import solve_shortrun, solve_longrun
from baseline import baseline_formulas as bf


def random_parameters():
    # eta = 1 and sigma = 1 are special cases for nonlinear, so sometimes test them
    alpha = np.random.uniform(0.1, 0.9)
    eta = np.random.uniform(0.1, 4) if np.random.rand() > 0.5 else 1
    gamma = np.random.uniform(0.1, 4)
    sigma = np.random.uniform(0.1, 4) if np.random.rand() > 0.5 else 1
    dtau = np.random.uniform(0.1, 1)
    return alpha, eta, gamma, sigma, dtau


def test_proposition_1():
    alpha, eta, gamma, _, dtau = random_parameters()
    r = solve_longrun(dtau, alpha, eta, gamma)
    assert np.isclose(r.dlogX, -gamma*eta/((1-alpha)*(gamma-1) + eta)*dtau)
    assert np.isclose(r.dlogM, -eta*(gamma-1)/((1-alpha)*(gamma-1) + eta)*dtau)
    assert np.isclose(r.dTB_Tr, 0)
    assert np.isclose(r.dlogE, -(eta - alpha*(gamma-1))/((1-alpha)*(gamma-1) + eta)*dtau)
    assert np.isclose(r.dlogE - r.dlogP, -eta/((1-alpha)*(gamma-1) + eta)*dtau)


def test_lerner_longrun():
    alpha, eta, gamma, _, dtau = random_parameters()
    r1 = solve_longrun(dtau, alpha, eta, gamma)
    r2 = solve_longrun(dtau, alpha, eta, gamma, export=True)

    assert np.isclose(r1.dlogX, r2.dlogX)
    assert np.isclose(r1.dlogM, r2.dlogM)
    assert np.isclose(r1.dlogP, r2.dlogP)
    assert np.isclose(r1.dTB_Tr, r2.dTB_Tr)
    assert np.isclose(r1.dlogE, r2.dlogE - dtau)


def test_proposition_2():
    alpha, eta, gamma, sigma, dtau = random_parameters()
    r = solve_shortrun(dtau, alpha, eta, gamma, sigma)

    assert np.isclose(r.dlogGDP, bf.GDP_unilateral(alpha, eta, gamma, sigma)*dtau)
    assert np.isclose(r.dTB_Tr, bf.TB_unilateral(alpha, eta, gamma, sigma)*dtau)

    # proposition also refers to X, M, CPI
    assert np.isclose(r.dlogX, -alpha*gamma*dtau)
    assert np.isclose(r.dlogM, -((1-alpha)*eta + alpha*((1-alpha)*sigma + alpha*gamma))*dtau)
    assert np.isclose(r.dlogP, alpha*dtau)


def test_proposition_3():
    alpha, eta, gamma, sigma, dtau = random_parameters()
    r = solve_shortrun(dtau, alpha, eta, gamma, sigma, natural=True)

    assert np.isclose(r.ihat, bf.rate_unilateral_natural(alpha, eta, gamma, sigma)*dtau)
    assert np.isclose(r.dTB_Tr, bf.TB_unilateral_natural(alpha, eta, gamma, sigma)*dtau)


def test_proposition_4():
    alpha, eta, gamma, sigma, dtau = random_parameters()
    r = solve_shortrun(dtau, alpha, eta, gamma, sigma, export=True)

    assert np.isclose(r.dlogGDP, -alpha*gamma*dtau)
    assert np.isclose(r.dlogX, -gamma*dtau)
    assert np.isclose(r.dlogM, -alpha*gamma*dtau)
    assert np.isclose(r.dTB_Tr, -((1-alpha)*gamma - 1)*dtau)

    r = solve_shortrun(dtau, alpha, eta, gamma, sigma, export=True, natural=True)

    assert np.isclose(r.ihat, -alpha/(1-alpha)*gamma/((1-alpha)*sigma + alpha*gamma + alpha/(1-alpha)*eta) * dtau)


def test_proposition_5():
    alpha, eta, gamma, sigma, dtau = random_parameters()
    r = solve_shortrun(dtau, alpha, eta, gamma, sigma, retaliation=True)

    assert np.isclose(r.dlogGDP, bf.GDP_retaliation(alpha, eta, gamma, sigma)*dtau)
    assert np.isclose(r.dTB_Tr, bf.TB_retaliation(alpha, eta, gamma, sigma)*dtau)

# note: propositions 6 through 9 are simple extensions that we develop analytically but don't solve in this code

@pytest.mark.parametrize("kw", [dict(), dict(retaliation=True),
                                dict(natural=True), dict(natural=True, retaliation=True)])
def test_proposition_10(kw):
    alpha, eta, gamma, sigma, dtau = random_parameters()
    r = solve_shortrun(dtau, alpha, eta, gamma, sigma, **kw)

    assert np.isclose(r.dW, bf.Welfare(alpha, eta, gamma, sigma, **kw)[0]*dtau)
