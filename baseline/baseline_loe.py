"""Linear large open economy model (extension)"""


import numpy as np
from types import SimpleNamespace


def solve_loe_inner(dtau, dtau_star, ihat, ihat_star, lamb, alpha, eta, gamma, sigma):
    prices = solve_prices(dtau, dtau_star, ihat, ihat_star, lamb, alpha)
    quantities = solve_quantities(prices, lamb, alpha, eta, gamma, sigma)
    return SimpleNamespace(**prices.__dict__, **quantities.__dict__)


def solve_loe(dtau, lamb, alpha, eta, gamma, sigma, retaliation=False, natural=False):
    dtau_star = dtau*retaliation

    if not natural:
        return solve_loe_inner(dtau, dtau_star, 0, 0, lamb, alpha, eta, gamma, sigma)
    else:
        def rate_effect(ihat, ihat_star):
            result = solve_loe_inner(dtau, dtau_star, ihat, ihat_star, lamb, alpha, eta, gamma, sigma)
            return result.dlogGDP, result.dlogGDP_star, result

        ihat, ihat_star = solve_vector_linear(lambda x, y: rate_effect(x, y)[:2], 2)
        return rate_effect(ihat, ihat_star)[2]


def solve_prices(dtau, dtau_star, ihat, ihat_star, lamb, alpha):
    def prices(dlogP, dlogPF_star):
        """Given dlogP and dlogPF_star, compute all prices"""
        # refer to equations (a)-(g) for prices in appendix
        dlogE = ihat_star - ihat                        # (a)
        dlogPF = dlogE + dtau + dlogPF_star             # (b)
        dlogPbar = lamb*dlogP + (1-lamb)*dlogPF         # (c)
        dlogP_gap = alpha*dlogPbar - dlogP              # (d) (if gap = 0)
        dlogP_star = -dlogE + dtau_star + dlogP         # (e)
        dlogPM = lamb*dlogP_star + (1-lamb)*dlogPF_star # (f)
        dlogPF_star_gap = alpha*dlogPM - dlogPF_star    # (g) (if gap = 0)
        return dlogP_gap, dlogPF_star_gap, SimpleNamespace(
            dlogE=dlogE, dlogPF=dlogPF, dlogPbar=dlogPbar,
            dlogP=dlogP, dlogP_star=dlogP_star,
            dlogPM=dlogPM, dlogPF_star=dlogPF_star,
            dtau=dtau, dtau_star=dtau_star, ihat=ihat, ihat_star=ihat_star)
    
    dlogP, dlogPF_star = solve_vector_linear(lambda x, y: prices(x, y)[:2], 2)
    return prices(dlogP, dlogPF_star)[2]


def solve_quantities(p, lamb, alpha, eta, gamma, sigma):
    def quantities(dlogX, dlogY, dlogYF_star):
        """Given dlogX, dlogY, dlogY_star, compute all quantities"""
        # equations (a)-(f) for home quantities in appendix
        dlogC = -sigma * (p.ihat + p.dlogP)                # (a)
        dlogN = dlogY + eta * p.dlogP                      # (c)
        dlogMbar = dlogN - eta * p.dlogPbar                # (d)
        dlogM = dlogMbar + gamma*(p.dlogPbar - p.dlogPF)   # (e) (if gap = 0)
        dlogQ = dlogMbar + gamma*(p.dlogPbar - p.dlogP)    # (f)

        dlogY_gap = ((1-lamb)*alpha*dlogX + (1-alpha)*dlogC 
                                + alpha*lamb*dlogQ - dlogY)    # (b) (if gap = 0)
        
        # equations (a)-(f) for foreign quantities in appendix
        dlogC_star = -sigma * (p.ihat_star + p.dlogPF_star)     # (a)
        dlogN_star = dlogYF_star + eta * p.dlogPF_star          # (c)  
        dlogMi = dlogYF_star + eta*(p.dlogPF_star - p.dlogPM)   # (d)
        dlogmij = dlogMi + gamma*(p.dlogPM - p.dlogPF_star)     # (e)
        
        dlogX_gap = dlogMi + gamma*(p.dlogPM - p.dlogP_star) - dlogX  # (f) (if gap = 0)
        dlogYF_star_gap = ((1-lamb)*alpha*dlogmij + (1-alpha)*dlogC_star
                            + lamb*alpha*dlogM - dlogYF_star)         # (b) (if gap = 0)
        
        # also report trade balance (% of pre-shock trade, final equation in appendix)
        dTB_Tr = p.dlogP - p.dlogE + dlogX - dlogM - p.dlogPF_star

        return dlogX_gap, dlogY_gap, dlogYF_star_gap, SimpleNamespace(
            dlogC=dlogC, dlogGDP=dlogN, dlogMbar=dlogMbar,
            dlogQ=dlogQ, dlogM=dlogM, dlogY=dlogY,
            dlogC_star=dlogC_star, dlogGDP_star=dlogN_star,
            dlogMi=dlogMi, dlogmij=dlogmij, dlogX=dlogX,
            dlogYF_star=dlogYF_star, dTB_Tr=dTB_Tr)
    

    dlogX, dlogY, dlogYF_star = solve_vector_linear(
        lambda x, y, z: quantities(x, y, z)[:3], 3)
    return quantities(dlogX, dlogY, dlogYF_star)[3]


def solve_vector_linear(f, n):
    """Solve linear f(x)=0 for x if f maps tuple of n args to tuple of n outputs"""
    z = np.zeros(n)
    y0 = np.array(f(*z))

    J = np.zeros((n, n))
    for i in range(n):
        z[i] = 1
        J[:, i] = np.array(f(*z)) - y0
        z[i] = 0
    x = np.linalg.solve(J, -y0)
    assert np.allclose(np.array(f(*x)), 0), "failed to correctly solve linear equation, maybe not linear?"
    return x
