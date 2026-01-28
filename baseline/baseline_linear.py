"""Linearized small open economy model"""

import numpy as np
from types import SimpleNamespace

"""Core short-run (beta -> 1) model without extensions"""

def solve_shortrun_inner(dtau, dtau_r, dtau_X, ihat, alpha, eta, gamma, sigma):
    # list corresponding (usually nonlinear) equation numbers from paper
    # these numbers are from first NBER working paper version; recheck consistency
    # here we assume nominal wage rigidity, only valid if ultimately dlogN <= 0

    dlogE = -ihat                                       # equation (7), given unchanged future E
    dlogPF = dtau + dlogE                               # equation (8)
    dlogP = alpha * dlogPF                              # equation (5), given stable nominal wage
    dlogX = -gamma*(dlogP + dtau_r + dtau_X - dlogE)    # equation (10) with retaliatory and export tax added
    dlogC = -sigma*(dlogP + ihat)                       # Euler equation (unlabeled), given unchanged future C
    dlogY = (1-alpha)*dlogC + alpha*dlogX               # equation (13)
    dlogM = -eta*(dlogPF - dlogP) + dlogY               # equation (6-right)
    dlogN = eta*dlogP + dlogY                           # equation (6-left)
    dTB_Tr = (dlogP-dlogE+dlogX+dtau_X) - dlogM         # equation (12) (dividing by SS trade X=M, adding export tax)
    
    dlogGDP = (dlogY - alpha*dlogM) / (1-alpha)         # implied by equation (4) for dlogN, also GDP definition
    assert np.isclose(dlogN, dlogGDP), "dlogN should be dlogGDP"

    dW = dlogC + get_CE_NFA(alpha, eta, gamma)*dTB_Tr*alpha/(1-alpha)

    return SimpleNamespace(dlogE=dlogE, dlogPF=dlogPF, dlogP=dlogP, dlogX=dlogX,
                dlogC=dlogC, dlogY=dlogY, dlogM=dlogM, dlogN=dlogN, dlogGDP=dlogGDP,
                dTB_Tr=dTB_Tr, dW=dW, dtau=dtau, dtau_r=dtau_r, dtau_X=dtau_X, ihat=ihat)


def solve_shortrun(dtau, alpha, eta, gamma, sigma, retaliation=False, natural=False, export=False):
    # "retaliation = True" means dtau_r = dtau
    dtau_r = dtau*retaliation

    # "export = True" means dtau_X = dtau and no import tax
    dtau_X = dtau*export
    dtau = dtau*(1-export)

    if not natural:
        return solve_shortrun_inner(dtau, dtau_r, dtau_X, 0, alpha, eta, gamma, sigma)
    else:
        # solve for natural rate ihat
        ihat = solve_scalar_linear(
            lambda ihat: solve_shortrun_inner(dtau, dtau_r, dtau_X, ihat, alpha, eta, gamma, sigma).dlogN)
        return solve_shortrun_inner(dtau, dtau_r, dtau_X, ihat, alpha, eta, gamma, sigma)


def get_CE_NFA(alpha, eta, gamma):
    """What is the consumption-scale value of marginal NFA, taking into account ToT effects so >1?"""
    # note: this is linear, but because of the beta -> 1 assumption, can use same for nonlinear model,
    # since effects after date 0 will always be infinitesimal
    return ((1-alpha)*gamma + eta) / ((1-alpha)*(gamma-1) + eta)


"""Core long-run model"""

def solve_longrun_inner(dtau, dtau_r, dtau_X, dlogE, alpha, eta, gamma):
    dlogN, dlogGDP = 0, 0                               # assume long-run natural allocation
    dlogPF = dtau + dlogE                               # equation (8)
    dlogP = alpha * dlogPF                              # equation (5), given stable nominal wage
    dlogX = -gamma*(dlogP + dtau_r + dtau_X - dlogE)    # equation (10) with retaliatory and export tax added
    dlogY = dlogN - eta*dlogP                           # equation (6-left)
    dlogC = (dlogY - alpha*dlogX) / (1-alpha)           # equation (13)
    dlogM = -eta*(dlogPF - dlogP) + dlogY               # equation (6-right)
    dTB_Tr = (dlogP-dlogE+dlogX+dtau_X) - dlogM         # equation (12) (dividing by SS trade X=M, adding export tax)
    
    return SimpleNamespace(dlogE=dlogE, dlogPF=dlogPF, dlogP=dlogP, dlogX=dlogX,
                dlogC=dlogC, dlogY=dlogY, dlogM=dlogM, dlogN=dlogN, dlogGDP=dlogGDP,
                dTB_Tr=dTB_Tr, dtau=dtau, dtau_r=dtau_r, dtau_X=dtau_X)


def solve_longrun(dtau, alpha, eta, gamma, retaliation=False, export=False):
    dtau_r = dtau*retaliation
    dtau_X = dtau*export
    dtau = dtau*(1-export)

    dlogE = solve_scalar_linear(
            lambda dlogE: solve_longrun_inner(dtau, dtau_r, dtau_X, dlogE, alpha, eta, gamma).dTB_Tr)
    return solve_longrun_inner(dtau, dtau_r, dtau_X, dlogE, alpha, eta, gamma)


def solve_scalar_linear(f):
    """Solve a scalar linear equation f(x) = 0 for x"""
    x = -f(0)/(f(1)-f(0))
    assert np.isclose(f(x), 0), "failed to correctly solve linear equation, maybe not linear?"
    return x