"""Analytical formulas here are used both for reported numerical results / plots
*and* also for testing in test_baseline_linear.py, which validates propositions.
Latter also validates additional formulas that are not included here"""


"""Proposition 2: unilateral, passive policy"""
def GDP_unilateral(alpha, eta, gamma, sigma):
    return -alpha*((1-alpha)*sigma + alpha*gamma - eta)

def TB_unilateral(alpha, eta, gamma, sigma):
    return alpha + (1-alpha)*(eta + alpha*(sigma-gamma))


"""Proposition 3: unilateral, stabilizing policy"""
def rate_unilateral_natural(alpha, eta, gamma, sigma):
    return (-alpha/(1-alpha)*((1-alpha)*sigma + alpha*gamma - eta) /
                      ((1-alpha)*sigma + alpha*gamma + alpha/(1-alpha)*eta))

def TB_unilateral_natural(alpha, eta, gamma, sigma):
    return (eta / (1-alpha) * ((1-alpha)*sigma + alpha) /
                      ((1-alpha)*sigma + alpha*gamma + alpha/(1-alpha)*eta))
    

"""Proposition 5: retaliation, passive policy"""
def GDP_retaliation(alpha, eta, gamma, sigma):
    return -alpha*((1-alpha)*sigma + alpha*gamma + gamma - eta)

def TB_retaliation(alpha, eta, gamma, sigma):
    return -((1-alpha)*(gamma - eta) + alpha*(1-alpha)*(gamma-sigma) - alpha)


"""Proposition 10: first-order welfare effects"""
def Welfare(alpha, eta, gamma, sigma, natural=False, retaliation=False, noboom=False):
    if natural:
        dlogGDP = 0
    else:
        if retaliation:
            dlogGDP = GDP_retaliation(alpha, eta, gamma, sigma)
        else:
            dlogGDP = GDP_unilateral(alpha, eta, gamma, sigma)
        dlogGDP = min(dlogGDP, 0) if noboom else dlogGDP

    tot_effect = alpha/(1-alpha)* (eta - (1-alpha)*gamma*retaliation) / ((1-alpha)*(gamma-1) + eta)
    gap_effect = (1 - alpha / ((1-alpha)*(gamma-1) + eta))*dlogGDP
    return tot_effect + gap_effect, tot_effect, gap_effect

