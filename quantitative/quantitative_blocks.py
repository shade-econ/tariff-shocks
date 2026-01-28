import sequence_jacobian as sj


""" GOAL: have the following unknowns globally:
 - Output Y
 - wage inflation pi_w
 - intermediate bundle G
 - Nominal interest rate i 
 - Exchange rate E

Ideal order: production
"""

@sj.solved(unknowns={'W': 1.}, targets=['W_cond'])
def nominal_wage(pi_w, W):
    W_cond = W(-1) * (1 + pi_w) - W
    return W_cond

# wage Phillips curve
@sj.simple
def phillips_curve(pi_w, W, P, C_ND, N, sigma, phi, kappa, beta, phi_N):
    labor_wedge = phi_N * C_ND ** (1/sigma) * N ** (1/phi) / (W/P) - 1
    nkpc_res = pi_w * (1 + pi_w) - kappa * labor_wedge - beta * pi_w(+1) * (1 + pi_w(+1))
    return nkpc_res, labor_wedge

# Foreign demand
@sj.simple
def foreign_demand(P, E, gamma, taustar, openness):
    Pstar = P/E * (1 + taustar)
    X = openness * Pstar ** (-gamma)
    return Pstar, X

@sj.simple
def foreign_price(tau, E, PFstar):
    P_M = (1 + tau) * E * PFstar  # P_M = Price of IMPORTS
    return P_M

# Final goods production: Y = CES(N, G) with elasticity eta and shares (1 - alpha/(1-chi)) and alpha/(1-chi)
@sj.simple
def production(Y, W, P_G, alpha_tilde, eta):  # P_G = Price of intermediates
    P = ((1 - alpha_tilde) * W ** (1 - eta) + alpha_tilde * P_G ** (1 - eta)) ** (1/(1-eta))
    # P = P_G ** alpha * W ** (1 - alpha)  # price index
    N = (1 - alpha_tilde) * Y * (W / P) ** (-eta)  # labor demand
    G_prod = alpha_tilde * Y * (P_G / P) ** (-eta)  # demand for intermediate bundle
    pi = P/P(-1) - 1
    return P, N, G_prod, pi

# Importers: G = CES(S, Mtilde) with elasticity upsilon and shares chi and 1-chi
@sj.solved(unknowns={'S': 1., 'QS': 1.}, targets=['inv_cond', 'val_cond'])
def importers(S, QS, G, P_M, ups, chi, eps_S, i):
    # compute how much inventory drawdown (Mtilde) we need, given S(-1)
    if ups == 1:
        drawdown = (chi ** chi * (1-chi) ** (1-chi) * G * (i.ss * S(-1)) ** (-chi)) ** (1/(1-chi))
    else:
        drawdown = ((1-chi)**(-1/ups) * G ** ((ups-1)/ups) - (chi/(1-chi))**(1/ups) * (i.ss * S(-1)) ** ((ups-1)/ups)) ** (ups/(ups-1))

    # compute marginals of G
    marginal_G_drawdown = (1 - chi) ** (1 / ups) * (G / drawdown) ** (1 / ups)
    marginal_G_S = chi ** (1/ups) * (G / S(-1)) ** (1/ups)

    # compute price of intermediates
    P_G = P_M / marginal_G_drawdown

    # adj cost (imported goods)
    adj_cost = 1/eps_S * 1/2 * ((S - S(-1))/S(-1)) ** 2 * S(-1)

    # valuation equation for Q (note adj cost paid in P here!)
    inv_cond = (S - S(-1))/S(-1) - eps_S * (QS(+1) - 1)
    val_cond = P_G/P_M * marginal_G_S - S/S(-1) + 1 - adj_cost + QS(+1) * S / S(-1) - QS * (1 + i(-1)) * P_M(-1)/P_M

    # imports (incl. adj cost)
    M = S - S(-1) + drawdown + adj_cost   

    # profits
    profits = P_G * G - P_M * M

    # inventory to sales ratio
    inv_sales = P_M * S / (P_G * G)

    return val_cond, inv_cond, P_G, drawdown, M, profits, adj_cost, inv_sales

# %% real income
@sj.simple
def real_income(profits, N, W, P, tau, E, PFstar, M):
    tax_revenue = tau * E * PFstar * M
    income = profits/P + N*W/P + tax_revenue/P
    return income, tax_revenue


# %% Static block:
# What is consumption and nominal exchange rate *if trade balance is being closed and consumption stable*?
# Taking as given: W, S, D, A
# Unknowns: M, P, E
@sj.solved(unknowns={'G_term': 1., 'E_term': 1., 'income_term': 1.}, targets=['cond_G', 'cond_goods', 'cond_income'])
def terminal_eqm(W, S, D, A,    E_term, G_term, income_term,    delta, beta, PFstar, chi, ups, eta, alpha_tilde, phi_N,
                 sigma, phi, gamma, openness):

    # prepare
    r_ss = 1/beta - 1
    C_D_term = delta * D
    net_income = income_term - C_D_term

    # Nondurable consumption: keep assets constant
    C_ND_term = r_ss * A(-1) + net_income

    P_M = E_term * PFstar

    if ups == 1:
        drawdown = (chi ** chi * (1-chi) ** (1-chi) * G_term * (r_ss * S(-1)) ** (-chi)) ** (1/(1-chi))
    else:
        drawdown = ((1-chi)**(-1/ups) * G_term ** ((ups-1)/ups) - (chi/(1-chi))**(1/ups) * (r_ss * S(-1)) ** ((ups-1)/ups)) ** (ups/(ups-1))

    # compute marginals of G
    marginal_G_drawdown = (1 - chi) ** (1 / ups) * (G_term / drawdown) ** (1 / ups)

    # compute price of intermediates
    P_G = P_M / marginal_G_drawdown

    # imports (incl. adj cost)
    M = drawdown

    # profits
    profits = P_G * G_term - P_M * M

    # price level
    P = ((1 - alpha_tilde) * W ** (1 - eta) + alpha_tilde * P_G ** (1 - eta)) ** (1 / (1 - eta))

    # Labor supply
    N = ( (W / P) / phi_N * C_ND_term ** (- 1 / sigma) ) ** phi

    # Output
    Y = N / (1 - alpha_tilde) * (W / P) ** (eta)

    # condition for G_term
    cond_G = alpha_tilde * Y * (P_G / P) ** (-eta) - G_term

    # Exports
    Pstar = P / E_term
    X = openness * Pstar ** (-gamma)

    # goods market clearing
    cond_goods = Y - C_ND_term - C_D_term - X

    # income condition
    cond_income = income_term - profits/P - N*W/P

    return cond_G, cond_goods, C_ND_term, cond_income

@sj.simple
def terminal_eqm_lag(E_term, C_ND_term):
    E_term_lag = E_term(-1)
    C_ND_term_lag = C_ND_term(-1)
    return E_term_lag, C_ND_term_lag

@sj.simple
def terminal_eqm_correction(E_term, C_ND_term, E_term_lag, C_ND_term_lag):
    E_corr = E_term - E_term_lag(+1)
    C_ND_corr = C_ND_term - C_ND_term_lag(+1)
    return E_corr, C_ND_corr


#%% household: Euler + budget
@sj.solved(unknowns={'C_ND': 1., 'A': 0., 'D': 1., 'QD': 0.}, targets=['euler', 'budget', 'dur_inv_cond', 'dur_val_cond'])
def household(C_ND, D, A, sigma, beta, income, delta, eps_D, QD, phi_D, i, pi, C_ND_corr2):

    r_ante = i - pi(+1)
    r_ante_last = i(-1) - pi  # this only works because we start with zero assets

    # Euler for nondurables
    euler = beta * (1+r_ante) * (C_ND(+1) + C_ND_corr2)**(-1/sigma) - C_ND**(-1/sigma)

    # Optimality for durables
    dur_inv_cond = (D - D(-1))/D(-1) - delta * eps_D * (QD(+1) - 1)
    dur_adj_cost = 1/(2*eps_D*delta) * ((D - D(-1))/D(-1)) ** 2 * D(-1)
    dur_val_cond = phi_D * (C_ND / D) ** (1/sigma) - D/D(-1) + 1 - delta - dur_adj_cost + D/D(-1) * QD(+1) - (1 + r_ante(-1)) * QD

    # spending on durables
    C_D = D - (1 - delta) * D(-1) + dur_adj_cost

    # total spending
    C = C_ND + C_D

    # budget constraint
    budget = (1 + r_ante_last) * A(-1) + income - A - C

    return euler, budget, dur_inv_cond, dur_val_cond, C, C_D, dur_adj_cost


#%% Continue
# trade balance (in foreign currency)
@sj.simple
def trade_balance(M, X, PFstar, E, P, A):
    TB = P/E * X - PFstar * M
    NFA = A * P/E  # NFA in foreign currency
    return TB, NFA

# GDP
@sj.simple
def compute_gdp(Y, M, S, P_M, P):
    gdp = P.ss * Y - P_M.ss * M + P_M.ss * (S - S(-1))
    gdp_no_inv = P.ss * Y - P_M.ss * M
    gdp_only_inv = P_M.ss * (S - S(-1))
    return gdp

# market clearing
@sj.simple
def mkt_clearing(TB, NFA, C, Y, X, istar, G, G_prod, C_ND_corr, C_ND_corr2, E_corr, E_corr2, N):
    # goods market clearing condition
    goods_mkt = Y - C - X

    # balance of payments
    bop = NFA(-1) * (1 + istar(-1)) + TB - NFA

    # G = G_prod
    G_cond = G - G_prod

    C_ND_corr_cond = C_ND_corr2 - C_ND_corr

    E_corr_cond = E_corr2 - E_corr

    stable_N = N - N.ss

    return bop, goods_mkt, G_cond, C_ND_corr_cond, E_corr_cond, stable_N

@sj.simple
def uip(E, istar, i, P, PFstar, E_corr2):
    uip_cond = (1 + i) - (1 + istar) * (E(+1) + E_corr2)/E  #  + zeta * NFA
    rer = E/P * PFstar
    return uip_cond, rer

#Inertial Taylor rule
@sj.simple
def taylor_inert(i, phi_pi, phi_x, rho_i, pi, pi_w, gdp, labor_wedge):
    # i_res = rho_i * i(-1) + (1-rho_i) * (i.ss + phi_pi * pi) - i  # goods inflation targeting
    i_res = rho_i * i(-1) + (1-rho_i) * (i.ss + phi_pi * pi_w + phi_x * labor_wedge) - i  # wage inflation targeting
    # i_res = rho_i * i(-1) + (1-rho_i) * (i.ss + phi_pi * gdp) - i  # gdp targeting
    return i_res

@sj.simple
def walras(N, W, P, i, pi, A, C, P_M, M, Y, P_G, G_prod):
    income = N * W / P
    r_ante_last = i(-1) - pi
    walras1 = (1 + r_ante_last) * A(-1) + N * W / P - A - C
    walras2 = N*W + P_G * G_prod - P * Y
    walras3 = P_G * G_prod - P_M * M
    return walras1, walras2, walras3


model = sj.create_model([nominal_wage, phillips_curve, foreign_price, production, importers, real_income,
                         foreign_demand, trade_balance, compute_gdp, mkt_clearing, uip, taylor_inert, walras,
                         household, terminal_eqm, terminal_eqm_lag, terminal_eqm_correction],
                        name="Open Economy Model with Inventories and Durables")