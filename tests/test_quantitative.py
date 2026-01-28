import numpy as np

from quantitative.quantitative_steadystate import get_steady_states
from quantitative.quantitative_blocks import model
from baseline.baseline_linear import solve_shortrun

"""
Focus on 1-time shocks in transitory shock limit.
First, check baseline model results against analytical ones, for natural allocation and passive MP
Next, check durables formulas in our main text. 
"""

def random_common_parameters(**kwargs):

    # set tiny interest rate
    r_ss = 1e-6  # quarterly real interest rate

    phi = np.random.uniform(0.1, 4)
    eta = np.random.uniform(0.1, 4)
    gamma = np.random.uniform(0.1, 4)
    eps_D = np.random.uniform(5, 15)
    openness = np.random.uniform(0.05, 0.3)
    sigma = np.random.uniform(0.1, 4)

    # To make comparable to simple model:
    kappa = 0

    common = {
        'beta': 1 / (1 + r_ss),  # Discount factor
        'delta': 0.20,  # Depreciation rate of durables
        'phi': phi,  # Frisch elasticity of labor supply
        'eta': eta,  # Elasticity of import substitution
        'gamma': gamma,  # Elasticity of foreign demand
        'eps_D': eps_D,  # Elasticity of durables demand
        'eps_S': 20,  # Elasticity of inventory demand
        'ups': 1,  # Elasticity of substitution between inventory and drawdown
        'rho_i': 0.0,  # Inertia in Taylor rule
        'phi_pi': 1.5,  # Taylor rule response to wage inflation
        'phi_x': 0.0,  # Taylor rule response to labor wedge
        'kappa': kappa,  # Phillips curve slope,
        'openness': openness,  # Imports / Output
        'delta_S': 0,
    }

    # update
    common.update(kwargs)

    return common, sigma

def test_analytical_results():

    # get common parameters
    common, sigma_basic = random_common_parameters()

    # generate associated steady states (only care about basic one here, so other parameters won't matter)
    ss = get_steady_states(common, sigma_basic=sigma_basic, sigma_dur=1., share_durables=0.10, inv_to_sales=0.33)['basic']


    # %% Fixed exchange rates
    # simulate a one-time tariff shock,
    T = 10
    dtau = 0.1
    taushock = dtau * (np.arange(T)==0)  # 10% tariff shock at time 0

    irf = model.solve_impulse_linear(ss, unknowns=['pi_w', 'Y', 'G', 'C_ND_corr2'],
                                                          targets=['nkpc_res', 'goods_mkt', 'G_cond', 'C_ND_corr_cond'],
                                                          inputs={'tau': taushock})
    irf['E'] = 0 * np.arange(T)
    irf['i'] = 0 * np.arange(T)

    # compare results to analytical formulas from baseline_formulas.py
    r = solve_shortrun(dtau, common['openness'], common['eta'], common['gamma'], sigma_basic)

    # compare results
    variable_mapping = {
        'Y': 'dlogY',
        'P': 'dlogP',
        'X': 'dlogX',
        'M': 'dlogM',
        'N': 'dlogN',
        'E': 'dlogE',
        'C': 'dlogC',
        'P_M': 'dlogPF',
    }

    for var_model, var_analytical in variable_mapping.items():
        model_value = irf[var_model][0] / ss[var_model]
        analytical_value = getattr(r, var_analytical)
        # print(f"  Model: {model_value:.7f}, Analytical: {analytical_value:.7f}")
        assert np.isclose(model_value, analytical_value, rtol=1e-4), f"Mismatch in {var_model}: model {model_value}, analytical {analytical_value}"


    # %% Natural allocation
    # Note: include labor preference shifter here to kill labor supply change (which is not in analytical model)
    irf = model.solve_impulse_linear(ss, unknowns=['i', 'Y', 'G', 'E', 'C_ND_corr2', 'E_corr2', 'phi_N'],
                                     targets=['labor_wedge', 'uip_cond', 'goods_mkt', 'G_cond', 'C_ND_corr_cond', 'E_corr_cond', 'stable_N'],
                                     inputs={'tau': taushock})
    irf['W'] = 0*np.arange(T)

    # compare results to analytical formulas from baseline_formulas.py
    r = solve_shortrun(dtau, common['openness'], common['eta'], common['gamma'], sigma_basic, natural=True)

    # compare results
    for var_model, var_analytical in variable_mapping.items():
        model_value = irf[var_model][0] / ss[var_model]
        analytical_value = getattr(r, var_analytical)
        # print(var_model + f" --- Model: {model_value:.7f}, Analytical: {analytical_value:.7f}")
        assert np.isclose(model_value, analytical_value, rtol=1e-4), f"Mismatch in {var_model} (natural): model {model_value}, analytical {analytical_value}"


def test_durables_formulas():

    # get common parameters, with really low delta
    common, sigma = random_common_parameters(delta=1e-6)

    # generate associated steady states (only care about durables one here, so other parameters won't matter)
    share_durables = np.random.uniform(0.05, 0.3)
    ss = get_steady_states(common, sigma_basic=1., sigma_dur=sigma, share_durables=share_durables, inv_to_sales=0.33)['durables']


    # %% Fixed exchange rates
    # simulate a one-time tariff shock,
    T = 10
    dtau = 0.1
    taushock = dtau * (np.arange(T) == 0)  # 10% tariff shock at time 0

    irf = model.solve_impulse_linear(ss, unknowns=['pi_w', 'Y', 'G', 'C_ND_corr2'],
                                     targets=['nkpc_res', 'goods_mkt', 'G_cond', 'C_ND_corr_cond'],
                                     inputs={'tau': taushock})
    irf['E'] = 0 * np.arange(T)
    irf['i'] = 0 * np.arange(T)

    # compare results to analytical formulas from baseline_formulas.py
    effective_sigma = (1 - share_durables) * sigma + share_durables * common['eps_D']
    r = solve_shortrun(dtau, common['openness'], common['eta'], common['gamma'], effective_sigma)

    # compare results
    variable_mapping = {
        'Y': 'dlogY',
        'P': 'dlogP',
        'X': 'dlogX',
        'M': 'dlogM',
        'N': 'dlogN',
        'E': 'dlogE',
        'C': 'dlogC',
        'P_M': 'dlogPF',
    }

    for var_model, var_analytical in variable_mapping.items():
        model_value = irf[var_model][0] / ss[var_model]
        analytical_value = getattr(r, var_analytical)
        # print(var_model + f" --- Model: {model_value:.7f}, Analytical: {analytical_value:.7f}")
        assert np.isclose(model_value, analytical_value, rtol=1e-4), f"Mismatch in {var_model} (durables): model {model_value}, analytical {analytical_value}"

