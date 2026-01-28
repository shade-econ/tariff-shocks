import numpy as np
from quantitative.quantitative_blocks import model, terminal_eqm

# Compute additional steady state values
def get_steady_states(common, sigma_basic, sigma_dur, share_durables, inv_to_sales):

    # update each with common initial conditions and normalizations
    common.update({
        'E': 1,  # Nominal exchange rate
        'W': 1,  # Nominal wage
        'Y': 1,  # Output
        'pi_w': 0,  # wage inflation
        'tau': 0,  # Tariff
        'taustar': 0,  # Foreign tariff
        'PFstar': 1,  # Foreign price level
        'NFA': 0,  # Net foreign assets
        'A': 0,  # Net foreign assets
        'QD': 1,  # Q for durables
        'QS': 1,  # Q for inventories
        'eps_S': 20  # irrelevant, eps_S calibrated below
    })
    common['i'] = common['istar'] = common['r_ante'] = common['r'] = 1 / common['beta'] - 1


    # Calibration 1: No durables and inventories
    ss_basic = {**common,
                'sigma': sigma_basic,  # EIS
                'inv_to_sales': 1e-9,  # inventories to sales ratio
                'share_durables': 1e-9,  # Share of durables in consumption
                }

    # Calibration 2: Durables but no inventories
    ss_dur = {**common,
              'sigma': sigma_dur,  # EIS
              'inv_to_sales': 1e-9,  # inventories to sales ratio
              'share_durables': share_durables,  # Share of durables in consumption
              }

    # Calibration 3: Durables and inventories
    ss_inv = {**common,
              'sigma': sigma_dur,  # EIS
              'share_durables': share_durables,  # Share of durables in consumption
              'inv_to_sales': inv_to_sales,  # inventories to sales ratio
              }


    for ss in [ss_basic, ss_dur, ss_inv]:

        ss['chi'] = ss['i'] * ss['inv_to_sales']  # share of inventories in production
        assert ss['chi'] < 1, "inv_to_sales ratio too high given interest rate"

        ss['alpha_tilde'] = ss['openness'] / (1 - ss['chi'])

        ss['S'] = ss['inv_to_sales'] * ss['alpha_tilde']

        ss['N'] = 1 - ss['alpha_tilde']  # labor share
        ss['G'] = ss['alpha_tilde']  # import bundle share in production
        ss['M'] = ss['X'] = (1 - ss['chi']) * ss['alpha_tilde']  # imports and exports

        ss['income'] = 1 - ss['alpha_tilde'] + ss['chi'] * ss['alpha_tilde']

        ss['C_D'] = ss['share_durables'] * ss['income']
        ss['C_ND'] = (1 - ss['share_durables']) * ss['income']
        ss['D'] = ss['C_D'] / ss['delta']

        ss['phi_D'] = (ss['r'] + ss['delta']) * (ss['D'] / ss['C_ND']) ** (1/ss['sigma'])
        ss['phi_N'] = ss['N'] ** (-1/ss['phi']) * ss['C_ND'] ** (-1/ss['sigma'])

        # get additional variables needed for household block
        ss['C_ND_corr2'] = ss['C_ND_corr'] = 0
        ss['E_corr2'] = ss['E_corr'] = 0
        ss['income_term'] = ss['income']

        # set terminal conditions
        ss['E_term'] = ss['E']


    for ss in [ss_basic, ss_dur, ss_inv]:

        ss['G_term'] = 1
        test = model.steady_state(ss, dissolve=['household', 'nominal_wage', 'importers', 'terminal_eqm'])
        # test = model.solve_steady_state(ss, unknowns={'G_term': 1.}, targets={'G_term': 'G'}, dissolve=['household', 'nominal_wage', 'importers', 'terminal_eqm'])
        test['G_term'] = test['G']

        for field in ['val_cond', 'inv_cond', 'euler', 'budget', 'dur_inv_cond', 'dur_val_cond', 'uip_cond',
                      'bop', 'goods_mkt', 'G_cond', 'nkpc_res',
                      'TB', 'labor_wedge']:
            if field in test:
                assert np.isclose(test[field], 0), f"{field} does not hold in steady state"

        for field in ['P', 'P_G', 'P_M', 'W', 'QD', 'QS']:
            assert np.isclose(test[field], 1), f"{field} not equal to 1 in steady state"

        ss['E_term'] = ss['E']
        ss['G_term'] = ss['G']
        ss_check = terminal_eqm.steady_state(ss, dissolve=['terminal_eqm'])
        for field in ['cond_goods', 'cond_G']:
            if field in ss_check:
                assert np.isclose(ss_check[field], 0), f"{field} does not hold in terminal equilibrium steady state"

    # Compute all objects
    ss_basic = model.steady_state(ss_basic, dissolve=['household', 'nominal_wage', 'importers', 'terminal_eqm'])
    ss_dur = model.steady_state(ss_dur, dissolve=['household', 'nominal_wage', 'importers', 'terminal_eqm'])
    ss_inv = model.steady_state(ss_inv, dissolve=['household', 'nominal_wage', 'importers', 'terminal_eqm'])

    # calibrate eps_S to match target inventory to sales response for ss_inv
    target_response = 160  # from Alessandria Khan Khederlarian
    def objective(eps_S, T=50, dtau=10, period_of_shock=4):
        ss_here = ss_inv.copy()
        ss_here['eps_S'] = eps_S
        irfs_here = model.solve_impulse_linear(ss_here, unknowns=['pi_w', 'Y', 'G', 'C_ND_corr2'],
                                                             targets=['nkpc_res', 'goods_mkt', 'G_cond',
                                                                      'C_ND_corr_cond'],
                                                             inputs={'tau': np.concatenate((np.zeros(period_of_shock),
                                                                                            np.ones(T - period_of_shock) * dtau))})
        peak_response = np.max(irfs_here['inv_sales'] / ss_inv['inv_sales'])
          # target peak response of inventory to sales ratio
        return peak_response - target_response
    from scipy.optimize import brentq
    eps_S_calibrated = brentq(objective, 3.0, 8.0)
    ss_inv['eps_S'] = eps_S_calibrated

    return {'basic': ss_basic, 'durables': ss_dur, 'inventories': ss_inv}



