import numpy as np
import matplotlib.pyplot as plt

from plotting import plots_baseline # enforce plt settings


def plot_irfs_quant_model(filename, ss, irfs):
    Tplot = 16
    plt.figure(figsize=(12, 6))

    other_variables = {'gdp': 'Real GDP',
                       'Y': 'Real Output',
                       'C_ND': 'Consumption',
                       'TB': 'Trade Balance',
                       'i': 'Nominal Interest Rate',
                       'E': 'Nominal Exchange Rate',
                       'P': 'Domestic Prices'}

    models = {'natural': 'Natural rate rule', 'taylor': 'Taylor rule',
              'fixed E': 'Fixed nominal rate'}

    # set normalizations
    norms = {'gdp': ss['gdp'], 'Y': ss['Y'], 'C_ND': ss['C_ND'], 'TB': ss['gdp'], 'i': 1/4, 'E': 1, 'P': 1}
    ylabels = {'gdp': r'$\%$', 'Y': r'$\%$', 'C_ND': r'$\%$', 'TB': r'$\%$ of GDP',
               'i': r'pp (annualized)', 'E': r'$\%$', 'P': r'$\%$'}

    nrows, ncols = 2, 4

    # Tariff panel
    plt.subplot(nrows, ncols, 1)
    key1 = list(irfs.keys())[0]
    plt.plot(irfs[key1]['tau'][:Tplot], color='black')
    plt.axhline(0, color='black', ls='--', lw=0.5)
    plt.title('Tariff')
    plt.ylabel(r'$\%$')

    # Loop over other variables
    for idx, (var, title) in enumerate(other_variables.items(), start=2):
        plt.subplot(nrows, ncols, idx)
        # loop over the models in models
        for key, title_model in models.items():
            plt.plot(irfs[key][var][:Tplot] / norms[var], label=title_model)
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.title(title)
        plt.ylabel(ylabels[var])
        if idx == 2:
            plt.legend(framealpha=0)
        if idx >= 5:
            plt.xlabel('Quarters')
    plt.tight_layout()
    plt.savefig(f'figures/' + filename + '.pdf', transparent=True)


def table_impact_effect_tariffs(ss, irfs_unilateral, irfs_retaliation):

    def impact(x):
        x = np.asarray(x)
        return x[0]

    # Unilateral (basic model)
    uni_no_mp = irfs_unilateral['fixed E']  # "No MP"
    uni_taylor = irfs_unilateral['taylor']  # Taylor rule
    uni_natural = irfs_unilateral['natural']  # Natural rate

    # Retaliation (basic model with taustar shock)
    ret_no_mp = irfs_retaliation['fixed E']
    ret_taylor = irfs_retaliation['taylor']
    ret_natural = irfs_retaliation['natural']

    def fmt_pct(v):
        return f"{v:.2f}%"

    def fmt_bp(v):
        return f"{v*4:.2f}pp"

    # Variables to summarize: GDP (N), Trade balance (TB), Nominal exchange rate (E)
    vars_for_table = ['gdp', 'TB', 'i', 'E']

    # normalizations
    norms = {'gdp': ss['gdp'], 'TB': ss['gdp'], 'i': 1, 'E': 1}

    rows = []
    for var in vars_for_table:
        vals = [
            impact(uni_no_mp[var] / norms[var]),
            impact(uni_taylor[var] / norms[var]),
            impact(uni_natural[var] / norms[var]),
            impact(ret_no_mp[var] / norms[var]),
            impact(ret_taylor[var] / norms[var]),
            impact(ret_natural[var] / norms[var]),
        ]
        if var == 'i':
            row_str = "\t".join(fmt_bp(v) for v in vals)
        else:
            row_str = "\t".join(fmt_pct(v) for v in vals)
        rows.append(row_str)

    # Print 3 x 6 block (GDP, TB, E in this order), tab-delimited
    for row in rows:
        print(row)


def plot_irfs_comparison_durables(ss_basic, ss_durables, irfs_basic, irfs_durables, period_of_shock=0):
    # Make a plot like the ones above, except only showing the following variables: (in a row)
    # - First panel: C from irfs_basic, C from irfs_durables
    # - Second panel: C_ND and C_D from irfs_durables
    # - Third panel: GDP from both models
    with plt.rc_context({'font.size': 12}):
        Tplot = 16
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(irfs_basic['C'][:Tplot] / ss_basic['C'], label=r'Basic model (high $\sigma$)', color='blue')
        plt.plot(irfs_durables['C'][:Tplot] / ss_durables['C'], label='Durables model', color='orange')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Consumption')
        plt.ylabel(r'$\%$')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        plt.yticks([-1.5, -1.0, -0.5, 0])
        plt.legend(framealpha=0)

        plt.subplot(1, 3, 2)
        plt.plot(irfs_durables['C_ND'][:Tplot] / ss_durables['C_ND'], label='Nondurable consumption', color='brown')
        plt.plot(irfs_durables['C_D'][:Tplot] / ss_durables['C_D'], label='Durable consumption', color='red', ls='dashed')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('$C$ components (durables model)')
        plt.ylabel(r'$\%$')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        plt.legend(framealpha=0)

        plt.subplot(1, 3, 3)
        plt.plot(irfs_basic['gdp'][:Tplot] / ss_basic['gdp'], label=r'Basic model (high $\sigma$)', color='blue')
        plt.plot(irfs_durables['gdp'][:Tplot] / ss_durables['gdp'], label='Durables model', color='orange')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Real GDP')
        plt.ylabel(r'$\%$')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        plt.legend(framealpha=0)

        plt.tight_layout()
        plt.savefig(f'figures/fig11_comparison_durables.pdf', transparent=True)


def plot_irfs_anticipated_tariff_durables(ss, irfs, period_of_shock):
    # Plot with following panels:
    # - First panel: C_D and C_ND
    # - Second panel: GDP
    # - Third panel: Trade balance
    with plt.rc_context({'font.size': 12}):
        Tplot = 16
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(irfs['C_ND'][:Tplot] / ss['C_ND'], label='Nondurable consumption', color='brown')
        plt.plot(irfs['C_D'][:Tplot] / ss['C_D'], label='Durable consumption', color='red', ls='dashed')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Consumption components')
        plt.ylabel(r'$\%$')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        plt.legend(framealpha=0, fontsize=11)

        plt.subplot(1, 3, 2)
        plt.plot(irfs['TB'][:Tplot] / ss['gdp'], label='Trade balance', color='orange')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Trade balance')
        plt.ylabel(r'$\%$ of GDP')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        plt.yticks([0, 0.5, 1, 1.5])
        #plt.legend(framealpha=0)

        plt.subplot(1, 3, 3)
        plt.plot(irfs['gdp'][:Tplot] / ss['gdp'], label='Real GDP', color='orange')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Real GDP')
        plt.ylabel(r'$\%$')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        #plt.legend(framealpha=0)

        plt.tight_layout()
        plt.savefig(f'figures/fig12_anticipated_tariff_durables.pdf', transparent=True)


def plot_comparison_inventories(filename, ss_durables, ss_inventories, irfs_durables, irfs_inventories, period_of_shock=0):
    # Make a plot with the following panels
    # - First panel: M (imports) from both models, Exports X from both models
    # - Second panel: trade balance TB from both models
    # - Third panel: GDP from both models

    # use orange for durables model, and green for inventories model
    with plt.rc_context({'font.size': 12}):
        Tplot = 16
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(irfs_durables['M'][:Tplot] / ss_durables['M'], label='Imports (durables model)', color='orange')
        plt.plot(irfs_inventories['M'][:Tplot] / ss_inventories['M'], label='Imports (+ inventories)', color='green')
        plt.plot(irfs_durables['X'][:Tplot] / ss_durables['X'], label='Exports (durables model)', color='orange', ls='dashed')
        plt.plot(irfs_inventories['X'][:Tplot] / ss_inventories['X'], label='Exports (+ inventories)', color='green', ls='dashed')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Imports and Exports')
        plt.ylabel(r'$\%$')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        if period_of_shock == 0:
            plt.legend(framealpha=0, fontsize=11)
        else:
            plt.legend(framealpha=0.75, fontsize=11)

        plt.subplot(1, 3, 2)
        plt.plot(irfs_durables['TB'][:Tplot] / ss_durables['gdp'], label='Durables model', color='orange')
        plt.plot(irfs_inventories['TB'][:Tplot] / ss_inventories['gdp'], label='+ Inventories', color='green')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Trade Balance')
        plt.ylabel(r'$\%$ of GDP')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        plt.legend(framealpha=0)

        plt.subplot(1, 3, 3)
        plt.plot(irfs_durables['gdp'][:Tplot] / ss_durables['gdp'], label='Durables model', color='orange')
        plt.plot(irfs_inventories['gdp'][:Tplot] / ss_inventories['gdp'], label='+ Inventories', color='green')
        plt.axhline(0, color='black', ls='--', lw=0.5)
        plt.axvline(period_of_shock, color='gray', ls=':', lw=0.5)
        plt.title('Real GDP')
        plt.ylabel(r'$\%$')
        plt.xlabel('Quarters')
        plt.xticks([0, 5, 10, 15])
        plt.legend(framealpha=0)

        plt.tight_layout()
        plt.savefig(f'figures/' + filename + '.pdf', transparent=True)


def plot_peaks_with_anticipation_horizon(ss, irfs_list, horizons):
    # Plot the following three panels
    # - First panel: Peak negative imports response vs anticipation horizon
    # - Second panel: Peak positive trade balance response vs anticipation horizon
    # - Third panel: Peak negative GDP response vs anticipation horizon
    with plt.rc_context({'font.size': 12}):
        plt.figure(figsize=(12, 4))
        peak_imports = []
        peak_tb = []
        peak_gdp = []
        for irfs in irfs_list:
            peak_imports.append(np.min(irfs['M'] / ss['M']))
            peak_tb.append(np.max(irfs['TB'] / ss['gdp']))
            peak_gdp.append(np.min(irfs['gdp'] / ss['gdp']))

        plt.subplot(1, 3, 1)
        plt.plot(horizons, peak_imports, marker='o')
        plt.title('Peak negative import response', fontsize=13)
        plt.ylabel(r'$\%$')
        plt.yticks([-26, -26.5, -27, -27.5])
        plt.xlabel('Anticipation horizon (quarters)')
        plt.xticks(horizons)
        #plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(horizons, peak_tb, marker='o')
        plt.title('Peak positive trade balance response', fontsize=13)
        plt.ylabel(r'$\%$ of GDP')
        plt.xlabel('Anticipation horizon (quarters)')
        plt.xticks(horizons)
        #plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(horizons, peak_gdp, marker='o')
        plt.title('Peak negative GDP response', fontsize=13)
        plt.ylabel(r'$\%$')
        plt.xlabel('Anticipation horizon (quarters)')
        plt.xticks(horizons)
        #plt.grid()
        plt.tight_layout()
        plt.savefig(f'figures/fig15_anticipated_horizon_peaks.pdf', transparent=True)
