import numpy as np
import matplotlib.pyplot as plt
from shutil import which

# plot appearance settings
plt.rcParams['font.family'] = 'serif'
if which("latex") is not None:
    # note: if still having latex issues with plots, set usetex=False instead
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')


def plot_gamma_sigma_split(GDPH_fn, TB_fn,
                        xlim_left, xlim_right, ylim,
                        filename, calibrated_point=None, 
                        num_points_left=150, num_points_right=50, num_points_sigma=200):
    """
    Plot recession and trade balance regions over (gamma, sigma) space,
    splitting plot into two panels with a broken x-axis to show GDP and TB conditions

    Parameters:
    - GDPH_fn: function (gamma, sigma) → float, GDP change
    - TB_fn: function (gamma, sigma) → float, trade balance change
    - xlim_left, xlim_right: tuples (min, max) for left and right gamma axes
    - ylim: tuple (min, max) for sigma axis
    - calibrated_point: optional (gamma, sigma) tuple to mark on plot
    - num_points_left, num_points_right, num_points_sigma: grid density
    """

    # Setup grids
    sigma_vals = np.linspace(ylim[0] + 0.001, ylim[1], num_points_sigma)
    gamma_vals = np.concatenate([
        np.linspace(xlim_left[0] + 0.001, xlim_left[1], num_points_left),
        np.linspace(xlim_right[0], xlim_right[1], num_points_right)
    ])
    sigma_grid, gamma_grid = np.meshgrid(sigma_vals, gamma_vals)

    # Evaluate functions over the grid
    GDPH_grid = np.vectorize(GDPH_fn)(gamma_grid, sigma_grid)
    TB_grid = np.vectorize(TB_fn)(gamma_grid, sigma_grid)

    # Setup figure with broken axis
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True,
                                   gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.01})

    def plot_contours(ax, g_grid, s_grid, GDP_grid, TB_grid):
        recession_mask = (GDP_grid < 0) & (TB_grid < 0)
        surplus_mask = (GDP_grid < 0) & (TB_grid > 0)

        ax.contourf(g_grid, s_grid, recession_mask, levels=[0.5, 1], colors='lightcoral', alpha=0.5)
        ax.contourf(g_grid, s_grid, surplus_mask, levels=[0.5, 1], colors='lightsteelblue', alpha=0.5)

        ax.contour(g_grid, s_grid, TB_grid, levels=[0], colors='red', linestyles='-', linewidths=3)
        ax.contour(g_grid, s_grid, GDP_grid, levels=[0], colors='blue', linestyles='-', linewidths=3)

    # Split masks
    left_mask = gamma_vals <= xlim_left[1]
    right_mask = gamma_vals >= xlim_right[0]

    # Plot left panel
    plot_contours(ax1, gamma_grid[left_mask], sigma_grid[left_mask],
                  GDPH_grid[left_mask], TB_grid[left_mask])
    ax1.set_xlim(*xlim_left)
    ax1.set_ylim(*ylim)
    ax1.set_ylabel(r'Intertemporal elasticity $\sigma$', fontsize=24)
    ax1.set_xlabel(r'Export demand elasticity $\gamma$', fontsize=24)
    ax1.grid(True, linestyle=':', linewidth=0.7)

    # Plot right panel
    plot_contours(ax2, gamma_grid[right_mask], sigma_grid[right_mask],
                  GDPH_grid[right_mask], TB_grid[right_mask])
    ax2.set_xlim(*xlim_right)
    ax2.grid(True, linestyle=':', linewidth=0.7)

    # Tick label size
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)

    # Add diagonal break marks
    d = 0.015  # size of diagonal lines
    kwargs = dict(color='k', clip_on=False, linewidth=2)
    for y in [0, 1]:
        ax1.plot((1, 1), (y - d, y + d), transform=ax1.transAxes, **kwargs)
        ax2.plot((0, 0), (y - d, y + d), transform=ax2.transAxes, **kwargs)

    # Plot calibrated point if provided
    if calibrated_point:
        gamma_star, sigma_star = calibrated_point
        target_ax = ax1 if xlim_left[0] <= gamma_star <= xlim_left[1] else ax2
        target_ax.plot(gamma_star, sigma_star, 'x', color='black', markersize=20)
        target_ax.text(gamma_star, sigma_star - 0.2, 'Calibrated parameters', fontsize=22,
                       color='black', ha='center', va='top')

    # Add annotations
    ax1.text(np.mean(xlim_left), ylim[1] * 0.6, 'Recession,\nImproving trade balance',
             fontsize=22, color='navy', ha='center')
    ax2.text(xlim_right[1] - 0.1, ylim[0] + 0.05, 'Recession,\nWorsening\n trade balance',
             fontsize=22, color='darkred', ha='right', va='bottom')

    # Adjust layout and save
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
    plt.savefig(f'figures/{filename}', bbox_inches='tight', transparent=True)


def plot_gamma_sigma_simple(GDPH_fn, TB_fn,
                            xlim, ylim, filename,
                            num_points_gamma=200, num_points_sigma=200,
                            calibrated_point=None):
    """
    Plot recession and trade balance regions over (gamma, sigma) space.

    Parameters:
    - GDPH_fn: function (gamma, sigma) → float, GDP change
    - TB_fn: function (gamma, sigma) → float, trade balance change
    - xlim, ylim: axis bounds (tuples)
    - num_points_gamma, num_points_sigma: grid density
    - calibrated_point: optional (gamma, sigma) tuple to mark on plot
    """
    # Create grid
    sigma_vals = np.linspace(ylim[0] + 0.001, ylim[1], num_points_sigma)
    gamma_vals = np.linspace(xlim[0] + 0.001, xlim[1], num_points_gamma)
    sigma_grid, gamma_grid = np.meshgrid(sigma_vals, gamma_vals)

    # Evaluate functions over the grid
    GDPH_grid = np.vectorize(GDPH_fn)(gamma_grid, sigma_grid)
    TB_grid = np.vectorize(TB_fn)(gamma_grid, sigma_grid)

    # Setup figure
    _, ax = plt.subplots(figsize=(14, 8))

    # Shading regions
    recession_mask = (GDPH_grid < 0) & (TB_grid < 0)
    surplus_mask = (GDPH_grid < 0) & (TB_grid > 0)

    ax.contourf(gamma_grid, sigma_grid, recession_mask, levels=[0.5, 1],
                colors='lightcoral', alpha=0.5)
    ax.contourf(gamma_grid, sigma_grid, surplus_mask, levels=[0.5, 1],
                colors='lightsteelblue', alpha=0.5)

    # Contour lines
    tb_zero = ax.contour(gamma_grid, sigma_grid, TB_grid, levels=[0],
                         colors='red', linestyles='-', linewidths=3)
    gdp_zero = ax.contour(gamma_grid, sigma_grid, GDPH_grid, levels=[0],
                          colors='blue', linestyles='-', linewidths=3)

    # Axis labels and limits
    ax.set_xlabel(r'Export demand elasticity $\gamma$', fontsize=24)
    ax.set_ylabel(r'Intertemporal elasticity $\sigma$', fontsize=24)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.tick_params(labelsize=20)

    # Annotations
    ax.text(xlim[0] + 0.6, ylim[1] * 0.5, 'Recession,\n Improving trade balance',
            fontsize=22, color='navy', ha='center')
    ax.text(xlim[1] * 0.7, ylim[1] * 0.5, 'Recession,\n Worsening trade balance',
            fontsize=22, color='darkred', ha='center')

    # Calibrated point
    if calibrated_point:
        gamma_star, sigma_star = calibrated_point
        ax.plot(gamma_star, sigma_star, 'x', color='black', markersize=20)
        ax.text(gamma_star + 0.1, sigma_star - 0.1, 'Calibrated parameters',
                fontsize=22, color='black', ha='left', va='top')

    plt.tight_layout()
    plt.savefig(f'figures/{filename}', transparent=True)


def plot_nonlinear(results, results_nat, filestem,
                   trade_ylim=(-53, 50), gdp_ylim=(-10, 3.5), retaliation=False):
    # "results" should be list of result objects from model_nonlinear.solve_shortrun
    tau, X, M, E, TB, GDP, W, N = (np.array([getattr(res, k) for res in results])
                                for k in ('tau', 'X', 'M', 'E', 'TB', 'GDP', 'W', 'N'))
    
    tau_nat, X_nat, M_nat, E_nat, TB_nat, GDP_nat, W_nat, N_nat = (np.array([getattr(res, k) for res in results_nat])
                                for k in ('tau', 'X', 'M', 'E', 'TB', 'GDP', 'W', 'N'))
    
    assert np.array_equal(tau, tau_nat), "tau should be the same for both result sets"
    extra = 'and retaliation ' if retaliation else ''
    
    plt.figure(figsize=(4, 3.5))
    plt.plot(100*tau, 100*(X/X[0] - 1), label='exports', linewidth=2)
    plt.plot(100*tau, 100*(M/M[0] - 1), label='imports', linewidth=2)
    plt.plot(100*tau, 100*(E/E[0] - 1), label='exchange rate', linewidth=2)
    plt.plot(100*tau, 100*(TB / X[0]), label='trade balance (vs. SS exports/imports)', linewidth=2)
    plt.plot(100*tau, 100*(X_nat/X[0] - 1), label='(natural)', linewidth=2, color='C0', linestyle='--')
    plt.plot(100*tau, 100*(M_nat/M[0] - 1), linewidth=2, color='C1', linestyle='--')
    plt.plot(100*tau, 100*(E_nat/E[0] - 1), linewidth=2, color='C2', linestyle='--')
    plt.plot(100*tau, 100*(TB_nat / X[0]), linewidth=2, color='C3', linestyle='--')
    plt.ylabel(r'\% change from steady state')
    plt.xlabel(rf'Import {extra}tariff $\tau$')
    plt.ylim(trade_ylim)
    if retaliation:
        plt.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig(f'figures/{filestem}_trade.pdf', transparent=True)

    plt.figure(figsize=(4, 3.5))
    plt.plot(100*tau, 100*(N / N[0] - 1), label='labor', linewidth=2)
    plt.plot(100*tau, 100*(GDP / GDP[0] - 1), label='GDP', linewidth=2)
    plt.plot(100*tau, 100*(W - W[0])/GDP[0], label='welfare (cons. units)', linewidth=2)
    plt.plot(100*tau, 100*(N_nat / N[0] - 1), label='(natural)',linewidth=2, color='C0', linestyle='--')
    plt.plot(100*tau, 100*(GDP_nat / GDP[0] - 1), linewidth=2, color='C1', linestyle='--')
    plt.plot(100*tau, 100*(W_nat - W[0])/GDP[0], linewidth=2, color='C2', linestyle='--')
    plt.ylabel(r'\% change from steady state')
    plt.xlabel(rf'Import {extra}tariff $\tau$')
    plt.ylim(gdp_ylim)
    if retaliation:
        plt.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig(f'figures/{filestem}_gdp.pdf', transparent=True)


def plot_longrun_nonlinear(taus, N_unilateral, N_retaliation, filename):
    plt.figure(figsize=(5, 3))
    plt.plot(100*taus, 100*(N_unilateral / N_unilateral[0] - 1), label='unilateral', linewidth=2)
    plt.plot(100*taus, 100*(N_retaliation / N_retaliation[0] - 1), label='retaliation', linewidth=2)
    plt.ylabel(r'\% change in employment on impact')
    plt.xlabel(r'Permanent import tariff $\tau$ (\%)')
    plt.legend(framealpha=0)
    plt.yticks([-3, -2, -1, 0])
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', transparent=True)


def plot_longrun_nonlinear_withGDP(taus, N_unilateral, N_retaliation,
                                   GDP_SR_unilateral, GDP_SR_retaliation, GDP_LR_unilateral, GDP_LR_retaliation, filename):
    plt.figure(figsize=(5, 3))
    plt.plot(100*taus, 100*(N_unilateral / N_unilateral[0] - 1), label='unilateral (labor)', color='C0', linewidth=2)
    plt.plot(100*taus, 100*(GDP_SR_unilateral / GDP_SR_unilateral[0] - 1), label='unilateral (GDP)', linewidth=2, color='C0', linestyle='--')
    plt.plot(100*taus, 100*(GDP_LR_unilateral / GDP_LR_unilateral[0] - 1), label='unilateral (long-run GDP)', linewidth=2, color='C0', linestyle=':')
    plt.plot(100*taus, 100*(N_retaliation / N_retaliation[0] - 1), label='retaliation (labor)', color='C1', linewidth=2)
    plt.plot(100*taus, 100*(GDP_SR_retaliation / GDP_SR_retaliation[0] - 1), label='retaliation (GDP)', linewidth=2, color='C1', linestyle='--')
    plt.plot(100*taus, 100*(GDP_LR_retaliation / GDP_LR_retaliation[0] - 1), label='retaliation (long-run GDP)', linewidth=2, color='C1', linestyle=':')
    plt.ylabel(r'\% change')
    plt.xlabel(r'Permanent import tariff $\tau$ (\%)')
    plt.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', transparent=True)


def plot_gamma_sigma_welfare(gammas, sigmas_unilateral, sigmas_retaliation, alpha, filename):
    _, ax = plt.subplots(figsize=(14, 8))
    ax.tick_params(labelsize=20)

    # Shading between solid and dashed (retaliation and unilateral)
    between_mask = sigmas_unilateral > sigmas_retaliation
    ax.fill_between(
        gammas,
        sigmas_retaliation,
        sigmas_unilateral,
        where=between_mask,
        interpolate=True,
        color='lightcoral',
        alpha=0.5
    )

    # Shading above the dashed (unilateral) threshold
    above_mask = sigmas_unilateral < 4.0
    ax.fill_between(
        gammas,
        sigmas_unilateral,
        4.0,
        where=above_mask,
        interpolate=True,
        color='lightcoral',
        alpha=0.25
    )

    # Plot the thresholds
    ax.plot(gammas, sigmas_unilateral, 'r--', linewidth=3, label='Unilateral (temporary tariff)')
    ax.plot(gammas, sigmas_retaliation, 'r-', linewidth=3, label='Retaliation (temporary tariff)')

    # Vertical line at gamma = 1 / (1 - alpha)
    gamma_perm = 1 / (1 - alpha)
    ax.axvline(
        x=gamma_perm,
        color='gray',
        linestyle='-',
        linewidth=3,
        alpha=0.4,
        label='Retaliation (permanent tariff or stabilizing policy)'
    )

    ax.plot(1.5, 1.78, 'x', color='black', markersize=20)
    ax.text(1.5+0.05, 1.78-0.22, 'Calibrated parameters', fontsize=22, color='black', ha='left', va='top')

    # Axes labels and limits
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel(r'Export demand elasticity $\gamma$', fontsize=24)
    ax.set_ylabel(r'Intertemporal elasticity $\sigma$', fontsize=24)

    # Annotations
    ax.text(1.8, 0.63, 'Welfare loss\nunder retaliation', fontsize=22, color='darkred', ha='center')
    ax.text(2.8, 2.4, 'Welfare loss under\nunilateral or retaliation', fontsize=22, color='firebrick', ha='center')

    # Grid and legend
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.legend(frameon=False, loc='upper right', fontsize=18)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}", transparent=True)


def plot_nonlinear_welfare(tau_grid, output_gap, terms_of_trade, distortion, filename, retaliation=False, legend=False):
    plt.figure(figsize=(4, 3.5))
    if np.all(output_gap <= 1E-6):
        plt.plot(100*tau_grid, 100*output_gap, label='output gap', linewidth=2)
        total = terms_of_trade + output_gap + distortion
    else:
        total = terms_of_trade + distortion

    plt.plot(100*tau_grid, 100*terms_of_trade, label='terms of trade', linewidth=2, color='C1')
    plt.plot(100*tau_grid, 100*distortion, label='distortion', linewidth=2, color='C2')
    plt.plot(100*tau_grid, 100*total, label='total', color='k', linewidth=3.5)
    extra = 'and retaliatory ' if retaliation else ''
    plt.xlabel(rf'Import {extra}tariff $\tau$ (\%)')
    plt.ylabel('Welfare effect (pp in consumption units)')
    plt.tight_layout()
    if legend:
        plt.legend(framealpha=0)
    plt.savefig(f'figures/{filename}', transparent=True)


def plot_large_open_economy(lambdas, dlogGDP, dlogGDP_star, dlogX, dlogM, dTB_Tr, filestem):
    plt.figure(figsize=(4, 3.5))
    plt.plot(100*lambdas, 100*dlogGDP, label='Home GDP', linewidth=2)
    plt.plot(100*lambdas, 100*dlogGDP_star, label='Foreign GDP', linewidth=2)
    plt.ylabel(r'\% of steady state GDP')
    plt.xlabel(r'Home \% of world GDP')
    plt.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig(f'figures/{filestem}_GDP.pdf', transparent=True)

    plt.figure(figsize=(4, 3.5))
    plt.plot(100*lambdas, 100*dlogX, label='Home exports', linewidth=2)
    plt.plot(100*lambdas, 100*dlogM, label='Home imports', linewidth=2)
    plt.plot(100*lambdas, 100*dTB_Tr, label='Home trade balance', linewidth=2)
    plt.ylabel(r'\% of steady state imports (or exports)')
    plt.xlabel(r'Home \% of world GDP')
    plt.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig(f'figures/{filestem}_trade.pdf', transparent=True)
