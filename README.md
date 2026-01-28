# The Macroeconomics of Tariff Shocks
This repository replicates the figures, tables, and results in "[The Macroeconomics of Tariff Shocks](https://shade-econ.github.io/tariff_shocks.pdf)", by Adrien Auclert, Matthew Rognlie, and Ludwig Straub.

The code requires standard numerical Python packages (`numpy`, `scipy`, and `matplotlib`), and the quantitative part also requires the [sequence-space Jacobian toolkit](https://github.com/shade-econ/sequence-jacobian), which also requires `numba` and can be installed using `pip install sequence-jacobian`. There are also tests, which can be run using the `pytest` package.

If there are latex errors in formatting plots, try setting `usetex=False` near the top of `plotting/plots_baseline.py`. If you run into any other difficulties with the code, please feel free to post on the repository's issue tracker. We have tested the code using Python 3.12+.

## Notebooks

The results of the paper are obtained in two Jupyter notebooks, both of which should run in just a few seconds:
- `baseline_results.ipynb` has all figures, tables, and numerical results in the paper that do not involve the quantitative model in section 6.
- `quantitative_results.ipynb` has results associated with the quantitative model in section 6. This includes figures 10–15 and table 3 in section 6, and figures 19–21 in the appendix. Note that this requires the [`sequence-jacobian`](https://github.com/shade-econ/sequence-jacobian) toolkit to be installed.

## Supporting code

The code supporting these notebooks is contained in several subfolders:

- `baseline/` has the code for the paper's main model (everything excluding the quantitative model in section 6), including:
   * `baseline_formulas.py`, which implements the analytical formulas in propositions 2, 3, 5, and 10.
   * `baseline_linear.py`, which implements and numerically solves the linearized equations of the model, for both the short and long run.
   * `baseline_nonlinear.py`, which implements and numerically solves the nonlinear equations of the model, including welfare and chained GDP, for both the short and long run.
   * `baseline_loe.py`, which implements and numerically solves the linearized equations of the "large open economy" version of the model from appendix C.6.
 
- `quantitative/` has the code for the quantitative model in section 6, including:
   * `quantitative_blocks.py`, which builds the quantitative model by defining and combining "blocks" using the sequence-space Jacobian toolkit.
   * `quantitative_steadystate.py`, which calibrates the steady state and parameters of the quantitative model.

- `tests/` has testing code that can be automatically run using `pytest`, by calling `pytest` from the main directory. This includes:
   * `test_baseline_linear.py`, which validates that propositions 1–5 and 10 hold in our numerically-solved linear model.
   * `test_baseline_nonlinear.py`, which validates that the nonlinear model agrees with the linear model for small shocks, and that proposition 11 on welfare holds numerically.
   * `test_baseline_loe.py`, which validates that the linearized large open economy model agrees with our main model in the $\lambda\rightarrow 0$ small open economy limit, and also that all outcomes are symmetric for the home and foreign economies when $\lambda=0.5$.
   * `test_quantitative.py`, which validates that the basic quantitative model (without durables or inventories) agrees with our linearized baseline in the $\beta\rightarrow 1$ limit, and also that our equivalence result for durables from section 4.5 holds in the relevant limit.

- `plotting/` has supporting functions to build figures (customizing appearance, labels, and so on).
   * By default, we use latex rendering with the `mathpazo` package for our plots, but if there is not a local latex installation, this will not be used. If you run into latex issues with plots, try setting `usetex=False` near the top of `plots_baseline.py`.

- `figures/` is where figures are saved after being produced.
