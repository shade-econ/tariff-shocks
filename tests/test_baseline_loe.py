import numpy as np
import pytest

from baseline import baseline_linear, baseline_loe
from test_baseline_linear import random_parameters


@pytest.mark.parametrize("kw", [dict(), dict(retaliation=True),
                                dict(natural=True), dict(natural=True, retaliation=True)])
def test_soe_limit_consistency(kw):
    """Our standard small open economy model should agree with large open economy for lambda=0 limit"""
    *params, dtau = random_parameters()
    results_soe = baseline_linear.solve_shortrun(dtau, *params, **kw)
    results_soe_limit = baseline_loe.solve_loe(dtau, 0, *params, **kw)

    overlap = ["dlogE", "dlogPF", "dlogP", "dlogX", "dlogC", "dlogY", "dlogM", "dlogGDP", "dTB_Tr"]
    for k in overlap:
        assert np.isclose(getattr(results_soe, k), getattr(results_soe_limit, k)), f"Mismatch in {k} for SOE limit"


@pytest.mark.parametrize("kw", [dict(retaliation=True), dict(natural=True, retaliation=True)])
def test_symmetry(kw):
    """Large open economy model should be symmetry for home and foreign when lambda=50% and retaliation"""
    *params, dtau = random_parameters()
    results = baseline_loe.solve_loe(dtau, 0.5, *params, **kw)

    pairs = [('dlogGDP', 'dlogGDP_star'), ('dlogP', 'dlogPF_star'), ('dlogX', 'dlogM'), ('ihat', 'ihat_star')]
    for k1, k2 in pairs:
        assert np.isclose(getattr(results, k1), getattr(results, k2)), f"Mismatch in {k1} and {k2}"
