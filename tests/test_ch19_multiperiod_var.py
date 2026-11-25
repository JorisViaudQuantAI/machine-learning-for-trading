import ast
from pathlib import Path

import numpy as np
import pytest
from scipy import stats


# Execute only the notebook function, not the top-level data and plotting cells.
def _load_function(name):
    path = Path(__file__).resolve().parents[1] / "19_risk_management/01_var_cvar.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "stats": stats, "N_MC_SIMULATIONS": 100, "SEED": 42}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _mock_paths(monkeypatch, paths):
    class RNG:
        def standard_t(self, df, size):
            return np.broadcast_to(np.array(paths), size)

    monkeypatch.setattr(stats.t, "fit", lambda returns: (5.0, 0.0, 1.0))
    monkeypatch.setattr(np.random, "default_rng", lambda seed: RNG())


def test_multi_period_var_compounds(monkeypatch):
    mc_var = _load_function("monte_carlo_var")
    _mock_paths(monkeypatch, [0.1, -0.1])
    assert mc_var(np.array([0.01, -0.01]), horizon=2) == pytest.approx(0.01)


def test_one_period_var_unchanged(monkeypatch):
    mc_var = _load_function("monte_carlo_var")
    _mock_paths(monkeypatch, [-0.1])
    assert mc_var(np.array([0.01, -0.01]), horizon=1) == pytest.approx(0.1)


def test_invalid_simple_return_rejected(monkeypatch):
    mc_var = _load_function("monte_carlo_var")
    _mock_paths(monkeypatch, [-1.1, 0.1])
    with pytest.raises(ValueError, match="simple returns"):
        mc_var(np.array([0.01, -0.01]), horizon=2)
