import ast
from pathlib import Path

import numpy as np
import pytest


# Execute only the notebook function, not the top-level data and plotting cells.
def _load_function(name):
    path = Path(__file__).resolve().parents[1] / "19_risk_management/01_var_cvar.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_expected_shortfall_is_subadditive_with_atoms():
    es = _load_function("historical_cvar")
    a = np.zeros(100)
    b = np.zeros(100)
    a[:3] = -0.10
    b[3:6] = -0.10
    es_a, es_b, es_ab = map(es, (a, b, a + b))
    np.testing.assert_allclose([es_a, es_b, es_ab], [0.06, 0.06, 0.10])
    assert es_ab <= es_a + es_b


def test_expected_shortfall_fractional_tail_mass():
    es = _load_function("historical_cvar")
    assert es(np.array([-0.3, -0.1, 0.0]), 0.5) == pytest.approx(0.35 / 1.5)
