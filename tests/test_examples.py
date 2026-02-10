from pathlib import Path
import runpy

import matplotlib
import numpy as np


matplotlib.use("Agg", force=True)


def _run_example(path: str):
    namespace = runpy.run_path(str(Path(path)))
    assert "main" in namespace
    result = namespace["main"](show=False)
    assert result is not None
    return result


def test_examples_e04():
    rrfracr0, mmfracm0, rr, vv = _run_example("examples/e04.py")
    assert rrfracr0.shape == mmfracm0.shape
    assert rr.shape == vv.shape
    assert np.all(np.isfinite(vv))


def test_examples_e11():
    radii, v_er11, r_out = _run_example("examples/e11.py")
    assert radii.shape == v_er11.shape
    assert np.nanmax(v_er11) > 0
    assert np.isfinite(r_out)


def test_examples_er11e04():
    rr, vv, rmerge, vmerge, rmax = _run_example("examples/er11e04.py")
    assert rr.shape == vv.shape
    assert np.nanmax(vv) > 0
    assert np.isfinite(rmerge)
    assert np.isfinite(vmerge)
    assert np.isfinite(rmax)
