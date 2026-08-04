# the demos are documentation, and documentation that is never run rots.
# this repository already had notebooks importing a module that no longer
# existed, so the scripted demo gets a smoke test.

import runpy
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "demos" / "symbolic_representations.py"


@pytest.mark.skipif(not DEMO.exists(), reason="demo script not present")
def test_symbolic_representations_demo_runs(capsys):
    namespace = runpy.run_path(str(DEMO), run_name="imported_by_test")
    namespace["main"]()

    out = capsys.readouterr().out
    assert "SAX" in out
    assert "lower bound holds          :  True" in out
    assert "ANOMALY" in out
