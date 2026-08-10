"""Regression tests for `vulture_whitelist.py`.

`vulture_whitelist.py` is parsed as plain Python by both vulture (an AST walk, to register names as "used" so its dead-code scan doesn't flag them) and ruff (full name resolution). Every name the file references therefore has to actually be defined there via a real import or assignment -- a bare, unimported name is not valid Python and raises `NameError` the moment the module is executed, which is exactly what ruff's F821 (undefined-name) rule flags statically without needing to run it.
"""

import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WHITELIST_PATH = REPO_ROOT / "vulture_whitelist.py"


def test_vulture_whitelist_has_no_undefined_names():
    """Executing vulture_whitelist.py must not raise NameError.

    A stale entry -- e.g. a bare name left behind after the unused import it was whitelisting was itself removed elsewhere in the codebase -- is invalid Python: running the module raises NameError for that name. Vulture only walks the AST and never surfaces this, but the file must still be valid, executable Python.
    """
    runpy.run_path(str(WHITELIST_PATH), run_name="__main__")


def test_no_f821_findings_anywhere_in_the_repo():
    """ruff's F821 (undefined-name) check must be clean across the whole repository.

    A direct, automated version of the same check this fix was verified against manually, so any future undefined-name regression -- in vulture_whitelist.py or anywhere else -- fails the suite instead of only being caught by a separate lint step.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
