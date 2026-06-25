"""Pytest bootstrap.

Adds the ``src`` directory to ``sys.path`` so the test suite can ``import
ao_fed_ifu`` without an editable install.  CI installs the package properly
(``pip install -e .[test]``); this just makes ``pytest`` work out of the box for
a fresh checkout.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
