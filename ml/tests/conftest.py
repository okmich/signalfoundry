"""
Root conftest for okmich_quant_ml tests.

apricot (a submodular selection library) fails to load with the current version
of Numba due to a JIT compilation bug in featureBased.py. It is a transitive
dependency pulled in by the custom pomegranate build via _utils.py and is not
used anywhere in this test suite. Mock it out before any imports execute.
"""

import sys
from unittest.mock import MagicMock

# Stub the entire apricot namespace so pomegranate._utils can import
# FacilityLocationSelection without triggering the broken Numba compilation.
_apricot_mock = MagicMock()
sys.modules["apricot"] = _apricot_mock
sys.modules["apricot.functions"] = _apricot_mock.functions
sys.modules["apricot.functions.featureBased"] = _apricot_mock.functions.featureBased