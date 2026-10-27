"""EASYSPYN_NUMBA=0 switches the compiled spectrum kernels off; Numba is on by default when installed."""

import os
import subprocess
import sys

import pytest

numba = pytest.importorskip("numba")  # the default install has it; skip on interpreters without it

CODE = "import easyspyn.kernels as k; print(k.HAVE_NUMBA)"


def _have_numba(env_value: str | None) -> str:
    env = {key: value for key, value in os.environ.items() if key != "EASYSPYN_NUMBA"}
    if env_value is not None:
        env["EASYSPYN_NUMBA"] = env_value
    result = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_numba_is_used_by_default():
    assert _have_numba(None) == "True"


def test_environment_variable_switches_it_off():
    assert _have_numba("0") == "False"
