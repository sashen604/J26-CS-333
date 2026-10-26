"""Tests for drift/ewma.py, including interface parity with CUSUM."""
import inspect

import numpy as np
import pytest

from drift.cusum import CUSUM
from drift.ewma import EWMA

INTERFACE = ("fit_reference", "update", "reset", "run")


def test_ewma_has_same_public_methods_as_cusum():
    for name in INTERFACE:
        assert callable(getattr(EWMA, name))
        assert list(inspect.signature(getattr(EWMA, name)).parameters) == \
            list(inspect.signature(getattr(CUSUM, name)).parameters)


@pytest.mark.parametrize("cls", [CUSUM, EWMA])
def test_detectors_are_interchangeable(cls):
    """The same experiment code runs on either detector."""
    rng = np.random.default_rng(0)
    d = cls("fused_windowed_anomaly_score").fit_reference(rng.normal(0.0, 1.0, 500))
    assert d.monitored == "fused_windowed_anomaly_score"
    assert not any(d.run(np.full(200, d.mu0)))
    alarms = d.run(np.full(50, d.mu0 + 3 * d.sigma))
    assert any(alarms)
    assert d.last_direction in ("+", None)
    d.reset()


@pytest.mark.parametrize("cls", [CUSUM, EWMA])
@pytest.mark.parametrize("bad", [np.array([1.0, 2.0]), [1.0], np.zeros((2, 2))])
def test_both_detectors_reject_vectors(cls, bad):
    d = cls("s").fit_reference([0.0, 1.0, 2.0])
    with pytest.raises(TypeError):
        d.update(bad)
    with pytest.raises(TypeError):
        d.fit_reference(np.zeros((5, 2)))


def test_ewma_recursion_by_hand():
    d = EWMA("s", lam=0.5, L=3.0).fit_reference([-1.0, 1.0])  # mu0 = 0
    d.update(1.0)
    assert d.z == pytest.approx(0.5)
    d.update(1.0)
    assert d.z == pytest.approx(0.75)


def test_ewma_limits_grow_towards_steady_state():
    d = EWMA("s", lam=0.2, L=3.0).fit_reference([-1.0, 1.0])
    widths = []
    for _ in range(50):
        d.update(d.mu0)
        widths.append(d.limit())
    assert all(a <= b for a, b in zip(widths, widths[1:]))
    steady = 3.0 * d.sigma * np.sqrt(0.2 / 1.8)
    assert widths[-1] == pytest.approx(steady, rel=1e-6)


def test_ewma_fires_after_injected_shift_and_not_before():
    rng = np.random.default_rng(0)
    d = EWMA("s").fit_reference(rng.normal(0.0, 1.0, 500))
    series = np.concatenate([np.full(50, d.mu0), np.full(50, d.mu0 + 2 * d.sigma)])
    alarms = d.run(series)
    assert alarms.index(True) >= 50


def test_ewma_fires_downward():
    rng = np.random.default_rng(0)
    d = EWMA("s").fit_reference(rng.normal(0.0, 1.0, 500))
    for x in np.full(50, d.mu0 - 3 * d.sigma):
        if d.update(x):
            break
    assert d.last_direction == "-"


def test_ewma_alarm_restarts_statistic():
    d = EWMA("s").fit_reference([-1.0, 1.0])
    for x in np.full(50, 10.0):
        if d.update(x):
            break
    assert d.z == d.mu0 and d.i == 0


def test_ewma_parameter_validation():
    with pytest.raises(ValueError):
        EWMA("s", lam=0.0)
    with pytest.raises(ValueError):
        EWMA("s", L=-1.0)
    with pytest.raises(RuntimeError):
        EWMA("s").update(0.0)
