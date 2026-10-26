"""Tests for drift/cusum.py and the shared scalar checks in drift/base.py."""
import numpy as np
import pytest

from drift.base import as_scalar, as_scalar_series
from drift.cusum import CUSUM, DEFAULT_H, DEFAULT_K


@pytest.fixture
def det():
    rng = np.random.default_rng(0)
    return CUSUM("test_stream").fit_reference(rng.normal(0.0, 1.0, 500))


def test_locked_defaults_are_half_sigma_and_five_sigma():
    assert (DEFAULT_K, DEFAULT_H) == (0.5, 5.0)
    d = CUSUM("s")
    assert (d.k, d.h) == (0.5, 5.0)


def test_k_and_h_are_multiples_of_sigma():
    d = CUSUM("s", k=1.0, h=4.0).fit_reference([0.0, 2.0, 4.0])  # mean 2, sample std 2
    assert d.mu0 == pytest.approx(2.0)
    assert d.sigma == pytest.approx(2.0)
    assert d.K == pytest.approx(2.0)
    assert d.H == pytest.approx(8.0)


def test_recursion_matches_formula_by_hand():
    d = CUSUM("s").fit_reference([-1.0, 1.0, -1.0, 1.0])  # mu0 = 0
    K = d.K
    d.update(1.0)
    assert d.c_plus == pytest.approx(1.0 - K)
    assert d.c_minus == 0.0
    d.update(1.0)
    assert d.c_plus == pytest.approx(2 * (1.0 - K))
    d.update(-3.0)
    assert d.c_plus == pytest.approx(max(0.0, -3.0 - K + 2 * (1.0 - K)))
    assert d.c_minus == pytest.approx(max(0.0, -K + 3.0))


def test_quiet_when_stream_sits_at_reference_mean(det):
    assert not any(det.run(np.full(1000, det.mu0)))
    assert det.c_plus == 0.0 and det.c_minus == 0.0


def test_quiet_on_seeded_stationary_data(det):
    rng = np.random.default_rng(1)
    assert not any(det.run(rng.normal(0.0, 1.0, 100)))


def test_fires_after_injected_upward_mean_shift_and_not_before(det):
    rng = np.random.default_rng(2)
    before = rng.normal(0.0, 1.0, 50)
    after = rng.normal(2.0, 1.0, 50)  # +2 sigma shift at index 50
    first, direction = None, None
    for i, x in enumerate(np.concatenate([before, after])):
        if det.update(x):
            first, direction = i, det.last_direction
            break
    assert first is not None and first >= 50
    assert direction == "+"


def test_fires_downward_on_negative_shift(det):
    alarms = det.run(np.full(30, det.mu0 - 2 * det.sigma))
    assert any(alarms)
    first = alarms.index(True)
    # Each step adds exactly 2*sigma - K = 1.5*sigma to C-, so it crosses 5*sigma on step 4.
    assert first == 3


def test_post_alarm_reset_sets_opposite_accumulator_to_half_h(det):
    for x in np.full(20, det.mu0 + 3 * det.sigma):
        if det.update(x):
            break
    assert det.last_direction == "+"
    assert det.c_plus == 0.0
    assert det.c_minus == pytest.approx(0.5 * det.H)


def test_reset_direction_minus_and_plain_reset(det):
    det.reset("-")
    assert det.c_minus == 0.0 and det.c_plus == pytest.approx(0.5 * det.H)
    det.reset()
    assert det.c_plus == 0.0 and det.c_minus == 0.0
    with pytest.raises(ValueError):
        det.reset("up")


def test_auto_reset_off_leaves_accumulator_above_h(det):
    d = CUSUM("s", auto_reset=False).fit_reference([-1.0, 1.0, -1.0, 1.0])
    for x in np.full(20, 3.0):
        if d.update(x):
            break
    assert d.c_plus > d.H


@pytest.mark.parametrize("bad", [np.array([1.0, 2.0]), [1.0], np.zeros((2, 2)), np.array([1.0]), "1.0", None])
def test_update_rejects_non_scalars(det, bad):
    with pytest.raises(TypeError):
        det.update(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_update_rejects_non_finite(det, bad):
    with pytest.raises(ValueError):
        det.update(bad)


def test_update_accepts_numpy_scalars(det):
    det.update(np.float32(0.1))
    det.update(np.array(0.2))
    det.update(3)


def test_fit_reference_rejects_matrix_and_bad_windows():
    d = CUSUM("s")
    with pytest.raises(TypeError):
        d.fit_reference(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        d.fit_reference([1.0])
    with pytest.raises(ValueError):
        d.fit_reference([2.0, 2.0, 2.0])


def test_must_fit_before_update():
    with pytest.raises(RuntimeError):
        CUSUM("s").update(0.0)


def test_monitored_stream_must_be_named():
    with pytest.raises(ValueError):
        CUSUM("")


def test_base_helpers():
    assert as_scalar(np.int64(4)) == 4.0
    with pytest.raises(TypeError):
        as_scalar(True)
    assert as_scalar_series([1, 2, 3]).shape == (3,)
    with pytest.raises(TypeError):
        as_scalar_series("abc")
