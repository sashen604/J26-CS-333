"""Fast tests for experiments/drift_experiment.py. Synthetic scores only, no data or model needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import drift_experiment as de  # noqa: E402


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# Benign rows score 0.0 and Mirai rows score 1.0, so each window's mean P(attack)
# equals its Mirai share and can be read back from the logit stream.
BENIGN = np.zeros(20_000)
MIRAI = np.ones(15_000)


def shares(stream):
    return np.round(expit(stream), 4)


def test_scenario_b_ramp_rises_linearly_from_0_to_30_percent():
    counts = de.ramp_mirai_counts()
    assert len(counts) == 100
    assert counts[0] == 0
    assert counts[-1] == round(0.30 * 50)
    assert np.all(np.diff(counts) >= 0)
    expected = np.rint(np.linspace(0, 0.30, 100) * 50)
    assert np.array_equal(counts, expected)


def test_scenario_layouts():
    a, b = de.mirai_counts("A"), de.mirai_counts("B")
    assert len(a) == len(b) == 300
    assert a[:200].sum() == 0 and b[:200].sum() == 0
    assert np.all(a[200:] == 50)
    assert np.array_equal(b[200:], de.ramp_mirai_counts())
    assert de.mirai_counts("benign").sum() == 0
    with pytest.raises(ValueError):
        de.mirai_counts("C")


def test_built_stream_has_the_requested_mirai_share_per_window():
    counts = de.mirai_counts("B")
    s = de.build_stream(BENIGN, MIRAI, counts, seed=0)
    assert s.shape == (300,)
    got = shares(s)
    want = np.clip(counts / 50, de.CLIP_EPS, 1 - de.CLIP_EPS)
    assert np.allclose(got, np.round(want, 4))


def test_abrupt_scenario_is_all_mirai_after_drift():
    s = de.build_stream(BENIGN, MIRAI, de.mirai_counts("A"), seed=1)
    assert np.allclose(expit(s[:200]), de.CLIP_EPS)
    assert np.allclose(expit(s[200:]), 1 - de.CLIP_EPS)


def test_same_seed_same_stream_different_seed_different_rows():
    b = np.random.default_rng(0).uniform(0, 0.01, 20_000)
    counts = de.mirai_counts("benign")
    s0 = de.build_stream(b, MIRAI, counts, seed=0)
    assert np.array_equal(s0, de.build_stream(b, MIRAI, counts, seed=0))
    assert not np.array_equal(s0, de.build_stream(b, MIRAI, counts, seed=1))


def test_sampling_without_replacement_needs_enough_rows():
    with pytest.raises(ValueError):
        de.build_stream(np.zeros(100), MIRAI, de.mirai_counts("A"), seed=0)


def test_logit_is_clipped_and_finite():
    out = de.logit([0.0, 0.5, 1.0])
    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx(0.0)
    assert out[0] == pytest.approx(-out[2])


def test_window_logits_drops_incomplete_window():
    assert de.window_logits(np.full(120, 0.5), window=50).shape == (2,)


def test_sigma_floor_applied_only_when_needed():
    r = de.fit_reference([1.0, 1.0001, 0.9999])
    assert r.floor_used and r.sigma == de.SIGMA_FLOOR
    r = de.fit_reference([0.0, 1.0, 2.0])
    assert not r.floor_used and r.sigma == pytest.approx(1.0)


def test_score_run():
    assert de.score_run([50, 203, 250]) == (3, 1, False)
    assert de.score_run([10, 20]) == (None, 2, True)
    assert de.score_run([]) == (None, 0, True)


@pytest.mark.parametrize("kind", ["cusum", "ewma"])
def test_detectors_use_shared_reference_and_fire_on_scenario_a(kind):
    ref = de.Reference(mu0=0.0, sigma=1.0, measured_sigma=1.0, floor_used=False, n_windows=10)
    det = de.make_detector(kind, ref, 5.0 if kind == "cusum" else 3.0)
    assert (det.mu0, det.sigma) == (0.0, 1.0)
    stream = np.concatenate([np.zeros(200), np.full(100, 10.0)])
    delay, fa, missed = de.score_run(de.alarm_indices(kind, ref, 5.0 if kind == "cusum" else 3.0, stream))
    assert (delay, fa, missed) == (0, 0, False)


def test_calibration_hits_target_on_synthetic_noise():
    rng = np.random.default_rng(0)
    ref = de.Reference(0.0, 1.0, 1.0, False, 1000)
    streams = [rng.normal(0, 1, 300) for _ in range(20)]
    h, far = de.calibrate("cusum", ref, streams, target=0.01, iters=20)
    assert far == pytest.approx(0.01, abs=0.005)
    assert de.false_alarm_rate("cusum", ref, h * 2, streams) <= far
