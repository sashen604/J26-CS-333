"""EWMA drift detector: the required baseline (p4_robustness/CLAUDE.md).

Same interface as drift.cusum.CUSUM (fit_reference, update, reset, run,
last_direction), so the two can be swapped in experiments.

    z_i = lam * x_i + (1 - lam) * z_{i-1},   z_0 = mu0
    limits: mu0 +/- L * sigma * sqrt(lam / (2 - lam) * (1 - (1 - lam)^(2i)))

An alarm fires when z_i leaves the limits. These are the time-varying
(exact) limits, which are narrower for the first few steps after a restart.

The defaults lam=0.2 and L=3.0 are common textbook choices, NOT locked
decisions; tune them for the experiment and report what was used.
After an alarm, z restarts at mu0 and the step counter at 0.
"""
import math

from drift.base import as_scalar, as_scalar_series, check_name

DEFAULT_LAMBDA = 0.2
DEFAULT_L = 3.0


class EWMA:
    def __init__(self, monitored, lam=DEFAULT_LAMBDA, L=DEFAULT_L, auto_reset=True):
        if not 0 < lam <= 1 or L <= 0:
            raise ValueError("lam must be in (0, 1] and L must be > 0")
        self.monitored = check_name(monitored)
        self.lam = float(lam)
        self.L = float(L)
        self.auto_reset = auto_reset
        self.mu0 = None
        self.sigma = None
        self.z = None
        self.i = 0
        self.last_direction = None

    def fit_reference(self, values):
        """Estimate mu0 (mean) and sigma (sample std, ddof=1) from an in-control reference window."""
        arr = as_scalar_series(values)
        if len(arr) < 2:
            raise ValueError("Reference window needs at least 2 values to estimate sigma")
        sigma = float(arr.std(ddof=1))
        if sigma == 0.0:
            raise ValueError("Reference window has zero variance; sigma cannot be estimated")
        self.mu0 = float(arr.mean())
        self.sigma = sigma
        self.reset()
        return self

    def limit(self):
        """Current half-width of the control limits around mu0."""
        self._require_fit()
        lam = self.lam
        return self.L * self.sigma * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * self.i)))

    def update(self, x):
        """Feed one scalar. Returns True if an alarm fired at this step."""
        self._require_fit()
        x = as_scalar(x)
        self.i += 1
        self.z = self.lam * x + (1 - self.lam) * self.z
        width = self.limit()
        if self.z > self.mu0 + width:
            self.last_direction = "+"
        elif self.z < self.mu0 - width:
            self.last_direction = "-"
        else:
            self.last_direction = None
            return False
        if self.auto_reset:
            self.reset(self.last_direction)
        return True

    def reset(self, direction=None):
        """Restart the statistic at mu0. `direction` is accepted for interface parity with CUSUM."""
        if direction not in (None, "+", "-"):
            raise ValueError("direction must be None, '+' or '-'")
        self._require_fit()
        self.z = self.mu0
        self.i = 0

    def run(self, values):
        """Feed a 1-D series one value at a time. Returns a list of booleans, one per step."""
        return [self.update(x) for x in as_scalar_series(values)]

    def _require_fit(self):
        if self.mu0 is None:
            raise RuntimeError("Call fit_reference() before using the detector")
