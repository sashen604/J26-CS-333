"""Two-sided CUSUM drift detector on a single named scalar stream.

    C+_i = max[0, x_i - (mu0 + K) + C+_{i-1}]
    C-_i = max[0, (mu0 - K) - x_i + C-_{i-1}],    C+_0 = C-_0 = 0

Update rule from Estaji et al., "Evaluation of Drift Detection Algorithms in
the Condition Monitoring Domain", IEEE TII 21(1), 2025, eq. (6), p. 320.
K = k * sigma and decision interval H = h * sigma. An alarm fires when
C+ > H (upward shift) or C- > H (downward shift), as in Estaji p. 320.

Defaults (locked in p4_robustness/CLAUDE.md; change only after agreement):
- h = 5 (H = 5 sigma): Estaji, Sec. VI guideline 4, p. 324.
- k = 0.5 (K = 0.5 sigma): Estaji's guideline 4 (p. 324) sets K to half the
  target shift. With a 1 sigma target shift that gives K = 0.5 sigma, the
  standard CUSUM choice for a 1 sigma shift (Montgomery, Introduction to
  Statistical Quality Control). Estaji does not state k = 0.5 sigma directly.

After an alarm (applied by update() when auto_reset=True, or via reset(direction)):
- The OPPOSITE accumulator is set to 0.5 * H: Estaji, Sec. III-C, p. 320.
- The accumulator that fired restarts at 0: our design choice (the standard
  restart after a signal), not from Estaji.

Not implemented (future work), both from Estaji Sec. III-C, p. 320:
- the growth cap on C+ / C- with a margin of 10% of H;
- K-hat truncation (deviations larger than K-hat are truncated to K-hat).
"""
from drift.base import as_scalar, as_scalar_series, check_name

DEFAULT_K = 0.5  # multiples of sigma; half of a 1 sigma target shift (Estaji guideline 4; Montgomery)
DEFAULT_H = 5.0  # multiples of sigma; Estaji guideline 4, p. 324
POST_ALARM_FRACTION = 0.5  # opposite accumulator restarts at 0.5 * H (Estaji Sec. III-C, p. 320)


class CUSUM:
    def __init__(self, monitored, k=DEFAULT_K, h=DEFAULT_H, auto_reset=True):
        if k < 0 or h <= 0:
            raise ValueError("k must be >= 0 and h must be > 0 (both in multiples of sigma)")
        self.monitored = check_name(monitored)
        self.k = float(k)
        self.h = float(h)
        self.auto_reset = auto_reset
        self.mu0 = None
        self.sigma = None
        self.c_plus = 0.0
        self.c_minus = 0.0
        self.last_direction = None  # '+', '-' or None, for the most recent update

    @property
    def K(self):
        self._require_fit()
        return self.k * self.sigma

    @property
    def H(self):
        self._require_fit()
        return self.h * self.sigma

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

    def update(self, x):
        """Feed one scalar. Returns True if an alarm fired at this step."""
        self._require_fit()
        x = as_scalar(x)
        self.c_plus = max(0.0, x - (self.mu0 + self.K) + self.c_plus)
        self.c_minus = max(0.0, (self.mu0 - self.K) - x + self.c_minus)
        if self.c_plus > self.H:
            self.last_direction = "+"
        elif self.c_minus > self.H:
            self.last_direction = "-"
        else:
            self.last_direction = None
            return False
        if self.auto_reset:
            self.reset(self.last_direction)
        return True

    def reset(self, direction=None):
        """direction=None: both accumulators to 0 (fresh start, reference kept).
        direction='+' or '-': post-alarm restart. The side that fired goes to 0 (our design
        choice) and the opposite side to 0.5 * H (Estaji Sec. III-C, p. 320)."""
        if direction is None:
            self.c_plus = 0.0
            self.c_minus = 0.0
        elif direction == "+":
            self.c_plus = 0.0
            self.c_minus = POST_ALARM_FRACTION * self.H
        elif direction == "-":
            self.c_minus = 0.0
            self.c_plus = POST_ALARM_FRACTION * self.H
        else:
            raise ValueError("direction must be None, '+' or '-'")

    def run(self, values):
        """Feed a 1-D series one value at a time. Returns a list of booleans, one per step."""
        return [self.update(x) for x in as_scalar_series(values)]

    def _require_fit(self):
        if self.mu0 is None:
            raise RuntimeError("Call fit_reference() before using the detector")
