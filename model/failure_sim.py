"""
failure_sim.py - device-level sensor-failure simulator for wearable HAR.
v0.2 (frozen). Trace-driven generation (D4); parametric fallback (D1/D3).
Empirical pools are loaded from the run/gap CSVs produced in Phase 0/1.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats as st

SEED = 42
FS = {"pamap2": 100, "opportunity": 30}

CATASTROPHIC = {("S2-ADL5", "Accelerometer RH"),
                ("S1-Drill", "Accelerometer RWR"),
                ("S4-ADL3", "Accelerometer LH")}

PARAMETRIC_FALLBACK = {
    "pamap2": {"off": ("pareto", {"alpha": 2.072, "xmin": 0.5}),
               "on": ("lognormal", {"mu": 3.79, "sigma": 2.182})},
    "opportunity": {"off": ("splice", {"mu": 3.39, "sigma": 0.7154,
                                       "u": 121, "alpha": 2.459,
                                       "w_tail": 0.060}),
                    "on": ("lognormal", {"mu": 4.786, "sigma": 2.24})},
}

BURST_PARAMS = None  # set by make_sim / set_pools


def load_pools(output_dir):
    p_runs = pd.read_csv(f"{output_dir}/pamap2_run_lengths.csv")
    o_runs = pd.read_csv(f"{output_dir}/opportunity_run_lengths.csv")
    o_b = o_runs[(o_runs["kind"] == "Accelerometer") & (~o_runs["boundary"])]
    o_b = o_b[~o_b.apply(lambda r: (r["session"], r["device"])
                         in CATASTROPHIC, axis=1)]
    p_gaps = pd.read_csv(f"{output_dir}/pamap2_gaps.csv")
    o_gaps = pd.read_csv(f"{output_dir}/opportunity_gaps.csv")
    return {
        "pamap2": {"off": p_runs["run_samples"].to_numpy(int),
                   "on":  p_gaps["gap_samples"].to_numpy(int)},
        "opportunity": {"off": o_b["run_samples"].to_numpy(int),
                        "on":  o_gaps["gap_samples"].to_numpy(int)},
    }


def set_pools(pools):
    global BURST_PARAMS
    BURST_PARAMS = {
        c: {"fs": FS[c],
            "off": ("empirical", {"pool": pools[c]["off"]}),
            "on":  ("empirical", {"pool": pools[c]["on"]})}
        for c in pools}


def _draw(dist, prm, rng, size):
    """Draw `size` positive integer durations (samples)."""
    if dist == "empirical":
        return rng.choice(prm["pool"], size=size)
    if dist == "pareto":
        x = st.pareto(b=prm["alpha"], scale=prm["xmin"]).rvs(
            size, random_state=rng)
    elif dist == "lognormal":
        x = st.lognorm(s=prm["sigma"], scale=np.exp(prm["mu"])).rvs(
            size, random_state=rng)
    elif dist == "splice":
        tail = rng.random(size) < prm["w_tail"]
        x = np.empty(size)
        body = st.lognorm(s=prm["sigma"], scale=np.exp(prm["mu"]))
        qb = rng.random(int((~tail).sum())) * body.cdf(prm["u"])
        x[~tail] = body.ppf(qb)
        x[tail] = st.pareto(b=prm["alpha"], scale=prm["u"]).rvs(
            int(tail.sum()), random_state=rng)
    return np.maximum(np.round(x).astype(int), 1)


@dataclass
class FailureSim:
    regime: str                  # burst | persistent | gradual | iid
    calibration: str = "pamap2"
    mode: str = "empirical"      # empirical (D4 default) | parametric
    rate_scale: float = 1.0
    iid_p: float = 0.1
    persistent_onset: tuple = (0.0, 1.0)
    gradual_window_s: float = 60.0
    seed: int = SEED
    events: list = field(default_factory=list)

    def _durations(self, which):
        if self.mode == "empirical":
            return BURST_PARAMS[self.calibration][which]
        return PARAMETRIC_FALLBACK[self.calibration][which]

    def mask(self, n_samples, n_devices, affected=None):
        """True = available. affected: device indices hit by the regime
        (default: all for burst/iid; one random device otherwise)."""
        rng = np.random.default_rng(self.seed)
        m = np.ones((n_samples, n_devices), dtype=bool)
        fs = BURST_PARAMS[self.calibration]["fs"]
        if affected is None:
            affected = (range(n_devices) if self.regime in ("burst", "iid")
                        else [rng.integers(n_devices)])
        for d in affected:
            if self.regime == "iid":
                m[:, d] = rng.random(n_samples) >= self.iid_p
            elif self.regime == "burst":
                off_dist, off_prm = self._durations("off")
                on_dist, on_prm = self._durations("on")
                t = 0
                while t < n_samples:
                    on = int(_draw(on_dist, on_prm, rng, 1)[0])
                    on = max(int(on / self.rate_scale), 1)
                    off = int(_draw(off_dist, off_prm, rng, 1)[0])
                    m[t + on: t + on + off, d] = False
                    self.events.append((int(d), int(t + on), int(off)))
                    t += on + off
            elif self.regime == "persistent":
                lo, hi = self.persistent_onset
                onset = int(rng.uniform(lo, hi) * n_samples)
                m[onset:, d] = False
                self.events.append((int(d), onset, n_samples - onset))
            elif self.regime == "gradual":
                w = int(self.gradual_window_s * fs)
                onset = int(rng.uniform(0.1, 0.9) * n_samples)
                span = min(w, n_samples - onset)
                ramp = np.arange(span) / max(w, 1)
                drop = np.zeros(n_samples, dtype=bool)
                drop[onset:onset + span] = rng.random(span) < ramp
                m[drop, d] = False
                m[onset + w:, d] = False
                self.events.append((int(d), onset, n_samples - onset))
        return m


def make_sim(regime, calibration, output_dir, **kw):
    """Factory: loads empirical pools from CSVs and returns a FailureSim."""
    set_pools(load_pools(output_dir))
    return FailureSim(regime, calibration=calibration, **kw)
