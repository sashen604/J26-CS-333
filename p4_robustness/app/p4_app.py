"""Streamlit demonstration UI for Pillar 4 (PP1 demo).

Run from the repo root:  .venv\\Scripts\\python.exe -m streamlit run p4_robustness/app/p4_app.py

Four tabs, all on the real saved proxy model and the local mirror data:
  1. Overview:          what Pillar 4 does and the three committed findings.
  2. Data audit:        window-size leakage audit (experiments/leakage_audit.py), PASS/FAIL vs data/README.md.
  3. Explain an alert:  SHAP and LIME on one test row; "Explain again" shows LIME's run-to-run change.
  4. Drift monitor:     one Scenario C stream with the committed calibrated CUSUM h and EWMA L.

Every number is computed live or read from committed results files (results/*.md, data/README.md
figures in leakage_audit.README_EXPECTED); none is typed in here. The helper functions above
main() are plain Python, so tests can import this module without running Streamlit.

Data: "CICIoT2023 subset (HF mirror)"; see data/README.md. Model: XGBoost proxy, not P3's AI model.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

P4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P4_DIR))
sys.path.insert(0, str(P4_DIR / "experiments"))

import data_prep  # noqa: E402
import demo  # noqa: E402
import drift_experiment as de  # noqa: E402
import leakage_audit  # noqa: E402
import xai_experiment as xe  # noqa: E402
from xai.metrics import top_k  # noqa: E402

RESULTS_DIR = P4_DIR / "results"
DRIFT_RESULTS = RESULTS_DIR / "drift_results.md"
XAI_RESULTS = RESULTS_DIR / "xai_results.md"
SUMMARY = RESULTS_DIR / "SUMMARY.md"

BANNER = ("Prototype. HF mirror features, not official CICIoT2023. "
          "Proxy model stands in for P2's AI model.")
K = xe.K
MIRAI_DEFAULT_MIN_P = 0.99   # default Mirai row: first with P(Mirai) above this
BENIGN_DEFAULT_MAX_P = 0.01  # default benign row: first with P(Mirai) below this
FACTOR_MIN, FACTOR_MAX = de.SCALE_START, de.SCALE_END


# ---------------------------------------------------------------------------
# Helpers (no Streamlit; tested in tests/test_p4_app.py)
# ---------------------------------------------------------------------------
def common_top_k_share(top_sets, k=K):
    """Share of top-k features common to every run: |T_1 ∩ ... ∩ T_N| / k (E-XAI formula). None if no runs."""
    if not top_sets:
        return None
    common = set(top_sets[0])
    for t in top_sets[1:]:
        common &= set(t)
    return len(common) / k


def committed_h_L(path=DRIFT_RESULTS):
    """(h, L) of the calibrated CUSUM and EWMA rows in drift_results.md."""
    cal = demo.committed_calibration(path)
    if "CUSUM" not in cal or "EWMA" not in cal:
        raise ValueError(f"{path.name} has no calibrated CUSUM and EWMA rows")
    return (float(cal["CUSUM"][0].split("=", 1)[1]), float(cal["EWMA"][0].split("=", 1)[1]))


def build_scenario_c_frame(benign_df, features, seed, max_factor=FACTOR_MAX, window=de.WINDOW):
    """drift_experiment.build_scenario_c with the ramp ending at max_factor instead of 2.0.

    Same steps and same random draws: sample benign rows without replacement, multiply
    `features` by the per-window factor (1.0 for 200 windows, then linear 1.0 -> max_factor
    over 100), re-round QUANTISED_FEATURES. At max_factor = 2.0 the output equals
    build_scenario_c exactly (tested). Returns (frame, per-row factor).
    """
    if max_factor < 1.0:
        raise ValueError("max_factor must be >= 1.0 (multiplicative drift, 1.0 = no drift)")
    factors = de.scenario_c_factors(end=max_factor)
    n_rows = len(factors) * window
    if n_rows > len(benign_df):
        raise ValueError("Not enough benign rows to sample Scenario C without replacement")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(benign_df), n_rows, replace=False)
    frame = benign_df.iloc[idx].reset_index(drop=True).copy()
    row_factor = np.repeat(factors, window)
    for f in features:
        frame[f] = frame[f] * row_factor
    return data_prep.quantise(frame), row_factor


def run_drift_stream(model, benign_df, features, ref, h, L, seed, max_factor=FACTOR_MAX):
    """One Scenario C stream, scored by CUSUM (h) and EWMA (L) with the shared reference."""
    frame, _ = build_scenario_c_frame(benign_df, features, seed, max_factor)
    stream = de.window_logits(model.predict_proba(frame)[:, 1])
    out = {"stream": stream, "seed": seed, "max_factor": max_factor}
    for kind, param in (("cusum", h), ("ewma", L)):
        alarms = de.alarm_indices(kind, ref, param, stream)
        delay, false_alarms, missed = de.score_run(alarms)
        out[kind] = {"param": param, "alarms": alarms, "delay": delay, "false_alarms": false_alarms,
                     "alarms_from_200": len(alarms) - false_alarms}
    return out


def readme_check_rows(measured, expected=leakage_audit.README_EXPECTED):
    """[(figure, measured, data/README.md, 'PASS'/'FAIL')] for every documented audit figure."""
    return [(k, str(measured[k]), str(v), "PASS" if measured[k] == v else "FAIL") for k, v in expected.items()]


def summary_section(heading, path=SUMMARY):
    """Markdown body of the '## <heading>...' section of SUMMARY.md (up to the next '## ')."""
    text = path.read_text(encoding="utf-8")
    start = text.index("\n## " + heading)
    body = text[start + 1:].split("\n", 1)[1]
    return body.split("\n## ", 1)[0].strip()


def first_row_where(p_mirai, mask_fn):
    """Position of the first row whose P(Mirai) satisfies mask_fn, else 0."""
    hits = np.flatnonzero(mask_fn(np.asarray(p_mirai)))
    return int(hits[0]) if len(hits) else 0


# ---------------------------------------------------------------------------
# Cached model, data and explainers
# ---------------------------------------------------------------------------
def _cached():
    import streamlit as st

    @st.cache_resource(show_spinner="Loading the saved proxy model...")
    def model():
        from proxy_model import ProxyFusionModel
        return ProxyFusionModel.load()

    @st.cache_resource(show_spinner="Loading the mirror data...")
    def splits():
        return {s: data_prep.load_split(s) for s in ("train", "test", "validation")}

    @st.cache_resource(show_spinner="Scoring the test split...")
    def test_by_class():
        m, test = model(), splits()["test"]
        out = {}
        for cls, label in ((0, "benign"), (1, "Mirai")):
            rows = test[test[data_prep.TARGET] == cls]
            out[label] = (rows, m.predict_proba(rows)[:, 1])
        return out

    @st.cache_resource
    def benign_medians():
        rows, _ = test_by_class()["benign"]
        return rows[model().feature_names].median()

    @st.cache_resource(show_spinner="Building the explainers...")
    def explainers():
        from xai.explainers import make_lime_explainer, make_shap_explainer
        lime_ref = xe.sample_rows(splits()["train"], xe.N_LIME_REFERENCE, xe.SEED)
        # LIME is seeded once at construction (as in xai_experiment.py) and never reseeded per call.
        return make_shap_explainer(model()), make_lime_explainer(model(), lime_ref, xe.SEED), lime_ref

    @st.cache_resource(show_spinner="Fitting the drift reference...")
    def drift_setup():
        m, val, test = model(), splits()["validation"], splits()["test"]
        val_benign = m.predict_proba(val[val[data_prep.TARGET] == 0])[:, 1]
        ref = de.fit_reference(de.window_logits(val_benign))
        benign_df = test[test[data_prep.TARGET] == 0].reset_index(drop=True)
        return ref, benign_df, de.select_scale_safe(de.gain_shares(m))

    return model, splits, test_by_class, benign_medians, explainers, drift_setup


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] { font-size: 20px; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: 1.2rem; line-height: 1.5; }
button[data-baseweb="tab"] p { font-size: 1.35rem; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 2.4rem; }
[data-testid="stMetricLabel"] p { font-size: 1.1rem; }
[data-testid="stCaptionContainer"], .stCaption p { font-size: 1.05rem !important; }
[data-testid="stDataFrame"] { font-size: 1.05rem; }
.block-container { padding-top: 2rem; max-width: 1500px; }
</style>
"""


def _big(chart):
    return (chart.configure_axis(labelFontSize=17, titleFontSize=19, labelLimit=600)
            .configure_legend(labelFontSize=17, titleFontSize=17, symbolSize=200)
            .configure_title(fontSize=21))


def banner(st):
    st.warning(BANNER, icon="⚠️")


def tab_overview(st):
    banner(st)
    st.header("What Pillar 4 does")
    st.markdown(
        "- **Real alerts:** checks that the detector's alerts come from attack behaviour, not from an "
        "artefact of how the data was captured.\n"
        "- **Clear reasons:** explains each alert by the features that pushed it towards Mirai.\n"
        "- **Detector health:** watches the model's output over time and raises an alarm when normal "
        "traffic starts to look different.")

    st.header("Three findings (committed results)")
    fig = leakage_audit.README_EXPECTED
    cal = demo.committed_calibration(DRIFT_RESULTS)
    scen_c = demo.committed_scenario_c(DRIFT_RESULTS)
    rep = demo.committed_repeatability(XAI_RESULTS)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. The raw data leaks the window size")
        st.markdown(
            f"`Number` is 10 for **{fig['number_benign_share_10']}** of benign rows and 100 for "
            f"**{fig['number_mirai_share_100']}** of Mirai rows; one split on it scores "
            f"**{fig['number_stump_accuracy']}**. The rule \"off the 0.1 grid → Mirai\" scores "
            f"**{fig['grid_rule_accuracy_before']}** before rounding and **{fig['grid_rule_accuracy_after']}** "
            f"after, the majority baseline ({fig['majority_baseline']}). Leaky features are dropped or rounded.")
        st.caption("Source: figures documented in data/README.md, checked live in the Data audit tab.")
    with c2:
        st.subheader("2. CUSUM and EWMA are tied")
        st.markdown(
            f"Both calibrated to {cal['CUSUM'][1]} false alarms per 1000 windows "
            f"(CUSUM {cal['CUSUM'][0]}, EWMA {cal['EWMA'][0]}). On slow drift in benign traffic "
            f"(Scenario C, 20 seeds) CUSUM detects in **{scen_c['CUSUM'][1]}** windows and EWMA in "
            f"**{scen_c['EWMA'][1]}**; missed {scen_c['CUSUM'][3]} and {scen_c['EWMA'][3]}.")
        st.caption("Source: results/drift_results.md.")
    with c3:
        st.subheader("3. SHAP repeats, LIME does not")
        st.markdown(
            f"Explaining the same row 3 times keeps the same top-5 for SHAP "
            f"(**{rep[('SHAP', 'benign')]}** benign, **{rep[('SHAP', 'Mirai')]}** Mirai) but not for LIME "
            f"(**{rep[('LIME', 'benign')]}** benign, **{rep[('LIME', 'Mirai')]}** Mirai). We report SHAP.")
        st.caption("Source: results/xai_results.md (E-XAI repeatability, 50 rows per class).")

    st.header("Planned for PP2")
    st.markdown(summary_section("6. Planned for PP2"))
    st.caption("Source: results/SUMMARY.md.")


def tab_audit(st, alt):
    banner(st)
    st.header("Data audit: does the raw data give the answer away?")
    st.markdown("Runs `experiments/leakage_audit.py` on the raw mirror test split (not rounded) and "
                "compares every figure with the values documented in `data/README.md`.")
    if st.button("Run audit", type="primary", key="run_audit"):
        t0 = time.perf_counter()
        with st.spinner("Running the leakage audit..."):
            r = leakage_audit.run_audit()
            raw_test = leakage_audit.load_raw("test")
            dist = leakage_audit.number_distribution(raw_test, top=10)
        st.session_state["audit"] = {"r": r, "dist": dist, "seconds": time.perf_counter() - t0}
    audit = st.session_state.get("audit")
    if audit is None:
        st.info("Press **Run audit** to compute the figures live.")
        return
    r, dist = audit["r"], audit["dist"]
    rows = readme_check_rows(leakage_audit.as_readme_figures(r))
    n_fail = sum(1 for *_, s in rows if s == "FAIL")
    if n_fail:
        st.error(f"FAIL: {n_fail} of {len(rows)} figures differ from data/README.md.")
    else:
        st.success(f"PASS: all {len(rows)} audit figures match data/README.md.")
    st.caption(f"Audit computed in {audit['seconds']:.1f} s.")

    st.subheader("1. Packets per window (`Number`) by class")
    df = pd.DataFrame([{"class": label, "Number": int(v), "share of rows": s}
                       for label, key in (("benign", "benign_top"), ("Mirai", "mirai_top"))
                       for v, s in dist[key].items()])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Number:O", title="Number (packets in the capture window)"),
        y=alt.Y("share of rows:Q", axis=alt.Axis(format="%"), title="share of that class's rows"),
        color=alt.Color("class:N", scale=alt.Scale(domain=["benign", "Mirai"], range=["#4C78A8", "#E45756"])),
        xOffset="class:N",
        tooltip=["class", "Number", alt.Tooltip("share of rows:Q", format=".2%")],
    ).properties(height=380)
    st.altair_chart(_big(chart), width="stretch")
    st.caption(f"Benign windows are almost always 10 packets ({dist['benign_share_10']:.2%} of "
               f"{dist['n_benign']:,} rows) and Mirai windows 100 packets ({dist['mirai_share_100']:.2%} of "
               f"{dist['n_mirai']:,}). One split on `Number` alone scores {r['stump'][0]:.4f} test accuracy, "
               "so this feature measures the capture setup, not the attack. It is dropped.")

    st.subheader("2. `Tot sum` is just `Number × AVG`")
    ratio = pd.DataFrame(r["totsum_ratio"], index=[f"p{p}" for p in leakage_audit.PERCENTILES]).T
    ratio.index = ["benign", "Mirai"]
    st.dataframe(ratio.style.format("{:.4f}"), width="content")
    st.caption("Tot sum ÷ (Number × AVG) at the 1st, 50th and 99th percentiles. A value of 1.0000 everywhere "
               "means `Tot sum` carries the window size too, so it is dropped as well.")

    st.subheader("3. The 0.1 grid rule, before and after rounding")
    bars = pd.DataFrame({"data": ["before rounding", "after rounding"],
                         "accuracy": [r["grid_before"], r["grid_after"]]})
    base = alt.Chart(bars).mark_bar(size=110).encode(
        x=alt.X("data:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="test accuracy of the rule"),
        color=alt.Color("data:N", legend=None, scale=alt.Scale(range=["#E45756", "#4C78A8"])))
    labels = base.mark_text(dy=-14, fontSize=20).encode(text=alt.Text("accuracy:Q", format=".4f"))
    line = alt.Chart(pd.DataFrame({"y": [r["majority"]]})).mark_rule(
        color="black", strokeDash=[8, 5], size=2).encode(y="y:Q")
    line_label = alt.Chart(pd.DataFrame({"y": [r["majority"]],
                                         "t": [f"majority-class baseline {r['majority']:.4f}"]})).mark_text(
        align="left", x=8, dy=-12, fontSize=17).encode(y="y:Q", text="t:N")
    st.altair_chart(_big((base + labels + line + line_label).properties(height=400)), width="stretch")
    on, n = r["benign_rows_on_grid"]
    st.caption(f"The rule \"any of 28 features off the 0.1 grid → Mirai\". Before rounding it scores "
               f"{r['grid_before']:.4f}, because benign means are over 10 packets ({on:,} of {n:,} benign rows sit on "
               f"the 0.1 grid). After rounding both classes to 0.1 it scores {r['grid_after']:.4f}, the same as always "
               "guessing the larger class (dashed line), so the leak is gone.")

    with st.expander("All audit figures vs data/README.md"):
        st.dataframe(pd.DataFrame(rows, columns=["figure", "measured", "data/README.md", "result"]),
                     width="stretch", hide_index=True)


def tab_explain(st, alt, model, test_by_class, benign_medians, explainers):
    from xai.explainers import lime_explain, shap_explain

    banner(st)
    st.header("Explain an alert")
    m = model()
    shap_exp, lime_exp, lime_ref = explainers()
    c1, c2 = st.columns([1, 2])
    with c1:
        cls = st.radio("Traffic class", ["Mirai", "benign"], horizontal=True, key="xai_class")
    rows, p_all = test_by_class()[cls]
    if cls == "Mirai":
        default = first_row_where(p_all, lambda p: p > MIRAI_DEFAULT_MIN_P)
        default_note = f"default: first Mirai row with P(Mirai) > {MIRAI_DEFAULT_MIN_P}"
    else:
        default = first_row_where(p_all, lambda p: p < BENIGN_DEFAULT_MAX_P)
        default_note = f"default: first benign row with P(Mirai) < {BENIGN_DEFAULT_MAX_P}"
    with c2:
        pos = st.number_input(f"Row among the {len(rows):,} {cls} test rows ({default_note})",
                              min_value=0, max_value=len(rows) - 1, value=default, step=1, key=f"xai_row_{cls}")
    row = rows.iloc[[int(pos)]]
    p = float(p_all[int(pos)])
    verdict = "Mirai" if p >= 0.5 else "benign"

    m1, m2, m3 = st.columns(3)
    m1.metric("Model verdict", verdict)
    m2.metric("P(Mirai)", f"{p:.4f}")
    m3.metric("True class", cls)

    sv = shap_explain(m, row, shap_exp)
    vals = sv.values[0]
    order = sorted(top_k(vals, K), key=lambda i: -abs(vals[i]))
    med = benign_medians()
    df = pd.DataFrame([{
        "feature": sv.feature_names[i],
        "label": f"{sv.feature_names[i]} = {row.iloc[0][sv.feature_names[i]]:g}  (benign median "
                 f"{med[sv.feature_names[i]]:g})",
        "SHAP": float(vals[i]),
        "direction": "towards Mirai" if vals[i] > 0 else "towards benign",
    } for i in order])
    st.subheader(f"Top {K} reasons (SHAP)")
    chart = alt.Chart(df).mark_bar(height=34).encode(
        y=alt.Y("label:N", sort=None, title=None),
        x=alt.X("SHAP:Q", title="SHAP value (log-odds of Mirai)"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["towards Mirai", "towards benign"],
                                                        range=["#E45756", "#4C78A8"]), title=None),
        tooltip=["feature", alt.Tooltip("SHAP:Q", format=".3f")],
    ).properties(height=70 * K)
    st.altair_chart(_big(chart), width="stretch")
    st.caption("Each bar is one feature's push on this decision: red pushes towards Mirai, blue towards benign. "
               "Next to each feature is this row's value and the median over benign test rows, so you can see how "
               "far from normal it is.")

    st.subheader("Is the explanation repeatable?")
    st.markdown("Each click explains **the same row** again with SHAP and with LIME (LIME's normal random "
                "sampling, not reseeded). If an explainer is repeatable, its top 5 never changes.")
    key = (cls, int(pos))
    hist = st.session_state.get("xai_history")
    if hist is None or hist["row"] != key:
        hist = {"row": key, "runs": []}
        st.session_state["xai_history"] = hist
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Explain again", type="primary", key="explain_again"):
        t0 = time.perf_counter()
        with st.spinner("Running SHAP and LIME..."):
            s = shap_explain(m, row, shap_exp).values[0]
            lime = lime_explain(m, row, lime_ref, k=K, explainer=lime_exp, num_samples=xe.LIME_SAMPLES, seed=None)
        shap_top = [sv.feature_names[i] for i in sorted(top_k(s, K), key=lambda i: -abs(s[i]))]
        hist["runs"].append({"run": len(hist["runs"]) + 1, "SHAP top 5": shap_top,
                             "LIME top 5": [f for f, _ in lime.top_k], "seconds": time.perf_counter() - t0})
    if b2.button("Reset", key="explain_reset"):
        hist["runs"] = []
    runs = hist["runs"]
    if not runs:
        st.info("Press **Explain again** a few times.")
        return
    o1, o2, o3 = st.columns(3)
    o1.metric("Runs so far", len(runs))
    if len(runs) >= 2:
        o2.metric("SHAP: top-5 shared by all runs", f"{common_top_k_share([r['SHAP top 5'] for r in runs]):.0%}")
        o3.metric("LIME: top-5 shared by all runs", f"{common_top_k_share([r['LIME top 5'] for r in runs]):.0%}")
    else:
        o2.metric("SHAP: top-5 shared by all runs", "needs 2 runs")
        o3.metric("LIME: top-5 shared by all runs", "needs 2 runs")
    table = pd.DataFrame([{"run": r["run"], "SHAP top 5": ", ".join(r["SHAP top 5"]),
                           "LIME top 5": ", ".join(r["LIME top 5"])} for r in runs])
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Each row is one run on the same input. The overlap is the share of the 5 top features that appear "
               "in every run so far (100% = identical every time). SHAP TreeExplainer is deterministic; LIME draws "
               f"new random samples each time. Last run took {runs[-1]['seconds']:.1f} s.")


def tab_drift(st, alt, model, drift_setup):
    banner(st)
    st.header("Drift monitor: Scenario C")
    st.markdown("Benign traffic only. For 200 windows nothing changes; then, over 100 windows, the five "
                "most important scale-safe features are multiplied by a factor rising from 1.0 to the "
                "maximum you choose. Each window is 50 rows. The detectors should alarm soon after window 200.")
    h, L = committed_h_L()
    c1, c2, c3 = st.columns([2, 1, 1])
    max_factor = c1.slider("Maximum drift factor", FACTOR_MIN, FACTOR_MAX, FACTOR_MAX, 0.05, key="drift_factor")
    seed = int(c2.number_input("Seed", min_value=0, value=0, step=1, key="drift_seed"))
    c3.markdown(f"**Calibrated (committed):**  \nCUSUM h = {h:g}  \nEWMA L = {L:g}")
    if st.button("Run stream", type="primary", key="run_stream"):
        t0 = time.perf_counter()
        with st.spinner("Building and scoring the stream..."):
            ref, benign_df, features = drift_setup()
            res = run_drift_stream(model(), benign_df, features, ref, h, L, seed, max_factor)
        res.update(ref=ref, features=features, seconds=time.perf_counter() - t0)
        st.session_state["drift"] = res
    res = st.session_state.get("drift")
    if res is not None:
        ref, stream = res["ref"], res["stream"]
        df = pd.DataFrame({"window": np.arange(len(stream)), "scalar": stream})
        line = alt.Chart(df).mark_line(color="#555", strokeWidth=2).encode(
            x=alt.X("window:Q", title="window (50 rows each)"),
            y=alt.Y("scalar:Q", title="logit of windowed P(attack)", scale=alt.Scale(zero=False)))
        drift_line = alt.Chart(pd.DataFrame({"x": [de.N_BEFORE]})).mark_rule(
            color="#E45756", strokeDash=[8, 5], size=2.5).encode(x="x:Q")
        drift_label = alt.Chart(pd.DataFrame({"x": [de.N_BEFORE], "t": ["drift starts"]})).mark_text(
            align="left", dx=6, y=14, fontSize=17, color="#E45756").encode(x="x:Q", text="t:N")
        mu = alt.Chart(pd.DataFrame({"y": [ref.mu0]})).mark_rule(color="#999", strokeDash=[2, 4]).encode(y="y:Q")
        alarms = pd.DataFrame(
            [{"window": i, "scalar": float(stream[i]), "detector": "CUSUM alarm"} for i in res["cusum"]["alarms"]]
            + [{"window": i, "scalar": float(stream[i]), "detector": "EWMA alarm"} for i in res["ewma"]["alarms"]])
        layers = line + drift_line + drift_label + mu
        if len(alarms):
            layers += alt.Chart(alarms).mark_point(size=260, filled=True).encode(
                x="window:Q", y="scalar:Q",
                color=alt.Color("detector:N", scale=alt.Scale(domain=["CUSUM alarm", "EWMA alarm"],
                                                               range=["#1f77b4", "#ff7f0e"]), title=None),
                shape=alt.Shape("detector:N", scale=alt.Scale(domain=["CUSUM alarm", "EWMA alarm"],
                                                               range=["triangle-down", "triangle-up"]), title=None))
        st.altair_chart(_big(layers.properties(height=430)), width="stretch")
        st.caption(f"The grey line is the one number both detectors watch: how attack-like each window looks to the "
                   f"model. The dotted grey line is its normal level (mean of benign validation windows, "
                   f"{ref.mu0:.2f}). Red dashed: drift starts at window {de.N_BEFORE}. Triangles: alarms. "
                   f"Seed {res['seed']}, factor 1.0 → {res['max_factor']:.2f}, features scaled: "
                   f"{', '.join(res['features'])}. Computed in {res['seconds']:.1f} s.")
        cols = st.columns(2)
        for col, kind, name, pname in ((cols[0], "cusum", "CUSUM", "h"), (cols[1], "ewma", "EWMA", "L")):
            r = res[kind]
            delay = "no alarm after 200" if r["delay"] is None else f"{r['delay']} windows"
            col.metric(f"{name} ({pname} = {r['param']:g}) detection delay", delay)
            col.metric(f"{name} false alarms before window 200", r["false_alarms"])
        if res["max_factor"] == 1.0:
            st.info(f"Factor 1.0 means no drift at all, so every alarm here is a false alarm: CUSUM raised "
                    f"{len(res['cusum']['alarms'])} and EWMA {len(res['ewma']['alarms'])} in {len(stream)} windows. "
                    "Both were calibrated to about 1 per 1000 windows.")
    else:
        st.info("Choose a factor and seed, then press **Run stream**.")

    st.subheader("Committed result for comparison (20 seeds, factor 2.0)")
    scen_c = demo.committed_scenario_c(DRIFT_RESULTS)
    st.dataframe(pd.DataFrame([{"detector": d, "parameter": p, "detection delay (windows)": delay,
                                "false alarms before window 200": fa, "missed": missed}
                               for d, (p, delay, fa, missed) in scen_c.items()]),
                 width="stretch", hide_index=True)
    st.caption("Mean ± std over seeds 0–19, from results/drift_results.md. One live run is one seed, so its delay "
               "can differ from the mean.")


def main():
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Pillar 4 demo", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Pillar 4: robustness, explanations and drift")
    model, _, test_by_class, benign_medians, explainers, drift_setup = _cached()
    t1, t2, t3, t4 = st.tabs(["Overview", "Data audit", "Explain an alert", "Drift monitor"])
    with t1:
        tab_overview(st)
    with t2:
        tab_audit(st, alt)
    with t3:
        tab_explain(st, alt, model, test_by_class, benign_medians, explainers)
    with t4:
        tab_drift(st, alt, model, drift_setup)


if __name__ == "__main__":
    main()
