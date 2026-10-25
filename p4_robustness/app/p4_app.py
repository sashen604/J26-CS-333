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

Data: "CICIoT2023 subset (HF mirror)"; see data/README.md. Model: XGBoost proxy, not P2's AI model.
"""
import html
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
# Rendering (presentation only: layout, styling and wording)
# ---------------------------------------------------------------------------
PRIMARY = "#1D6F8A"   # calm blue/teal: primary series (benign, CUSUM, SHAP)
SECOND = "#D08A1E"    # amber: second series (Mirai, EWMA, LIME)
GREY = "#94A3B8"      # baselines and reference lines
INK = "#334155"
MUTED = "#64748B"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

CSS = f"""
<style>
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp table {{ font-family: {FONT}; }}
.stApp {{ font-size: 15px; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ font-size: 15px; line-height: 1.55; }}
[data-testid="stMarkdownContainer"] li {{ margin-bottom: 0.15rem; }}
.stApp h1 {{ font-size: 26px !important; font-weight: 650 !important; }}
.stApp h2 {{ font-size: 20px !important; font-weight: 600 !important; }}
.stApp h3 {{ font-size: 18px !important; font-weight: 600 !important; padding: 0.4rem 0 0.1rem !important; }}
.stApp h4 {{ font-size: 15px !important; font-weight: 600 !important; padding: 0.2rem 0 0 !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {MUTED} !important;
    font-size: 13px !important; line-height: 1.45; }}
[data-testid="stMetricLabel"] p {{ font-size: 13px !important; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-size: 24px !important; font-weight: 600; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width: 1200px; padding: 1.2rem 1.5rem 2rem;
    margin: 0 auto; }}
[data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}
[data-testid="stTab"] p {{ font-size: 15px; font-weight: 600; }}
[data-testid="stTab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] p {{ color: {PRIMARY} !important; }}
[data-testid="stTab"] .react-aria-SelectionIndicator {{ background-color: {PRIMARY} !important; }}
[data-testid="stTabs"] {{ margin-top: 0.2rem; }}
button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background-color: {PRIMARY} !important;
    border-color: {PRIMARY} !important; color: #fff !important; }}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ filter: brightness(0.92); }}
[data-testid="stSlider"] [role="group"], [data-testid="stRadioOption"] > div:first-of-type {{
    filter: hue-rotate(195deg) saturate(0.55) brightness(0.8); }}  /* Streamlit's default red -> palette teal */
[data-testid="stDataFrame"] {{ font-size: 13px; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.p4-header {{ display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25); padding: 0.2rem 0 0.6rem; }}
.p4-title {{ font-size: 26px; font-weight: 650; letter-spacing: -0.01em; }}
.p4-sub {{ font-size: 13px; color: {MUTED}; }}
.p4-strip {{ font-size: 13px; color: {MUTED}; background: rgba(29, 111, 138, 0.07);
    border-left: 3px solid {PRIMARY}; border-radius: 4px; padding: 0.35rem 0.75rem; margin-top: 0.6rem; }}
.p4-card-label {{ font-size: 13px; color: {MUTED}; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.03em; }}
.p4-card-value {{ font-size: 24px; font-weight: 650; margin: 0.25rem 0 0.3rem; }}
.p4-card-context {{ font-size: 13px; color: {MUTED}; line-height: 1.45; }}
</style>
"""


def _style(chart, height):
    """Shared chart look: system font, 12-13px axis text, light gridlines, legend on top, no border."""
    return (chart.properties(height=height)
            .configure(font=FONT)
            .configure_axis(labelFontSize=12, titleFontSize=13, titleFontWeight="normal", labelColor=MUTED,
                            titleColor=INK, gridColor="#E5E7EB", gridOpacity=0.7, domainColor="#CBD5E1",
                            tickColor="#CBD5E1", labelLimit=400)
            .configure_legend(labelFontSize=12, titleFontSize=12, orient="top", labelColor=INK, symbolSize=90)
            .configure_view(strokeWidth=0))


def header(st):
    st.markdown(
        '<div class="p4-header"><div class="p4-title">Pillar 4 · Trustworthy Detection</div>'
        '<div class="p4-sub">IT4010 · J26-CS-333 · Kavish</div></div>'
        f'<div class="p4-strip">ⓘ&nbsp; {html.escape(BANNER)}</div>', unsafe_allow_html=True)


def card(st, label, value, context):
    with st.container(border=True):
        st.markdown(f'<div class="p4-card-label">{html.escape(label)}</div>'
                    f'<div class="p4-card-value">{html.escape(value)}</div>'
                    f'<div class="p4-card-context">{html.escape(context)}</div>', unsafe_allow_html=True)


def _mean(text):
    """'16.65 ± 10.38' -> '16.65' (display only: the committed text up to the ±)."""
    return text.split(" ±", 1)[0]


def tab_overview(st):
    st.subheader("What Pillar 4 does")
    st.markdown(
        "- **Real alerts:** the detector's alerts come from attack behaviour, not from how the data was captured.\n"
        "- **Clear reasons:** each alert is explained by the features that pushed it towards Mirai.\n"
        "- **Detector health:** the model's output is watched over time, with an alarm when normal traffic changes.")

    st.subheader("Key findings")
    fig = leakage_audit.README_EXPECTED
    cal = demo.committed_calibration(DRIFT_RESULTS)
    scen_c = demo.committed_scenario_c(DRIFT_RESULTS)
    rep = demo.committed_repeatability(XAI_RESULTS)
    c1, c2, c3 = st.columns(3)
    with c1:
        card(st, "Window-size leak removed",
             f"{fig['grid_rule_accuracy_before']} → {fig['grid_rule_accuracy_after']}",
             f"Accuracy of the \"off the 0.1 grid → Mirai\" rule before → after rounding; majority baseline "
             f"{fig['majority_baseline']}. Number alone scored {fig['number_stump_accuracy']}.")
    with c2:
        card(st, "Drift delay: CUSUM vs EWMA",
             f"{_mean(scen_c['CUSUM'][1])} vs {_mean(scen_c['EWMA'][1])}",
             f"Windows to detect slow benign drift (Scenario C, 20 seeds): {scen_c['CUSUM'][1]} vs "
             f"{scen_c['EWMA'][1]}. CUSUM {cal['CUSUM'][0]}, EWMA {cal['EWMA'][0]}, both at "
             f"{cal['CUSUM'][1]} false alarms per 1000 windows. Tied.")
    with c3:
        card(st, "Explanation repeatability",
             f"SHAP {_mean(rep[('SHAP', 'benign')])} vs LIME {_mean(rep[('LIME', 'benign')])}",
             f"Top-5 kept over 3 runs on the same row (benign / Mirai): SHAP {rep[('SHAP', 'benign')]} / "
             f"{rep[('SHAP', 'Mirai')]}, LIME {rep[('LIME', 'benign')]} / {rep[('LIME', 'Mirai')]}. We report SHAP.")
    st.caption("Committed results: data/README.md figures, results/drift_results.md, results/xai_results.md.")

    st.subheader("Planned for PP2")
    st.markdown(summary_section("6. Planned for PP2"))
    st.caption("Source: results/SUMMARY.md.")


def tab_audit(st, alt):
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("Data audit")
        st.caption("Does the raw data give the answer away? Runs experiments/leakage_audit.py on the raw "
                   "(unrounded) test split and checks every figure against data/README.md.")
        if st.button("Run audit", type="primary", key="run_audit", width="stretch"):
            t0 = time.perf_counter()
            with st.spinner("Running the leakage audit..."):
                r = leakage_audit.run_audit()
                raw_test = leakage_audit.load_raw("test")
                dist = leakage_audit.number_distribution(raw_test, top=10)
            st.session_state["audit"] = {"r": r, "dist": dist, "seconds": time.perf_counter() - t0}
        audit = st.session_state.get("audit")
        if audit is not None:
            rows = readme_check_rows(leakage_audit.as_readme_figures(audit["r"]))
            n_fail = sum(1 for *_, s in rows if s == "FAIL")
            if n_fail:
                st.error(f"FAIL: {n_fail} of {len(rows)} figures differ from data/README.md.")
            else:
                st.success(f"PASS: all {len(rows)} audit figures match data/README.md.")
            st.caption(f"Computed in {audit['seconds']:.1f} s.")
    with right:
        if audit is None:
            st.info("Press **Run audit** to compute the figures live.")
            return
        r, dist = audit["r"], audit["dist"]
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("#### Packets per window by class")
            df = pd.DataFrame([{"class": label, "Number": int(v), "share of rows": s}
                               for label, key in (("benign", "benign_top"), ("Mirai", "mirai_top"))
                               for v, s in dist[key].items()])
            chart = alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
                x=alt.X("Number:O", title="Number (packets in the capture window)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("share of rows:Q", axis=alt.Axis(format="%"), title="share of class rows"),
                color=alt.Color("class:N", title=None,
                                scale=alt.Scale(domain=["benign", "Mirai"], range=[PRIMARY, SECOND])),
                xOffset="class:N",
                tooltip=["class", "Number", alt.Tooltip("share of rows:Q", format=".2%")])
            st.altair_chart(_style(chart, 250), width="stretch")
            st.caption(f"Benign is 10 packets in {dist['benign_share_10']:.2%} of {dist['n_benign']:,} rows, Mirai "
                       f"100 in {dist['mirai_share_100']:.2%} of {dist['n_mirai']:,}; one split on Number scores "
                       f"{r['stump'][0]:.4f}. Dropped.")
        with c2:
            st.markdown("#### Off-grid rule, before vs after rounding")
            bars = pd.DataFrame({"data": ["before rounding", "after rounding"],
                                 "accuracy": [r["grid_before"], r["grid_after"]]})
            base = alt.Chart(bars).mark_bar(size=70, cornerRadiusEnd=2).encode(
                x=alt.X("data:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="test accuracy of the rule"),
                color=alt.Color("data:N", legend=None,
                                scale=alt.Scale(domain=["before rounding", "after rounding"], range=[SECOND, PRIMARY])))
            labels = base.mark_text(dy=-9, fontSize=13, color=INK).encode(text=alt.Text("accuracy:Q", format=".4f"))
            line = alt.Chart(pd.DataFrame({"y": [r["majority"]]})).mark_rule(
                color=GREY, strokeDash=[6, 4], size=1.5).encode(y="y:Q")
            st.altair_chart(_style(base + labels + line, 250), width="stretch")
            on, n = r["benign_rows_on_grid"]
            st.caption(f"\"Off the 0.1 grid → Mirai\": {r['grid_before']:.4f} before rounding ({on:,} of {n:,} benign "
                       f"rows on the grid), {r['grid_after']:.4f} after, equal to the majority baseline "
                       f"{r['majority']:.4f} (dashed line).")
        c3, c4 = st.columns([2, 3], gap="medium")
        with c3:
            st.markdown("#### Tot sum ÷ (Number × AVG)")
            ratio = pd.DataFrame(r["totsum_ratio"], index=[f"p{p}" for p in leakage_audit.PERCENTILES]).T
            ratio.index = ["benign", "Mirai"]
            st.dataframe(ratio.style.format("{:.4f}"), width="stretch")
            st.caption("1.0000 at p1, p50 and p99: Tot sum carries the window size too. Dropped.")
        with c4:
            st.markdown("#### Audit figures vs data/README.md")
            st.dataframe(pd.DataFrame(rows, columns=["figure", "measured", "data/README.md", "result"]),
                         width="stretch", hide_index=True, height=213)
            st.caption("Scroll for all 14 figures.")


def tab_explain(st, alt, model, test_by_class, benign_medians, explainers):
    from xai.explainers import lime_explain, shap_explain

    m = model()
    shap_exp, lime_exp, lime_ref = explainers()
    left, right = st.columns([1, 2.7], gap="large")
    with left:
        st.subheader("Explain an alert")
        cls = st.radio("Traffic class", ["Mirai", "benign"], horizontal=True, key="xai_class")
        rows, p_all = test_by_class()[cls]
        if cls == "Mirai":
            default = first_row_where(p_all, lambda p: p > MIRAI_DEFAULT_MIN_P)
            default_note = f"Default: first Mirai row with P(Mirai) > {MIRAI_DEFAULT_MIN_P}."
        else:
            default = first_row_where(p_all, lambda p: p < BENIGN_DEFAULT_MAX_P)
            default_note = f"Default: first benign row with P(Mirai) < {BENIGN_DEFAULT_MAX_P}."
        pos = st.number_input(f"Row (of {len(rows):,} {cls} test rows)", min_value=0, max_value=len(rows) - 1,
                              value=default, step=1, key=f"xai_row_{cls}", help=default_note)
        st.caption(default_note)
        again = st.button("Explain again", type="primary", key="explain_again", width="stretch")
        reset = st.button("Reset", key="explain_reset", width="stretch")
        st.caption("Explain again reruns SHAP and LIME on the same row; LIME is not reseeded.")
    row = rows.iloc[[int(pos)]]
    p = float(p_all[int(pos)])
    verdict = "Mirai" if p >= 0.5 else "benign"
    sv = shap_explain(m, row, shap_exp)
    vals = sv.values[0]
    order = sorted(top_k(vals, K), key=lambda i: -abs(vals[i]))
    med = benign_medians()

    with right:
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Model verdict", verdict)
            m2.metric("P(Mirai)", f"{p:.4f}")
            m3.metric("True class", cls)
        st.markdown(f"#### Top {K} reasons (SHAP)")
        df = pd.DataFrame([{
            "feature": sv.feature_names[i],
            "label": f"{sv.feature_names[i]} = {row.iloc[0][sv.feature_names[i]]:g}  (benign median "
                     f"{med[sv.feature_names[i]]:g})",
            "SHAP": float(vals[i]),
            "direction": "towards Mirai" if vals[i] > 0 else "towards benign",
        } for i in order])
        chart = alt.Chart(df).mark_bar(height=22, cornerRadiusEnd=2).encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("SHAP:Q", title="SHAP value (log-odds of Mirai)"),
            color=alt.Color("direction:N", title=None,
                            scale=alt.Scale(domain=["towards benign", "towards Mirai"], range=[PRIMARY, SECOND])),
            tooltip=["feature", alt.Tooltip("SHAP:Q", format=".3f")])
        st.altair_chart(_style(chart, alt.Step(38)), width="stretch")
        st.caption("Each bar is one feature's push on this decision (amber towards Mirai, blue towards benign), "
                   "with this row's value next to the benign median.")

    key = (cls, int(pos))
    hist = st.session_state.get("xai_history")
    if hist is None or hist["row"] != key:
        hist = {"row": key, "runs": []}
        st.session_state["xai_history"] = hist
    if again:
        t0 = time.perf_counter()
        with st.spinner("Running SHAP and LIME..."):
            s = shap_explain(m, row, shap_exp).values[0]
            lime = lime_explain(m, row, lime_ref, k=K, explainer=lime_exp, num_samples=xe.LIME_SAMPLES, seed=None)
        shap_top = [sv.feature_names[i] for i in sorted(top_k(s, K), key=lambda i: -abs(s[i]))]
        hist["runs"].append({"run": len(hist["runs"]) + 1, "SHAP top 5": shap_top,
                             "LIME top 5": [f for f, _ in lime.top_k], "seconds": time.perf_counter() - t0})
    if reset:
        hist["runs"] = []
    runs = hist["runs"]

    st.subheader("Is the explanation repeatable?")
    if not runs:
        st.info("Press **Explain again** a few times: a repeatable explainer keeps the same top 5 every run.")
        return
    o1, o2 = st.columns(2)
    if len(runs) >= 2:
        o1.metric(f"SHAP: top-5 shared by all {len(runs)} runs",
                  f"{common_top_k_share([r['SHAP top 5'] for r in runs]):.0%}", border=True)
        o2.metric(f"LIME: top-5 shared by all {len(runs)} runs",
                  f"{common_top_k_share([r['LIME top 5'] for r in runs]):.0%}", border=True)
    else:
        o1.metric("SHAP: top-5 shared by all runs", "needs 2 runs", border=True)
        o2.metric("LIME: top-5 shared by all runs", "needs 2 runs", border=True)
    table = pd.DataFrame([{"run": r["run"], "SHAP top 5": ", ".join(r["SHAP top 5"]),
                           "LIME top 5": ", ".join(r["LIME top 5"])} for r in runs])
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(f"Runs so far: {len(runs)}. Overlap = share of the top 5 present in every run (100% = identical). "
               f"SHAP TreeExplainer is deterministic; LIME samples afresh each run. Last run took "
               f"{runs[-1]['seconds']:.1f} s.")


def tab_drift(st, alt, model, drift_setup):
    st.subheader("Drift monitor · Scenario C")
    st.caption("Benign traffic only. Windows 0–199 are unchanged; over windows 200–299 the five most important "
               "scale-safe features are multiplied by a factor rising from 1.0 to the maximum below (50 rows per "
               "window). A healthy detector alarms soon after window 200.")
    h, L = committed_h_L()
    c1, c2, c3, c4 = st.columns([3, 1, 1.2, 2.2], gap="medium", vertical_alignment="bottom")
    max_factor = c1.slider("Maximum drift factor", FACTOR_MIN, FACTOR_MAX, FACTOR_MAX, 0.05, key="drift_factor")
    seed = int(c2.number_input("Seed", min_value=0, value=0, step=1, key="drift_seed"))
    run = c3.button("Run stream", type="primary", key="run_stream", width="stretch")
    c4.caption(f"Calibrated (committed): CUSUM h = {h:g} · EWMA L = {L:g}")
    if run:
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
        line = alt.Chart(df).mark_line(color=INK, strokeWidth=1.4, opacity=0.8).encode(
            x=alt.X("window:Q", title="window (50 rows each)", scale=alt.Scale(domain=[0, len(stream)])),
            y=alt.Y("scalar:Q", title="logit of windowed P(attack)", scale=alt.Scale(zero=False)))
        drift_line = alt.Chart(pd.DataFrame({"x": [de.N_BEFORE]})).mark_rule(
            color=MUTED, strokeDash=[6, 4], size=1.5).encode(x="x:Q")
        drift_label = alt.Chart(pd.DataFrame({"x": [de.N_BEFORE], "t": ["drift starts"]})).mark_text(
            align="left", dx=5, y=8, fontSize=12, color=MUTED).encode(x="x:Q", text="t:N")
        mu = alt.Chart(pd.DataFrame({"y": [ref.mu0]})).mark_rule(color=GREY, strokeDash=[2, 3]).encode(y="y:Q")
        alarms = pd.DataFrame(
            [{"window": i, "scalar": float(stream[i]), "detector": "CUSUM alarm"} for i in res["cusum"]["alarms"]]
            + [{"window": i, "scalar": float(stream[i]), "detector": "EWMA alarm"} for i in res["ewma"]["alarms"]])
        layers = line + drift_line + drift_label + mu
        if len(alarms):
            layers += alt.Chart(alarms).mark_point(size=110, filled=True, opacity=0.95).encode(
                x="window:Q", y="scalar:Q",
                color=alt.Color("detector:N", scale=alt.Scale(domain=["CUSUM alarm", "EWMA alarm"],
                                                               range=[PRIMARY, SECOND]), title=None),
                shape=alt.Shape("detector:N", scale=alt.Scale(domain=["CUSUM alarm", "EWMA alarm"],
                                                               range=["triangle-down", "triangle-up"]), title=None))
        st.altair_chart(_style(layers, 320), width="stretch")
        st.caption(f"Dark line: the scalar both detectors watch (how attack-like each window looks). Dotted: its "
                   f"normal level ({ref.mu0:.2f}, benign validation windows). Dashed: drift starts at window "
                   f"{de.N_BEFORE}. Triangles: alarms. Seed {res['seed']}, factor 1.0 → {res['max_factor']:.2f}; "
                   f"scaled: {', '.join(res['features'])}. Computed in {res['seconds']:.1f} s.")
        cols = st.columns(4)
        for i, (kind, name, pname) in enumerate((("cusum", "CUSUM", "h"), ("ewma", "EWMA", "L"))):
            r = res[kind]
            delay = "no alarm after 200" if r["delay"] is None else f"{r['delay']} windows"
            cols[2 * i].metric(f"{name} ({pname} = {r['param']:g}) detection delay", delay, border=True)
            cols[2 * i + 1].metric(f"{name} false alarms before window 200", r["false_alarms"], border=True)
        if res["max_factor"] == 1.0:
            st.info(f"Factor 1.0 means no drift at all, so every alarm here is a false alarm: CUSUM raised "
                    f"{len(res['cusum']['alarms'])} and EWMA {len(res['ewma']['alarms'])} in {len(stream)} windows. "
                    "Both were calibrated to about 1 per 1000 windows.")
    else:
        st.info("Choose a factor and seed, then press **Run stream**.")

    scen_c = demo.committed_scenario_c(DRIFT_RESULTS)
    st.caption("Committed result, 20 seeds at factor 2.0 (results/drift_results.md): "
               + "; ".join(f"{d} ({p}) delay {delay} windows, false alarms before 200 {fa}, missed {missed}"
                           for d, (p, delay, fa, missed) in scen_c.items())
               + ". One live run is one seed, so its delay can differ from the mean.")


def main():
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Pillar 4 · Trustworthy Detection", page_icon="🛡️", layout="wide",
                       menu_items={"Get help": None, "Report a bug": None, "About": None})
    st.markdown(CSS, unsafe_allow_html=True)
    header(st)
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
