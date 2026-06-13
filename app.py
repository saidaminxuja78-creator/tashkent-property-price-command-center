from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import (
    load_raw_data, clean_real_estate_data, evaluate_models, market_summary,
    PREFERRED_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
    predict_single, save_model, model_feature_importance, bootstrap_metric_ci,
    dataset_fingerprint, environment_metadata
)

APP_TITLE = "Tashkent Property Price Command Center"
DATA_PATH = Path("olx_massive_real_estate.xlsx")

st.set_page_config(page_title=APP_TITLE, page_icon="🏙️", layout="wide")

st.markdown(
    """
<style>
:root { --navy:#0f172a; --teal:#0ea5a4; --blue:#2563eb; --muted:#64748b; --card:#ffffff; --line:#e2e8f0; }
.block-container { padding-top: 1.1rem; max-width: 1320px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#f8fafc 0%,#eef6ff 100%); border-right:1px solid #dbeafe; }
.hero { padding: 2.2rem 2.4rem; border-radius: 28px; background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 48%,#0f766e 100%); color:white; box-shadow: 0 24px 65px rgba(15,23,42,.22); margin-bottom: 1.2rem; }
.hero h1 { font-size: 2.5rem; margin: .25rem 0 1rem; letter-spacing: -.04em; }
.hero p { font-size: 1.08rem; line-height: 1.7; color:#e0f2fe; max-width: 1080px; }
.kicker { color:#a5f3fc; font-weight:800; letter-spacing:.18em; text-transform:uppercase; font-size:.78rem; }
.pill { display:inline-block; padding:.45rem .75rem; border-radius:999px; border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.11); margin:.18rem .18rem .18rem 0; font-size:.82rem; }
.card { background:#fff; border:1px solid #e2e8f0; border-radius:20px; padding:1.1rem 1.25rem; box-shadow:0 12px 30px rgba(15,23,42,.06); margin-bottom:1rem; }
.card h3 { margin-top:0; color:#0f172a; }
.metric-card { background:#fff; border:1px solid #e2e8f0; border-radius:18px; padding:1rem; min-height:104px; box-shadow:0 10px 25px rgba(15,23,42,.04); }
.metric-label { color:#64748b; font-size:.86rem; }
.metric-value { color:#0f172a; font-size:1.65rem; font-weight:800; margin-top:.35rem; }
.good { color:#047857; font-weight:700; }
.warn { color:#b45309; font-weight:700; }
.bad { color:#b91c1c; font-weight:700; }
.workflow { display:flex; gap:.6rem; flex-wrap:wrap; margin:.8rem 0; }
.step { padding:.55rem .85rem; border-radius:14px; background:#eef6ff; border:1px solid #bfdbfe; color:#1e3a8a; font-weight:700; font-size:.86rem; }
.small-muted { color:#64748b; font-size:.9rem; }
hr { border:none; border-top:1px solid #e2e8f0; margin: 1.2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


def fmt_money(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:,.2f} bn UZS"
    return f"{x/1_000_000:,.1f} mln UZS"


def fmt_num(x, digits=3):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


@st.cache_data(show_spinner="Loading and cleaning OLX real-estate data...")
def get_data():
    raw = load_raw_data(DATA_PATH)
    clean = clean_real_estate_data(raw)
    return raw, clean


@st.cache_resource(show_spinner="Training and evaluating candidate models...")
def run_experiment(mode: str, max_rows: int | None):
    _, clean = get_data()
    return evaluate_models(clean, mode=mode, max_rows=max_rows)


raw_df, df = get_data()
summary = market_summary(df)

# Sidebar
st.sidebar.markdown("## 🏙️ Property Price Command Center")
st.sidebar.markdown("**PDP University · BTEC Level 6 · 2026**")
st.sidebar.markdown("Eltezorov Doriyorbek · Group 22-305")
st.sidebar.markdown("---")

sections = {
    "Operate": ["🏠 Overview", "📊 Data Audit", "🤖 Model Lab", "📈 Results"],
    "Use": ["💰 Price Prediction", "🎛️ Scenario Simulator", "🧭 Market Insights"],
    "Trust": ["🧪 Validation", "🔎 Explainability", "📋 Evidence Room"],
}
choices = []
for group, pages in sections.items():
    st.sidebar.markdown(f"**{group.upper()}**")
    for p in pages:
        choices.append(p)
page = st.sidebar.radio("Navigation", choices, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Clean rows: {summary['rows']:,}")
st.sidebar.caption(f"Features: {summary['features']}")
st.sidebar.caption("Target: apartment asking price in UZS")
st.sidebar.caption(f"Fingerprint: {dataset_fingerprint(df)[:12]}")

# Status bar
exp = st.session_state.get("experiment")
if exp is None:
    best_name = "Run training"
    best_mae = "—"
    best_r2 = "—"
else:
    lb = exp["leaderboard"]
    best_name = exp["best_model_name"]
    best_mae = fmt_money(lb.iloc[0]["MAE"])
    best_r2 = fmt_num(lb.iloc[0]["R2"])

st.markdown(
    f"""
<div class="card" style="display:flex;gap:1rem;align-items:center;justify-content:space-between;padding:.75rem 1rem;position:sticky;top:0;z-index:2;">
  <div><b>🟢 Property Price Lab</b></div>
  <div><b>Experiment:</b> {'Ready' if exp else 'Not trained'}</div>
  <div><b>Best model:</b> {best_name}</div>
  <div><b>MAE:</b> {best_mae}</div>
  <div><b>R²:</b> {best_r2}</div>
</div>
""",
    unsafe_allow_html=True,
)


def render_overview():
    st.markdown(
        """
<div class="hero">
  <div class="kicker">Pearson BTEC Level 6 · Applied AI Capstone</div>
  <h1>Tashkent Property Price Command Center</h1>
  <p>A reproducible real-estate intelligence prototype for estimating apartment asking prices from OLX listing data. The system audits market data, trains regression models, compares them with a baseline, explains drivers of price and supports scenario-based valuation decisions.</p>
  <span class="pill">OLX Uzbekistan dataset</span><span class="pill">Leakage-aware pipeline</span><span class="pill">MAE / RMSE / R²</span><span class="pill">Permutation explanations</span><span class="pill">Scenario simulator</span><span class="pill">Evidence export</span>
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    vals = [
        ("Listings after cleaning", f"{summary['rows']:,}"),
        ("Districts", f"{summary['districts']:,}"),
        ("Median price", fmt_money(summary["median_price"])),
        ("Median price/m²", fmt_money(summary["median_ppsqm"])),
        ("Candidate models", "5 + baseline"),
    ]
    for col, (label, value) in zip(cols, vals):
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

    angle = st.radio("Presentation angle", ["Executive story", "Technical proof", "Viva defence"], horizontal=True)
    if angle == "Executive story":
        text = "The app turns messy property-listing data into an interpretable valuation workflow: clean data → train models → compare errors → explain prices → simulate property changes."
    elif angle == "Technical proof":
        text = "The system avoids the original notebook's leakage-prone target encoding by keeping preprocessing inside a scikit-learn Pipeline and measuring performance on a held-out test set plus cross-validation."
    else:
        text = "For viva: this is not a black-box price guesser. It is an auditable prototype with a baseline, model comparison, error diagnostics, explainability and limitations."
    st.markdown(f"<div class='card'><h3>Project narrative</h3><p>{text}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='workflow'><div class='step'>1 · Data audit</div><div class='step'>2 · Feature engineering</div><div class='step'>3 · Model lab</div><div class='step'>4 · Error diagnostics</div><div class='step'>5 · Scenario simulation</div><div class='step'>6 · Evidence export</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, .9])
    with c1:
        st.markdown("### Why this project is stronger than a basic dashboard")
        st.write("A simple dashboard only visualises prices. This prototype creates a controlled machine-learning workflow: it cleans real scraped data, separates identifiers from predictive features, compares multiple models against a baseline and gives an evidence pack that can be defended academically.")
    with c2:
        fig = px.histogram(df, x=TARGET, nbins=50, title="Distribution of cleaned apartment asking prices")
        fig.update_layout(height=330, xaxis_title="Price (UZS)", yaxis_title="Listings")
        st.plotly_chart(fig, width="stretch")


def render_data_audit():
    st.title("📊 Data Audit")
    st.write("This page shows how raw OLX listings become a model-ready valuation dataset.")
    tabs = st.tabs(["Raw data", "Cleaning result", "Market distributions", "Quality flags"])
    with tabs[0]:
        st.dataframe(raw_df.head(100), width="stretch")
        st.caption(f"Raw shape: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
    with tabs[1]:
        st.dataframe(df.head(100), width="stretch")
        st.download_button("Download cleaned dataset CSV", df.to_csv(index=False).encode("utf-8"), "clean_olx_real_estate.csv", "text/csv")
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x="district", y=TARGET, title="Price by district")
            fig.update_layout(height=470, xaxis_tickangle=-40)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.scatter(df.sample(min(4000, len(df)), random_state=42), x="total_area", y=TARGET, color="number_of_rooms", hover_data=["district"], title="Area versus asking price")
            fig.update_layout(height=470)
            st.plotly_chart(fig, width="stretch")
    with tabs[3]:
        missing = raw_df.isna().mean().sort_values(ascending=False).head(15).reset_index()
        missing.columns = ["Column", "MissingRate"]
        st.dataframe(missing, width="stretch")
        st.markdown("""
        **Important audit decisions**
        - Price text such as `900 097 500 сум` is converted into numeric UZS.
        - Area values such as `63 м²` are parsed into square metres.
        - Identifiers, URLs and seller names are excluded from modelling.
        - Extreme price-per-square-metre records are filtered using percentile-based rules.
        - Missing values are not manually filled before model evaluation; imputation occurs inside the pipeline.
        """)


def render_model_lab():
    st.title("🤖 Model Lab")
    mode = st.radio("Computation mode", ["quick", "full"], horizontal=True, format_func=lambda x: "Quick — faster demo" if x == "quick" else "Full — more robust")
    max_rows = st.slider("Training sample cap", 5000, int(min(len(df), 45000)), int(min(len(df), 20000 if mode == "quick" else len(df))), step=1000)
    st.info("Quick mode is best for live demonstration. Full mode uses more rows and stronger ensembles, but takes longer on Streamlit Cloud.")
    if st.button("🚀 Train and evaluate models", type="primary", width="stretch"):
        st.session_state["experiment"] = run_experiment(mode, max_rows=max_rows)
        st.success("Training completed. Open Results, Validation and Price Prediction pages.")
    if st.session_state.get("experiment"):
        exp = st.session_state["experiment"]
        lb = exp["leaderboard"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", exp["best_model_name"])
        c2.metric("MAE", fmt_money(lb.iloc[0]["MAE"]))
        c3.metric("R²", fmt_num(lb.iloc[0]["R2"]))
        st.dataframe(lb, width="stretch")


def require_experiment():
    if st.session_state.get("experiment") is None:
        st.warning("Run Model Lab first. Training results are session-based.")
        st.stop()
    return st.session_state["experiment"]


def render_results():
    st.title("📈 Results")
    exp = require_experiment()
    lb = exp["leaderboard"].copy()
    tabs = st.tabs(["Leaderboard", "Error analysis", "Actual vs predicted", "Evidence export"])
    with tabs[0]:
        st.dataframe(lb, width="stretch")
        fig = px.bar(lb, x="Model", y="MAE", color="Model", title="Lower MAE is better")
        fig.update_layout(showlegend=False, height=430)
        st.plotly_chart(fig, width="stretch")
    with tabs[1]:
        pred_df = exp["predictions"]
        model = st.selectbox("Model", sorted(pred_df["Model"].unique()))
        sub = pred_df[pred_df["Model"] == model]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", fmt_money(sub["AbsoluteError"].mean()))
        c2.metric("Median APE", f"{sub['APE'].median():.1f}%")
        c3.metric("Within 20%", f"{(sub['APE'] <= 20).mean()*100:.1f}%")
        fig = px.histogram(sub, x="APE", nbins=60, title=f"Absolute percentage error distribution — {model}")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")
    with tabs[2]:
        best = exp["best_model"]
        X_test, y_test = exp["X_test"], exp["y_test"]
        y_pred = best.predict(X_test)
        plot_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        fig = px.scatter(plot_df, x="Actual", y="Predicted", title="Actual versus predicted prices")
        fig.add_trace(go.Scatter(x=[plot_df.Actual.min(), plot_df.Actual.max()], y=[plot_df.Actual.min(), plot_df.Actual.max()], mode="lines", name="Perfect prediction"))
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
    with tabs[3]:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("leaderboard.csv", lb.to_csv(index=False))
            z.writestr("fold_metrics.csv", exp["fold_metrics"].to_csv(index=False))
            z.writestr("predictions.csv", exp["predictions"].to_csv(index=False))
            z.writestr("environment.json", json.dumps(environment_metadata(), indent=2))
        st.download_button("Download evidence pack ZIP", buf.getvalue(), "house_price_evidence_pack.zip", "application/zip")


def render_prediction():
    st.title("💰 Price Prediction")
    exp = require_experiment()
    model = exp["best_model"]
    st.write("Enter a property profile. The model returns an estimated asking price, not a guaranteed sale price.")
    c1, c2, c3 = st.columns(3)
    with c1:
        district = st.selectbox("District", sorted(df["district"].dropna().unique()))
        type_of_market = st.selectbox("Market type", sorted(df["type_of_market"].dropna().unique()))
        house_type = st.selectbox("House type", sorted(df["house_type"].dropna().unique()))
    with c2:
        rooms = st.number_input("Rooms", 1, 12, 3)
        area = st.number_input("Total area, m²", 20.0, 400.0, 75.0, 1.0)
        living = st.number_input("Living area, m²", 10.0, 300.0, 55.0, 1.0)
    with c3:
        floor = st.number_input("Floor", 1, 60, 4)
        total_floors = st.number_input("Total floors", 1, 80, 9)
        repairs = st.selectbox("Repairs", sorted(df["repairs"].dropna().unique()))

    payload = {
        "region": "Ташкентская область", "district": district, "type_of_market": type_of_market,
        "number_of_rooms": rooms, "total_area": area, "total_living_area": living, "floor": floor,
        "total_floors": total_floors, "house_type": house_type, "building_age": 20,
        "wc": "Unknown", "furnished": "Нет", "repairs": repairs, "comission": "Нет", "layout": "Unknown",
        "ceiling_height": 2.8, "kitchen_area": 10, "photos_count": 8,
        "area_per_room": area / rooms, "floor_ratio": floor / total_floors,
        "is_top_floor": int(floor >= total_floors), "is_first_floor": int(floor == 1),
        "amenities_count": 3, "nearby_count": 3, "title_length": 60, "description_length": 450,
    }
    if st.button("Estimate asking price", type="primary"):
        pred = predict_single(model, payload)
        st.metric("Estimated asking price", fmt_money(pred))
        st.metric("Estimated price per m²", fmt_money(pred / area))
        st.caption("Responsible-use note: this estimate is a support signal for valuation discussion, not a formal appraisal.")
        rec = pd.DataFrame([{**payload, "predicted_price_uzs": pred}])
        st.download_button("Download prediction record", rec.to_csv(index=False).encode("utf-8"), "prediction_record.csv", "text/csv")


def render_scenario():
    st.title("🎛️ Scenario Simulator")
    exp = require_experiment()
    model = exp["best_model"]
    st.write("Compare how price changes under controlled property scenarios. This is not a causal claim; it is a model-based sensitivity simulation.")
    district = st.selectbox("Base district", sorted(df["district"].dropna().unique()), key="scenario_district")
    base_area = st.slider("Base area, m²", 30, 180, 75)
    rooms = st.slider("Rooms", 1, 6, 3)
    scenarios = [
        {"Scenario": "Current", "total_area": base_area, "rooms": rooms, "floor": 4, "repairs": "Средний"},
        {"Scenario": "+10 m² area", "total_area": base_area + 10, "rooms": rooms, "floor": 4, "repairs": "Средний"},
        {"Scenario": "Euro renovation", "total_area": base_area, "rooms": rooms, "floor": 4, "repairs": "Евроремонт"},
        {"Scenario": "Higher floor", "total_area": base_area, "rooms": rooms, "floor": 8, "repairs": "Средний"},
    ]
    rows = []
    for s in scenarios:
        payload = {
            "region": "Ташкентская область", "district": district, "type_of_market": "Вторичный рынок",
            "number_of_rooms": s["rooms"], "total_area": s["total_area"], "total_living_area": s["total_area"] * .72,
            "floor": s["floor"], "total_floors": 9, "house_type": "Панельный", "building_age": 25,
            "wc": "Unknown", "furnished": "Нет", "repairs": s["repairs"], "comission": "Нет", "layout": "Unknown",
            "ceiling_height": 2.8, "kitchen_area": 10, "photos_count": 8,
            "area_per_room": s["total_area"] / s["rooms"], "floor_ratio": s["floor"] / 9,
            "is_top_floor": int(s["floor"] >= 9), "is_first_floor": int(s["floor"] == 1),
            "amenities_count": 3, "nearby_count": 3, "title_length": 60, "description_length": 450,
        }
        pred = predict_single(model, payload)
        rows.append({"Scenario": s["Scenario"], "PredictedPrice": pred, "PricePerSqm": pred / s["total_area"]})
    sc = pd.DataFrame(rows)
    sc["DeltaVsCurrent"] = sc["PredictedPrice"] - sc.iloc[0]["PredictedPrice"]
    st.dataframe(sc, width="stretch")
    fig = px.bar(sc, x="Scenario", y="PredictedPrice", color="Scenario", title="Scenario-based predicted price")
    fig.update_layout(showlegend=False, height=430)
    st.plotly_chart(fig, width="stretch")


def render_market():
    st.title("🧭 Market Insights")
    by_d = df.groupby("district").agg(Listings=(TARGET, "size"), MedianPrice=(TARGET, "median"), MedianPriceSqm=("price_per_sqm", "median"), MedianArea=("total_area", "median")).reset_index().sort_values("MedianPrice", ascending=False)
    st.dataframe(by_d, width="stretch")
    fig = px.bar(by_d.head(12), x="district", y="MedianPriceSqm", color="MedianPriceSqm", title="Median asking price per m² by district")
    fig.update_layout(height=470, xaxis_tickangle=-35)
    st.plotly_chart(fig, width="stretch")


def render_validation():
    st.title("🧪 Validation")
    exp = require_experiment()
    best = exp["best_model"]
    X_test, y_test = exp["X_test"], exp["y_test"]
    y_pred = best.predict(X_test)
    ci = bootstrap_metric_ci(y_test, y_pred, n_boot=300)
    st.dataframe(ci, width="stretch")
    st.markdown("""
    **Validity interpretation.** The model is evaluated on a held-out test set and cross-validation folds. This supports internal validity, but external validity remains limited because OLX asking prices are not the same as final transaction prices and the dataset represents scraped listings rather than an official property registry.
    """)
    fig = px.histogram(pd.DataFrame({"Residual": y_test - y_pred}), x="Residual", nbins=60, title="Residual distribution")
    fig.update_layout(height=430)
    st.plotly_chart(fig, width="stretch")


def render_explainability():
    st.title("🔎 Explainability")
    exp = require_experiment()
    best = exp["best_model"]
    sample = st.slider("Rows for permutation importance", 300, min(3000, len(exp["X_test"])), min(1000, len(exp["X_test"])), step=100)
    if st.button("Calculate feature importance", type="primary"):
        Xs = exp["X_test"].sample(min(sample, len(exp["X_test"])), random_state=42)
        ys = exp["y_test"].loc[Xs.index]
        imp = model_feature_importance(best, Xs, ys, n_repeats=4)
        st.session_state["importance"] = imp
    if "importance" in st.session_state:
        imp = st.session_state["importance"]
        st.dataframe(imp, width="stretch")
        fig = px.bar(imp.head(15), x="Importance_MAE_Increase", y="Feature", orientation="h", title="Permutation feature importance")
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")


def render_evidence():
    st.title("📋 Evidence Room")
    st.markdown("""
    ### BTEC evidence mapping
    - **P1/P2:** clear aim and significance: valuation support for Uzbek real-estate listings.
    - **M1/D1:** alternative approaches considered: rule-based calculator, direct notebook, black-box model, and deployed decision-support command center.
    - **P3/P4/M2/D2:** project workflow, GitHub versioning, Streamlit deployment and iterative redesign.
    - **P5/M3/M4/D3:** data audit, regression evaluation, error analysis, validity and reliability discussion.
    - **P6/M5:** professional dashboard, evidence export and viva-ready explanation.
    - **P7/D4:** reflection should discuss technical learning, deployment issues and responsible AI thinking.
    """)
    st.markdown("### What makes this project different")
    st.write("The app is not just a house price form. It is a command-center workflow with audit, modelling, validation, scenario simulation and evidence export.")
    st.json(environment_metadata())


if page == "🏠 Overview":
    render_overview()
elif page == "📊 Data Audit":
    render_data_audit()
elif page == "🤖 Model Lab":
    render_model_lab()
elif page == "📈 Results":
    render_results()
elif page == "💰 Price Prediction":
    render_prediction()
elif page == "🎛️ Scenario Simulator":
    render_scenario()
elif page == "🧭 Market Insights":
    render_market()
elif page == "🧪 Validation":
    render_validation()
elif page == "🔎 Explainability":
    render_explainability()
elif page == "📋 Evidence Room":
    render_evidence()
