"""Streamlit dashboard for the repeat-purchase model.

    streamlit run app.py
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from repeat_purchase.config import (DATA_DIR, DECISION_THRESHOLD, FIGURES_DIR, INPUT_COLUMNS,
                                    METRICS_DIR, MODEL_PATH)
from repeat_purchase.explain import contributions
from repeat_purchase.predict import score

st.set_page_config(page_title="Repeat-Purchase Predictor", page_icon="🛒", layout="wide")

POSITIVE, NEGATIVE = "#2e7d32", "#c62828"


@st.cache_resource(show_spinner=False)
def get_model():
    import joblib
    if not MODEL_PATH.exists():
        from repeat_purchase.train import run
        with st.spinner("No trained model found - training one now (about 30 seconds)..."):
            run(make_plots=False)
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_summary():
    path = METRICS_DIR / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def sample_customers():
    return pd.read_csv(DATA_DIR / "sample_customers.csv")


model = get_model()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.05, 0.95, DECISION_THRESHOLD, 0.05,
                          help="Customers at or above this probability get a loyalty reward; "
                               "below it, a win-back offer. Lower it to catch more returners.")
    st.divider()
    st.markdown("**Model:** Logistic Regression  \n`C = 0.1`, `class_weight = balanced`")
    summary = get_summary()
    if summary:
        f = summary["final"]
        st.markdown(f"**Test ROC-AUC:** {f['ROC-AUC']:.3f}  \n**Recall:** {f['Recall']:.3f}  \n"
                    f"**F1-score:** {f['F1-Score']:.3f}")

st.title("🛒 Customer Repeat-Purchase Predictor")
st.caption("Estimates the chance that a customer buys again within the next 30 days "
           "and suggests the right campaign.")

tab_one, tab_batch, tab_perf = st.tabs(["Score a customer", "Score a CSV", "Model performance"])

# ------------------------------------------------------------------ single customer
with tab_one:
    base = sample_customers().iloc[0]
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Profile**")
            age = st.number_input("Age", 18, 90, int(base["age"]))
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            city_tier = st.selectbox("City tier", ["Tier-1", "Tier-2", "Tier-3"], index=0)
            membership = st.selectbox("Membership", ["None", "Silver", "Gold"], index=1)
        with c2:
            st.markdown("**Shopping behaviour**")
            channel = st.selectbox("Channel", ["In-store", "Online", "Both"], index=2)
            visits = st.number_input("Visits per month", 0.0, 30.0, float(base["visits_per_month"]), 0.5)
            basket = st.number_input("Average basket value (₹)", 50, 50000, int(base["avg_basket_value"]), 100)
            recency = st.slider("Days since last purchase", 1, 180, int(base["days_since_last_purchase"]))
        with c3:
            st.markdown("**Experience**")
            discount = st.slider("Discount usage %", 0.0, 100.0, float(base["discount_usage_pct"]), 1.0)
            satisfaction = st.slider("Satisfaction rating", 1, 5, int(base["satisfaction_rating"]))
            complaints = st.number_input("Complaints in last 6 months", 0, 20, int(base["complaints_last_6m"]))
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    customer = pd.DataFrame([{
        "age": age, "gender": gender, "city_tier": city_tier, "channel": channel,
        "membership": membership, "visits_per_month": visits, "avg_basket_value": basket,
        "days_since_last_purchase": recency, "discount_usage_pct": discount,
        "satisfaction_rating": satisfaction, "complaints_last_6m": complaints,
    }])
    result = score(customer, model=model, threshold=threshold).iloc[0]
    prob = float(result["repeat_probability"])

    st.divider()
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Repeat chance (30 days)", f"{prob:.1%}")
        st.progress(prob)
        if result["will_repeat"]:
            st.success(f"**Likely to return** → {result['recommended_action']}")
        else:
            st.warning(f"**At risk of lapsing** → {result['recommended_action']}")
        st.caption(f"Decision threshold: {threshold:.0%}")
    with right:
        st.markdown("**Why this prediction?**")
        contrib = contributions(model, customer).head(8)
        chart = alt.Chart(contrib).mark_bar().encode(
            x=alt.X("contribution:Q", title="Effect on prediction"),
            y=alt.Y("label:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            color=alt.Color("effect:N", scale=alt.Scale(domain=["raises chance", "lowers chance"],
                                                         range=[POSITIVE, NEGATIVE]),
                            legend=None),
            tooltip=[alt.Tooltip("label:N", title="Feature"),
                     alt.Tooltip("contribution:Q", title="Effect (log-odds)", format="+.2f")],
        ).properties(height=300)
        st.altair_chart(chart, width="stretch")
        st.caption("Compared with an average customer. Green bars push the chance up, red bars push it down.")

# ------------------------------------------------------------------ batch scoring
with tab_batch:
    st.markdown("Upload a CSV with one row per customer. Required columns:")
    st.code(", ".join(INPUT_COLUMNS), language=None)
    uploaded = st.file_uploader("Customer CSV", type="csv")
    use_sample = st.toggle("Use the bundled sample customers instead", value=uploaded is None)

    batch = None
    if uploaded is not None and not use_sample:
        batch = pd.read_csv(uploaded)
    elif use_sample:
        batch = sample_customers()

    if batch is not None:
        try:
            scored = score(batch, model=model, threshold=threshold)
        except ValueError as err:
            st.error(str(err))
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Customers scored", f"{len(scored):,}")
            k2.metric("Likely to return", f"{scored['will_repeat'].mean():.0%}")
            k3.metric("Average probability", f"{scored['repeat_probability'].mean():.1%}")

            hist = alt.Chart(scored).mark_bar(color="#5b8fd1").encode(
                x=alt.X("repeat_probability:Q", bin=alt.Bin(step=0.1), title="Repeat probability",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("count():Q", title="Customers"),
            ).properties(height=220)
            rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="black", strokeDash=[4, 4]) \
                .encode(x="t:Q")
            st.altair_chart(hist + rule, width="stretch")

            first = [c for c in ("customer_id", "repeat_probability", "will_repeat", "recommended_action")
                     if c in scored.columns]
            scored = scored[first + [c for c in scored.columns if c not in first]]
            st.dataframe(scored, width="stretch", hide_index=True,
                         column_config={"repeat_probability": st.column_config.ProgressColumn(
                             "Repeat probability", min_value=0, max_value=1, format="%.2f")})
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode(),
                               "scored_customers.csv", "text/csv")

# ------------------------------------------------------------------ performance
with tab_perf:
    summary = get_summary()
    if not summary:
        st.info("Run `python -m repeat_purchase.train` to generate performance metrics.")
    else:
        f = summary["final"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC", f"{f['ROC-AUC']:.3f}")
        m2.metric("Recall", f"{f['Recall']:.3f}")
        m3.metric("Precision", f"{f['Precision']:.3f}")
        m4.metric("F1-score", f"{f['F1-Score']:.3f}")
        m5.metric("Accuracy", f"{f['Accuracy']:.3f}")
        st.caption(f"Held-out test set of {summary['n_test']:,} customers; "
                   f"{summary['positive_rate']:.0%} of customers are repeat buyers.")

        st.subheader("All models compared")
        res = pd.DataFrame(summary["results"]).drop(columns=["Time (s)"]) \
            .sort_values("CV ROC-AUC", ascending=False)
        st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"})
                     .highlight_max(subset=["ROC-AUC", "CV ROC-AUC"],
                                    props=f"background-color: {POSITIVE}; color: white; font-weight: bold"),
                     width="stretch", hide_index=True)

        figs = [("confusion_roc.png", "Confusion matrix and ROC curves"),
                ("feature_importance.png", "Feature importance (tuned Random Forest)"),
                ("eda.png", "Exploratory analysis"), ("model_comparison.png", "Model comparison")]
        cols = st.columns(2)
        for i, (fname, cap) in enumerate(figs):
            path = FIGURES_DIR / fname
            if path.exists():
                cols[i % 2].image(str(path), caption=cap, width="stretch")
