"""
Britam Underwriting Risk Estimator
-----------------------------------
A minimal Streamlit front-end for the tuned Random Forest pipeline built in
underwriting.ipynb. Takes the same six raw intake fields the model was
trained on (age, sex, bmi, children, smoker, region) and returns an
expected-charges estimate, a rough pricing tier, and a flag for the
known model blind spot (older, high-BMI non-smokers).

HOW IT LOADS A MODEL (tries each in order):
  1. ./models/rf_underwriting_pipeline.pkl   <- the file your notebook saves
     (cell: joblib.dump(best_rf, 'models/rf_underwriting_pipeline.pkl'))
  2. A .pkl you upload in the sidebar
  3. A raw Kaggle-format CSV you upload in the sidebar, from which this
     app rebuilds the *exact same* pipeline (same preprocessor, same
     RandomForest hyperparameters, same random_state) live.

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


NAVY = "#1E2761"
TEAL = "#2A9D8F"
CORAL = "#E63946"
SLATE = "#5B6785"
CYAN = "#71BBF0"
OFFWHITE = "#F4F6FB"

MODEL_PATH = "models/rf_underwriting_pipeline.pkl"
RAW_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]

st.set_page_config(
    page_title="Britam Underwriting Risk Estimator",
    page_icon="\U0001F4CA",
    layout="wide",
)

# --------------------------------------------------------------------------
# Light CSS polish
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {OFFWHITE}; }}
    .metric-card {{
        background: white; border-radius: 10px; padding: 1.1rem 1.3rem;
        border: 1px solid #E1E6F2; box-shadow: 0 2px 8px rgba(30,39,97,0.08);
    }}
    h1, h2, h3 {{ color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model loading / training
# --------------------------------------------------------------------------
def build_pipeline():
    """Same architecture as `preprocessor_3` + tuned RF in the notebook."""
    categorical_cols = ["sex", "smoker", "region"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot_cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",  # age, bmi, children pass through untouched
    )
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42,
        )),
    ])
    return pipe


@st.cache_resource(show_spinner=False)
def train_from_csv(csv_bytes):
    """Rebuilds the exact notebook pipeline live from a raw Kaggle-format CSV."""
    df = pd.read_csv(csv_bytes)
    X = df[RAW_COLUMNS]
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=254, stratify=X["smoker"]
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
    }
    return pipe, metrics


@st.cache_resource(show_spinner=False)
def load_pickle(path_or_buffer):
    return joblib.load(path_or_buffer)


def get_feature_importance(pipe):
    reg = pipe.named_steps["regressor"]
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pd.Series(reg.feature_importances_, index=names).sort_values(ascending=False)
    # tidy labels
    pretty = {
        "onehot_cat__smoker_yes": "Smoker status",
        "remainder__bmi": "BMI",
        "remainder__age": "Age",
        "remainder__children": "Children",
        "onehot_cat__sex_male": "Sex",
    }
    imp.index = [pretty.get(i, i.replace("onehot_cat__region_", "Region: ")) for i in imp.index]
    return imp


# --------------------------------------------------------------------------
# Sidebar - model source
# --------------------------------------------------------------------------
st.sidebar.header("Model source")
st.sidebar.caption("Loaded your saved pipeline automatically!!")

model = None
train_metrics = None
source_note = ""

if os.path.exists(MODEL_PATH):
    model = load_pickle(MODEL_PATH)
    source_note = f"Loaded trained pipeline from `{MODEL_PATH}`"
else:
    uploaded_pkl = st.sidebar.file_uploader("Upload trained pipeline (.pkl)", type=["pkl"])
    uploaded_csv = st.sidebar.file_uploader("...or upload the raw dataset (.csv) to train live", type=["csv"])

    if uploaded_pkl is not None:
        model = load_pickle(uploaded_pkl)
        source_note = f"Loaded uploaded pipeline: {uploaded_pkl.name}"
    elif uploaded_csv is not None:
        with st.spinner("Training Random Forest (n_estimators=200, max_depth=4, min_samples_leaf=10)..."):
            model, train_metrics = train_from_csv(uploaded_csv)
        source_note = f"Trained live on uploaded dataset: {uploaded_csv.name}"

if model is None:
    st.title("Britam Underwriting Risk Estimator")
    st.warning(
        f"No model found. Place your saved pipeline at `{MODEL_PATH}` "
        "(the path your notebook already writes to), or use the uploaders "
        "in the sidebar to supply a `.pkl` or the raw `.csv`."
    )
    st.stop()

st.sidebar.success(source_note)
if train_metrics:
    st.sidebar.metric("Live-trained test R\u00B2", f"{train_metrics['R2']:.3f}")
    st.sidebar.metric("Live-trained test MAE", f"${train_metrics['MAE']:,.0f}")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Medical Insurance Underwriting Risk Estimator")
st.caption(
    "Enter a policyholder's intake details to get an expected annual charge estimate "
    "from the tuned Random Forest model (test R\u00B2 = 0.850, MAE = $2,606)."
)

left, right = st.columns([1, 1.4], gap="large")

# --------------------------------------------------------------------------
# LEFT - input form
# --------------------------------------------------------------------------
with left:
    st.subheader("Policyholder details")
    with st.form("intake_form"):
        age = st.slider("Age", 18, 80, 35)
        sex = st.radio("Sex", ["female", "male"], horizontal=True)
        bmi = st.number_input("BMI", min_value=14.0, max_value=55.0, value=27.5, step=0.1)
        children = st.slider("Number of children", 0, 10, 0)
        smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        submitted = st.form_submit_button("Estimate charges", use_container_width=True)

# --------------------------------------------------------------------------
# RIGHT - prediction + context
# --------------------------------------------------------------------------
with right:
    st.subheader("Estimate")

    if not submitted:
        st.info("Fill in the policyholder's details and click **Estimate charges**.")
    else:
        input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=RAW_COLUMNS)
        pred = float(model.predict(input_df)[0])

        # simple pricing tiers for demo purposes
        if pred < 8000:
            tier, tier_color = "Low", TEAL
        elif pred < 20000:
            tier, tier_color = "Medium", "#E9B44C"
        else:
            tier, tier_color = "High", CORAL

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="metric-card">
                <div style="color:{SLATE};font-size:0.85rem;">EXPECTED ANNUAL CHARGES</div>
                <div style="color:{NAVY};font-size:2.2rem;font-weight:700;">${pred:,.0f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="metric-card">
                <div style="color:{SLATE};font-size:0.85rem;">SUGGESTED PRICING TIER</div>
                <div style="color:{tier_color};font-size:2.2rem;font-weight:700;">{tier}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # blind-spot flag - ties back to the limitation slide
        if smoker == "no" and bmi >= 30 and age >= 40:
            st.markdown(
                f"""<div style="margin-top:1rem;padding:0.9rem 1.1rem;border-radius:8px;
                background:#FBEAEC;border:1px solid {CORAL};color:{NAVY};">
                <b>\u26A0 Known model blind spot:</b> older, high-BMI non-smokers are
                consistently <b>underpredicted</b> by this model in testing (by $19K&ndash;$25K
                on the worst cases). Treat this estimate as a floor, not a ceiling &mdash;
                recommend manual underwriting review for this profile.
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Why this estimate: feature importance")
        imp = get_feature_importance(model)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [CORAL if v == imp.max() else NAVY for v in imp.values]
        ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
        ax.set_xlabel("Importance")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Model: tuned Random Forest (n_estimators=200, max_depth=4, min_samples_leaf=10) \u00b7 "
    "trained on 1,337 policyholders \u00b7 test R\u00B2 = 0.850, MAE = $2,606, RMSE = $4,670."
)