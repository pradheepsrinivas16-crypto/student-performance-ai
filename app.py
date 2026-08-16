import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(content, unsafe_allow_html=True):
    cleaned = "\n".join(line.strip() for line in content.splitlines())
    st.html(cleaned)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

render_html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(91,79,255,.18), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(0,180,255,.12), transparent 25%),
        linear-gradient(135deg,#070b18 0%,#0b1025 50%,#071728 100%);
    color:#f5f7ff;
}

.block-container {
    padding-top:2rem;
    padding-bottom:3rem;
    max-width:1400px;
}

.hero {
    padding:38px 42px;
    border-radius:28px;
    margin-bottom:28px;
    background:linear-gradient(135deg,rgba(81,70,229,.90),rgba(0,130,210,.72));
    border:1px solid rgba(255,255,255,.15);
    box-shadow:0 20px 60px rgba(0,0,0,.35);
}

.hero-title {
    font-size:42px;
    font-weight:800;
    letter-spacing:-1px;
    margin-bottom:8px;
}

.hero-subtitle {
    font-size:17px;
    color:rgba(255,255,255,.85);
    line-height:1.6;
}

.badge {
    display:inline-block;
    margin-top:18px;
    padding:8px 14px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.18);
    font-size:13px;
    font-weight:600;
}

.section-title {
    font-size:27px;
    font-weight:800;
    margin:32px 0 18px 0;
}

.card-title {
    font-size:18px;
    font-weight:700;
    margin:28px 0 14px 0;
}

label {
    color:#dbe4ff !important;
    font-weight:600 !important;
}

div[data-baseweb="select"] > div {
    background-color:#252631 !important;
    border-color:rgba(255,255,255,.05) !important;
}

div[data-baseweb="input"] > div {
    background-color:#252631 !important;
    border-color:rgba(255,255,255,.05) !important;
}

input {
    color:#ffffff !important;
}

.stButton > button {
    width:100%;
    min-height:58px;
    border-radius:15px;
    border:none;
    background:linear-gradient(90deg,#6157ff,#009ee8);
    color:white;
    font-size:16px;
    font-weight:800;
    box-shadow:0 12px 30px rgba(55,100,255,.28);
    transition:all .2s ease;
}

.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 16px 36px rgba(55,100,255,.38);
}

.prediction-card {
    padding:34px;
    border-radius:26px;
    text-align:center;
    background:linear-gradient(135deg,rgba(34,41,85,.95),rgba(13,56,86,.95));
    border:1px solid rgba(90,180,255,.25);
    box-shadow:0 20px 50px rgba(0,0,0,.3);
    margin:25px 0;
}

.prediction-label {
    font-size:20px;
    font-weight:700;
    color:#dce8ff;
}

.prediction-score {
    font-size:64px;
    font-weight:800;
    margin:8px 0;
    background:linear-gradient(90deg,#ffffff,#75d9ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.prediction-sub {
    font-size:14px;
    color:#9fb1d3;
}

.score-gauge {
    margin:8px 0 24px;
}

.gauge-header {
    display:flex;
    justify-content:space-between;
    color:#9fb1d3;
    font-size:12px;
    margin-bottom:7px;
}

.gauge-track {
    height:13px;
    border-radius:999px;
    background:rgba(255,255,255,.08);
    overflow:hidden;
}

.gauge-fill {
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#ff5c6c,#ffc857,#3ddc97);
}

.gauge-percentage {
    text-align:center;
    margin-top:8px;
    color:#cbd6ee;
    font-weight:700;
}

.metric-card {
    background:rgba(19,25,43,.90);
    border:1px solid rgba(255,255,255,.08);
    border-radius:20px;
    padding:22px;
    min-height:145px;
}

.metric-icon {
    font-size:24px;
}

.metric-title {
    color:#9fb1d3;
    font-size:13px;
    margin-top:8px;
}

.metric-value {
    font-size:30px;
    font-weight:800;
    margin-top:6px;
}

.risk-low,
.risk-medium,
.risk-high {
    padding:22px 25px;
    border-radius:18px;
}

.risk-low {
    background:rgba(20,115,75,.22);
    border:1px solid rgba(70,220,145,.28);
}

.risk-medium {
    background:rgba(170,115,20,.22);
    border:1px solid rgba(255,190,70,.30);
}

.risk-high {
    background:rgba(160,40,55,.22);
    border:1px solid rgba(255,90,110,.30);
}

.risk-title {
    font-size:18px;
    font-weight:800;
    margin-bottom:8px;
}

.risk-description {
    color:#c4cce0;
    line-height:1.6;
}

.factor-card {
    background:rgba(20,25,42,.82);
    border:1px solid rgba(255,255,255,.07);
    border-radius:18px;
    padding:20px;
    min-height:180px;
}

.factor-title {
    font-size:16px;
    font-weight:800;
}

.factor-value {
    font-size:34px;
    font-weight:800;
    margin:10px 0;
}

.factor-description {
    color:#9eabc5;
    font-size:13px;
    line-height:1.6;
}

.profile-item {
    padding:8px 4px;
}

.profile-label {
    font-size:13px;
    color:#9fb1d3;
}

.profile-value {
    font-size:32px;
    font-weight:800;
    color:#f5f7ff;
    margin-top:5px;
}

.factor-snapshot {
    background:rgba(15,21,38,.78);
    border:1px solid rgba(255,255,255,.07);
    border-radius:20px;
    padding:24px 26px;
    margin-top:8px;
}

.snapshot-row {
    margin-bottom:22px;
}

.snapshot-row:last-child {
    margin-bottom:0;
}

.snapshot-header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:9px;
}

.snapshot-name {
    font-size:14px;
    font-weight:700;
    color:#dce7ff;
}

.snapshot-value {
    font-size:14px;
    font-weight:800;
    color:#72d8ff;
}

.snapshot-track {
    width:100%;
    height:9px;
    border-radius:999px;
    background:rgba(255,255,255,.08);
    overflow:hidden;
}

.snapshot-fill {
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#6157ff,#009ee8);
    box-shadow:0 0 12px rgba(0,158,232,.25);
}

.snapshot-scale {
    font-size:11px;
    color:#71809f;
    margin-top:5px;
}

.explain-card {
    background:rgba(15,21,38,.78);
    border:1px solid rgba(255,255,255,.07);
    border-radius:20px;
    padding:24px 26px;
}

.explain-row {
    margin-bottom:20px;
}

.explain-row:last-child {
    margin-bottom:0;
}

.explain-header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:8px;
}

.explain-name {
    font-size:14px;
    font-weight:700;
    color:#dce7ff;
}

.explain-value {
    font-size:13px;
    font-weight:800;
    color:#75d9ff;
}

.explain-track {
    height:9px;
    width:100%;
    background:rgba(255,255,255,.08);
    border-radius:999px;
    overflow:hidden;
}

.explain-fill {
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#6157ff,#00c6ff);
}

.explain-note {
    color:#71809f;
    font-size:11px;
    margin-top:5px;
}

.model-info {
    background:rgba(19,25,43,.55);
    border-top:1px solid rgba(255,255,255,.07);
    border-bottom:1px solid rgba(255,255,255,.07);
    padding:25px 5px;
}

.model-label {
    color:#71809f;
    font-size:12px;
    margin-bottom:7px;
}

.model-value {
    font-size:25px;
    font-weight:700;
    color:#eaf1ff;
}

.recommendation {
    padding:17px 20px;
    margin:10px 0;
    border-radius:14px;
    background:rgba(35,43,68,.72);
    border:1px solid rgba(255,255,255,.06);
    color:#f1f5ff;
}

.divider {
    height:1px;
    background:rgba(255,255,255,.08);
    margin:30px 0;
}

.footer {
    text-align:center;
    color:#71809f;
    padding:35px 10px 10px 10px;
    font-size:13px;
}

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0d1226,#080d1c);
}

section[data-testid="stSidebar"] h2 {
    color:#ffffff;
}

@media (max-width:768px) {

    .hero {
        padding:28px 24px;
    }

    .hero-title {
        font-size:32px;
    }

    .hero-subtitle {
        font-size:15px;
    }

    .prediction-score {
        font-size:52px;
    }

    .section-title {
        font-size:23px;
    }

}

</style>
""")


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "student_performance_model.pkl"
)

FEATURES_PATH = os.path.join(
    "models",
    "model_features.pkl"
)

METRICS_PATH = os.path.join(
    "models",
    "model_metrics.pkl"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.error(
            f"Model file not found: {MODEL_PATH}\n\n"
            "Make sure the 'models' folder contains "
            "'student_performance_model.pkl'."
        )

        st.stop()

    try:

        return joblib.load(MODEL_PATH)

    except Exception:

        import pickle

        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)


@st.cache_resource
def load_features():

    if os.path.exists(FEATURES_PATH):

        try:
            return joblib.load(FEATURES_PATH)

        except Exception:

            import pickle

            with open(FEATURES_PATH, "rb") as file:
                return pickle.load(file)

    return [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
        "traveltime", "studytime", "failures", "schoolsup",
        "famsup", "paid", "activities", "nursery", "higher",
        "internet", "romantic", "famrel", "freetime", "goout",
        "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"
    ]


@st.cache_resource
def load_model_metrics():

    if not os.path.exists(METRICS_PATH):
        return None

    try:
        return joblib.load(METRICS_PATH)

    except Exception:

        try:
            import pickle

            with open(METRICS_PATH, "rb") as file:
                return pickle.load(file)

        except Exception:
            return None


# ============================================================
# EXPLAINABLE AI HELPERS
# ============================================================

def get_final_estimator(model):

    """
    Extract the final estimator from a sklearn Pipeline.
    If the model is already an estimator, return it directly.
    """

    if hasattr(model, "steps"):

        try:
            return model.steps[-1][1]
        except Exception:
            pass

    if hasattr(model, "named_steps"):

        try:
            return list(model.named_steps.values())[-1]
        except Exception:
            pass

    return model


def get_feature_names(model, original_features):
    """
    Get the feature names after preprocessing.

    This keeps the original 30 student feature names when possible,
    while also supporting sklearn Pipelines with OneHotEncoder.
    """

    # Pipeline with a preprocessor
    if hasattr(model, "named_steps"):

        preprocessor = model.named_steps.get("preprocessor")

        if preprocessor is not None:
            try:
                return list(
                    preprocessor.get_feature_names_out(
                        original_features
                    )
                )
            except Exception:
                pass

    # Direct estimator
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    return list(original_features)


def get_feature_importance(model, original_features):
    """
    Get Random Forest feature importance.

    IMPORTANT:
    If OneHotEncoder created several columns for one original feature,
    this function combines those encoded columns back into the original
    student feature.

    Example:
        cat__Mjob_teacher
        cat__Mjob_health
        cat__Mjob_services

    are combined into:

        Mjob
    """

    estimator = get_final_estimator(model)

    if not hasattr(estimator, "feature_importances_"):
        return None

    try:
        importances = list(
            estimator.feature_importances_
        )
    except Exception:
        return None

    feature_names = get_feature_names(
        model,
        original_features
    )

    # If names and importance length do not match, use safe fallback.
    if len(feature_names) != len(importances):

        if len(original_features) == len(importances):

            feature_names = list(original_features)

        else:

            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(importances))
            ]

    # ------------------------------------------------------------
    # Combine OneHotEncoder columns into original features
    # ------------------------------------------------------------

    aggregated = {}

    for name, importance in zip(
        feature_names,
        importances
    ):

        name = str(name)

        # Remove sklearn transformer prefixes
        clean_name = name

        if clean_name.startswith("num__"):
            clean_name = clean_name[5:]

        if clean_name.startswith("cat__"):
            clean_name = clean_name[5:]

        matched_feature = None

        # Match against the ORIGINAL 30 features.
        # Longest names first avoids accidental partial matches.
        for original in sorted(
            original_features,
            key=len,
            reverse=True
        ):

            if (
                clean_name == original
                or clean_name.startswith(
                    original + "_"
                )
            ):

                matched_feature = original
                break

        # If it cannot be matched, keep the transformed name.
        if matched_feature is None:
            matched_feature = clean_name

        aggregated[matched_feature] = (
            aggregated.get(
                matched_feature,
                0.0
            )
            + float(importance)
        )

    importance_df = pd.DataFrame({
        "Feature": list(aggregated.keys()),
        "Importance": list(aggregated.values())
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    return importance_df.reset_index(
        drop=True
    )



def get_individual_shap_explanation(model, input_df, original_features):
    """
    Calculate SHAP contributions for the current student.

    The trained model is a Pipeline:
        preprocessing -> Random Forest

    SHAP works on the transformed feature matrix. If categorical
    variables were one-hot encoded, their SHAP contributions are
    aggregated back to the original student feature.
    """
    try:
        import shap

        estimator = get_final_estimator(model)

        if not hasattr(estimator, "estimators_"):
            return None, "The loaded final estimator does not expose Random Forest trees."

        preprocessor = None
        if hasattr(model, "named_steps"):
            preprocessor = model.named_steps.get("preprocessor")

        if preprocessor is None:
            return None, "The model does not contain the expected preprocessing step."

        transformed = preprocessor.transform(input_df)

        # TreeExplainer supports the Random Forest estimator directly.
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)

        # Regression returns a 1-D array for one sample.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 2:
            shap_values = shap_values[0]

        feature_names = list(
            preprocessor.get_feature_names_out(original_features)
        )

        if len(feature_names) != len(shap_values):
            return None, "The transformed feature names do not match SHAP output."

        # Aggregate encoded columns back into original features.
        aggregated = {}

        for name, value in zip(feature_names, shap_values):
            clean_name = str(name)

            if clean_name.startswith("num__"):
                clean_name = clean_name[5:]
            elif clean_name.startswith("cat__"):
                clean_name = clean_name[5:]

            matched = None

            for original in sorted(
                original_features,
                key=len,
                reverse=True
            ):
                if (
                    clean_name == original
                    or clean_name.startswith(original + "_")
                ):
                    matched = original
                    break

            if matched is None:
                matched = clean_name

            aggregated[matched] = (
                aggregated.get(matched, 0.0)
                + float(value)
            )

        explanation_df = pd.DataFrame({
            "Feature": list(aggregated.keys()),
            "SHAP": list(aggregated.values())
        })

        explanation_df["AbsSHAP"] = explanation_df["SHAP"].abs()

        explanation_df = explanation_df.sort_values(
            "AbsSHAP",
            ascending=False
        ).reset_index(drop=True)

        # Expected/base value for regression.
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value).reshape(-1)[0]

        return explanation_df, float(base_value)

    except ImportError:
        return None, (
            "SHAP is not installed. Run: pip install shap"
        )
    except Exception as exc:
        return None, str(exc)


def clean_feature_name(name):
    """
    Convert model feature names into friendly names.
    """

    feature_labels = {

        "school": "School",
        "sex": "Gender",
        "age": "Age",
        "address": "Address",
        "famsize": "Family Size",
        "Pstatus": "Parent Status",
        "Medu": "Mother's Education",
        "Fedu": "Father's Education",
        "Mjob": "Mother's Job",
        "Fjob": "Father's Job",
        "reason": "School Choice Reason",
        "guardian": "Guardian",
        "traveltime": "Travel Time",
        "studytime": "Study Time",
        "failures": "Previous Failures",
        "schoolsup": "School Support",
        "famsup": "Family Support",
        "paid": "Paid Extra Classes",
        "activities": "Extra Activities",
        "nursery": "Nursery Attendance",
        "higher": "Higher Education Goal",
        "internet": "Internet Access",
        "romantic": "Romantic Relationship",
        "famrel": "Family Relationship",
        "freetime": "Free Time",
        "goout": "Going Out",
        "Dalc": "Weekday Consumption",
        "Walc": "Weekend Consumption",
        "health": "Health",
        "absences": "Absences"
    }

    name = str(name)

    if name in feature_labels:
        return feature_labels[name]

    # Remove sklearn prefixes if any remain.
    name = name.replace(
        "num__",
        ""
    ).replace(
        "cat__",
        ""
    )

    # If it is an encoded feature such as Mjob_teacher,
    # display the original feature name.
    for original, label in feature_labels.items():

        if name.startswith(
            original + "_"
        ):

            return label

    return name.replace(
        "_",
        " "
    ).title()


# ============================================================
# LOAD EVERYTHING
# ============================================================

model = load_model()

saved_features = load_features()

model_metrics = load_model_metrics()


# ============================================================
# SESSION STATE - PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# FEATURES
# ============================================================

ALL_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]


FEATURES = [
    feature
    for feature in saved_features
    if feature in ALL_FEATURES
]


if len(FEATURES) == 0:

    FEATURES = ALL_FEATURES.copy()


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">

    <div class="hero-title">
        🎓 Student Performance AI
    </div>

    <div class="hero-subtitle">
        Machine Learning powered academic performance prediction
        and student risk analysis.
    </div>

    <div class="badge">
        🤖 Random Forest Regression
        &nbsp; • &nbsp;
        📊 Academic Analytics
        &nbsp; • &nbsp;
        🎯 G3 Prediction
    </div>

</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Student Performance AI")

    st.write(
        "Predict final academic performance using Machine Learning."
    )

    st.markdown("---")

    st.write("**Model:** Random Forest")
    st.write("**Target:** Final Grade (G3)")
    st.write("**Dataset:** Student Performance")

    st.markdown("---")

    st.caption("Academic analytics prototype")


# ============================================================
# INPUT SECTION
# ============================================================

render_html("""
<div class="section-title">
    👨‍🎓 Student Information
</div>
""")


# ============================================================
# BASIC INFORMATION
# ============================================================

render_html("""
<div class="card-title">
    📋 Basic Information
</div>
""")


col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17,
        step=1
    )

with col2:

    school = st.selectbox(
        "School",
        ["GP", "MS"],
        index=0
    )

with col3:

    sex = st.selectbox(
        "Gender",
        ["F", "M"],
        index=0
    )


col1, col2, col3 = st.columns(3)

with col1:

    address = st.selectbox(
        "Address",
        ["U", "R"],
        index=0,
        help="U = Urban, R = Rural"
    )

with col2:

    famsize = st.selectbox(
        "Family Size",
        ["GT3", "LE3"],
        index=0,
        help="GT3 = greater than 3, LE3 = less/equal to 3"
    )

with col3:

    pstatus = st.selectbox(
        "Parent Status",
        ["T", "A"],
        index=0,
        help="T = living together, A = apart"
    )


# ============================================================
# ACADEMIC FACTORS
# ============================================================

render_html("""
<div class="card-title">
    📚 Academic Factors
</div>
""")


col1, col2, col3 = st.columns(3)

with col1:

    medu = st.slider(
        "Mother's Education",
        0,
        4,
        3,
        help="0 = none, 4 = higher education"
    )

with col2:

    fedu = st.slider(
        "Father's Education",
        0,
        4,
        2,
        help="0 = none, 4 = higher education"
    )

with col3:

    studytime = st.select_slider(
        "Study Time",
        options=[1, 2, 3, 4],
        value=2,
        help="1 = <2 hrs, 4 = >10 hrs"
    )


col1, col2, col3 = st.columns(3)

with col1:

    failures = st.number_input(
        "Previous Failures",
        min_value=0,
        max_value=4,
        value=0,
        step=1
    )

with col2:

    traveltime = st.select_slider(
        "Travel Time",
        options=[1, 2, 3, 4],
        value=1,
        help="1 = <15 min, 4 = >60 min"
    )

with col3:

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=93,
        value=5,
        step=1
    )


# ============================================================
# FAMILY & PERSONAL FACTORS
# ============================================================

render_html("""
<div class="card-title">
    ❤️ Family & Personal Factors
</div>
""")


col1, col2, col3 = st.columns(3)

with col1:

    famrel = st.slider(
        "Family Relationship",
        1,
        5,
        4
    )

with col2:

    freetime = st.slider(
        "Free Time",
        1,
        5,
        3
    )

with col3:

    goout = st.slider(
        "Going Out",
        1,
        5,
        3
    )


col1, col2, col3 = st.columns(3)

with col1:

    health = st.slider(
        "Health",
        1,
        5,
        4
    )

with col2:

    dalc = st.slider(
        "Weekday Alcohol Consumption",
        1,
        5,
        1
    )

with col3:

    walc = st.slider(
        "Weekend Alcohol Consumption",
        1,
        5,
        1
    )


# ============================================================
# FAMILY & SOCIAL INFORMATION
# ============================================================

render_html("""
<div class="card-title">
    🏠 Family & Social Information
</div>
""")


col1, col2, col3 = st.columns(3)

with col1:

    mjob = st.selectbox(
        "Mother's Job",
        [
            "teacher",
            "health",
            "services",
            "at_home",
            "other"
        ]
    )

with col2:

    fjob = st.selectbox(
        "Father's Job",
        [
            "teacher",
            "health",
            "services",
            "at_home",
            "other"
        ]
    )

with col3:

    guardian = st.selectbox(
        "Guardian",
        [
            "mother",
            "father",
            "other"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:

    reason = st.selectbox(
        "Reason for Choosing School",
        [
            "course",
            "reputation",
            "home",
            "other"
        ]
    )

with col2:

    schoolsup = st.selectbox(
        "Extra School Support",
        ["yes", "no"]
    )

with col3:

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"]
    )


col1, col2, col3 = st.columns(3)

with col1:

    paid = st.selectbox(
        "Paid Extra Classes",
        ["yes", "no"]
    )

with col2:

    activities = st.selectbox(
        "Extra-curricular Activities",
        ["yes", "no"]
    )

with col3:

    nursery = st.selectbox(
        "Attended Nursery",
        ["yes", "no"]
    )


# ============================================================
# OTHER STUDENT FACTORS
# ============================================================

render_html("""
<div class="card-title">
    🌐 Other Student Factors
</div>
""")


col1, col2, col3 = st.columns(3)

with col1:

    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"]
    )

with col2:

    internet = st.selectbox(
        "Internet Access at Home",
        ["yes", "no"]
    )

with col3:

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )


# ============================================================
# BUILD INPUT DATA
# ============================================================

input_data = {

    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": pstatus,

    "Medu": medu,
    "Fedu": fedu,

    "Mjob": mjob,
    "Fjob": fjob,

    "reason": reason,
    "guardian": guardian,

    "traveltime": traveltime,
    "studytime": studytime,
    "failures": failures,

    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "nursery": nursery,
    "higher": higher,
    "internet": internet,
    "romantic": romantic,

    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,

    "Dalc": dalc,
    "Walc": walc,
    "health": health,

    "absences": absences
}


input_df = pd.DataFrame([
    {
        feature: input_data[feature]
        for feature in FEATURES
    }
])


input_df = input_df[FEATURES]


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_clicked = st.button(
    "🚀 Predict Student Performance",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    try:

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = float(
            model.predict(input_df)[0]
        )

        prediction = max(
            0.0,
            min(20.0, prediction)
        )


        # ====================================================
        # PERFORMANCE LEVEL
        # ====================================================

        if prediction >= 16:

            level = "Excellent"
            level_icon = "🏆"

        elif prediction >= 14:

            level = "Good"
            level_icon = "🟢"

        elif prediction >= 10:

            level = "Average"
            level_icon = "🟡"

        else:

            level = "Needs Improvement"
            level_icon = "🔴"


        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        risk_points = 0

        if absences >= 15:

            risk_points += 2

        elif absences >= 8:

            risk_points += 1


        if failures >= 2:

            risk_points += 2

        elif failures == 1:

            risk_points += 1


        if studytime == 1:

            risk_points += 1


        if health <= 2:

            risk_points += 1


        if prediction < 10:

            risk_points += 2

        elif prediction < 12:

            risk_points += 1


        if risk_points >= 4:

            risk = "HIGH RISK"
            risk_class = "risk-high"
            risk_icon = "🔴"

            risk_text = (
                "Several academic and behavioral indicators "
                "require attention. Consider additional academic "
                "support and regular progress monitoring."
            )

        elif risk_points >= 2:

            risk = "MEDIUM RISK"
            risk_class = "risk-medium"
            risk_icon = "🟡"

            risk_text = (
                "Some factors may affect academic performance. "
                "Improving study consistency and attendance may "
                "help reduce risk."
            )

        else:

            risk = "LOW RISK"
            risk_class = "risk-low"
            risk_icon = "🟢"

            risk_text = (
                "Current academic indicators do not show "
                "significant risk factors. Continue maintaining "
                "consistent study and attendance habits."
            )


        # ====================================================
        # RESULT HEADER
        # ====================================================

        render_html("""
        <div class="section-title">
            🎯 Prediction Result
        </div>
        """)


        # ====================================================
        # PREDICTION CARD
        # ====================================================

        render_html(f"""
        <div class="prediction-card">

            <div class="prediction-label">
                🎯 Predicted Final Grade
            </div>

            <div class="prediction-score">
                {prediction:.2f}
            </div>

            <div class="prediction-sub">
                out of 20
            </div>

        </div>
        """)


        # ====================================================
        # GAUGE
        # ====================================================

        gauge_pct = int(
            round(
                (prediction / 20) * 100
            )
        )


        render_html(f"""
        <div class="score-gauge">

            <div class="gauge-header">
                <span>0</span>
                <span>Performance score</span>
                <span>20</span>
            </div>

            <div class="gauge-track">

                <div
                    class="gauge-fill"
                    style="width:{gauge_pct}%;">
                </div>

            </div>

            <div class="gauge-percentage">
                {gauge_pct}% of maximum score
            </div>

        </div>
        """)


        # ====================================================
        # METRICS
        # ====================================================

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            render_html(f"""
            <div class="metric-card">

                <div class="metric-icon">
                    {level_icon}
                </div>

                <div class="metric-title">
                    Performance Level
                </div>

                <div class="metric-value">
                    {level}
                </div>

            </div>
            """)


        with c2:

            render_html(f"""
            <div class="metric-card">

                <div class="metric-icon">
                    🎯
                </div>

                <div class="metric-title">
                    Predicted Grade
                </div>

                <div class="metric-value">
                    {prediction:.2f}
                </div>

            </div>
            """)


        with c3:

            render_html(f"""
            <div class="metric-card">

                <div class="metric-icon">
                    📚
                </div>

                <div class="metric-title">
                    Study Time
                </div>

                <div class="metric-value">
                    {studytime}
                </div>

            </div>
            """)


        with c4:

            render_html(f"""
            <div class="metric-card">

                <div class="metric-icon">
                    📅
                </div>

                <div class="metric-title">
                    Absences
                </div>

                <div class="metric-value">
                    {absences}
                </div>

            </div>
            """)


        # ====================================================
        # PERFORMANCE OVERVIEW
        # ====================================================

        render_html("""
        <div class="section-title">
            📊 Performance Overview
        </div>
        """)


        st.info(
            f"The model predicts a final grade of "
            f"{prediction:.2f}/20. "
            f"The estimated performance level is **{level}**."
        )


        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        render_html("""
        <div class="section-title">
            🛡️ Performance Risk Analysis
        </div>
        """)


        render_html(f"""
        <div class="{risk_class}">

            <div class="risk-title">
                {risk_icon} {risk}
            </div>

            <div class="risk-description">
                {risk_text}
            </div>

        </div>
        """)


        # ====================================================
        # IMPORTANT FACTORS
        # ====================================================

        render_html("""
        <div class="section-title">
            💡 Important Student Factors
        </div>
        """)


        f1, f2, f3 = st.columns(3)


        with f1:

            render_html(f"""
            <div class="factor-card">

                <div class="factor-title">
                    📚 Study Time
                </div>

                <div class="factor-value">
                    {studytime}
                </div>

                <div class="factor-description">
                    Higher study time generally provides more
                    opportunity for academic preparation.
                </div>

            </div>
            """)


        with f2:

            render_html(f"""
            <div class="factor-card">

                <div class="factor-title">
                    ❌ Previous Failures
                </div>

                <div class="factor-value">
                    {failures}
                </div>

                <div class="factor-description">
                    Previous failures can indicate areas requiring
                    additional academic support.
                </div>

            </div>
            """)


        with f3:

            render_html(f"""
            <div class="factor-card">

                <div class="factor-title">
                    📅 Absences
                </div>

                <div class="factor-value">
                    {absences}
                </div>

                <div class="factor-description">
                    Attendance can influence consistency in learning
                    and classroom participation.
                </div>

            </div>
            """)


        # ====================================================
        # STUDENT PROFILE
        # ====================================================

        render_html("""
        <div class="section-title">
            🔍 Student Profile
        </div>
        """)


        p1, p2, p3, p4 = st.columns(4)


        with p1:

            render_html(f"""
            <div class="profile-item">

                <div class="profile-label">
                    ❤️ Health
                </div>

                <div class="profile-value">
                    {health}
                </div>

            </div>
            """)


        with p2:

            render_html(f"""
            <div class="profile-item">

                <div class="profile-label">
                    👨‍👩‍👧 Family Relationship
                </div>

                <div class="profile-value">
                    {famrel}
                </div>

            </div>
            """)


        with p3:

            render_html(f"""
            <div class="profile-item">

                <div class="profile-label">
                    🎮 Free Time
                </div>

                <div class="profile-value">
                    {freetime}
                </div>

            </div>
            """)


        with p4:

            render_html(f"""
            <div class="profile-item">

                <div class="profile-label">
                    👥 Going Out
                </div>

                <div class="profile-value">
                    {goout}
                </div>

            </div>
            """)


        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        render_html("""
        <div class="section-title">
            🧠 Explainable AI
        </div>
        """)


        render_html("""
        <div class="risk-low">

            <div class="risk-title">
                🔍 Why did the model make this prediction?
            </div>

            <div class="risk-description">
                These are the features that the Random Forest model
                considers most important when learning student
                performance patterns. Higher importance means the
                feature was more useful to the trained model across
                its decision trees.
            </div>

        </div>
        """)


        importance_df = get_feature_importance(
            model,
            FEATURES
        )


        if importance_df is not None and not importance_df.empty:

            top_features = importance_df.head(8).copy()

            top_features["Feature"] = (
                top_features["Feature"]
                .apply(clean_feature_name)
            )


            total_importance = float(
                importance_df["Importance"].sum()
            )


            if total_importance <= 0:
                total_importance = 1.0


            explanation_html = """
            <div class="explain-card">
            """


            for _, row in top_features.iterrows():

                feature_name = row["Feature"]

                importance = float(
                    row["Importance"]
                )

                percentage = (
                    importance /
                    total_importance
                ) * 100


                explanation_html += f"""

                <div class="explain-row">

                    <div class="explain-header">

                        <div class="explain-name">
                            📊 {feature_name}
                        </div>

                        <div class="explain-value">
                            {importance:.3f}
                        </div>

                    </div>

                    <div class="explain-track">

                        <div
                            class="explain-fill"
                            style="width:{percentage:.1f}%;">
                        </div>

                    </div>

                    <div class="explain-note">
                        Relative model importance
                    </div>

                </div>

                """


            explanation_html += """
            </div>
            """


            render_html(
                explanation_html
            )


            # Top 3 summary

            top_three = top_features.head(3)

            summary_text = ", ".join(
                top_three["Feature"].tolist()
            )


            render_html(f"""
            <div class="recommendation">

                🧠 <b>Key Model Insight:</b>

                The three strongest model features in this
                prediction model are
                <b>{summary_text}</b>.

            </div>
            """)


        else:

            st.info(
                "Feature importance is not available for the "
                "loaded model. The model may use a pipeline or "
                "algorithm that does not expose feature_importances_."
            )


        # ====================================================
        # INDIVIDUAL EXPLAINABLE AI - SHAP
        # ====================================================

        render_html("""
        <div class="section-title">
            🔬 Individual Prediction Explanation
        </div>
        """)

        render_html("""
        <div class="shap-card">

            <div class="risk-title">
                🔍 Why did the model make THIS prediction?
            </div>

            <div class="shap-summary">
                SHAP explains how each student feature pushed the
                prediction higher or lower for this individual student.
                Positive values push the prediction upward, while
                negative values push it downward.
            </div>

        </div>
        """)

        shap_df, shap_status = get_individual_shap_explanation(
            model,
            input_df,
            FEATURES
        )

        if shap_df is not None and not shap_df.empty:

            top_shap = shap_df.head(8).copy()
            max_abs_shap = float(top_shap["AbsSHAP"].max())

            if max_abs_shap <= 0:
                max_abs_shap = 1.0

            shap_html = """
            <div class="shap-card">
            """

            for _, row in top_shap.iterrows():

                feature_name = clean_feature_name(row["Feature"])
                contribution = float(row["SHAP"])

                bar_width = min(
                    100,
                    max(
                        3,
                        (abs(contribution) / max_abs_shap) * 100
                    )
                )

                if contribution >= 0:
                    value_class = "shap-positive"
                    fill_class = "shap-positive-fill"
                    direction = "↑ increases prediction"
                    sign = "+"
                else:
                    value_class = "shap-negative"
                    fill_class = "shap-negative-fill"
                    direction = "↓ decreases prediction"
                    sign = ""

                shap_html += f"""
                <div class="shap-row">

                    <div class="shap-header">

                        <div class="shap-name">
                            📊 {feature_name}
                        </div>

                        <div class="shap-value {value_class}">
                            {sign}{contribution:.3f}
                        </div>

                    </div>

                    <div class="shap-track">
                        <div
                            class="{fill_class}"
                            style="width:{bar_width:.1f}%;">
                        </div>
                    </div>

                    <div class="shap-note">
                        {direction}
                    </div>

                </div>
                """

            shap_html += """
            </div>
            """

            render_html(shap_html)

            # Key individual insight.
            strongest = top_shap.iloc[0]
            strongest_name = clean_feature_name(strongest["Feature"])
            strongest_value = float(strongest["SHAP"])

            if strongest_value >= 0:
                insight = (
                    f"{strongest_name} had the strongest positive "
                    f"influence on this student's prediction "
                    f"(+{strongest_value:.3f})."
                )
            else:
                insight = (
                    f"{strongest_name} had the strongest negative "
                    f"influence on this student's prediction "
                    f"({strongest_value:.3f})."
                )

            render_html(f"""
            <div class="recommendation">
                🧠 <b>Individual Model Insight:</b> {insight}
            </div>
            """)

        else:

            render_html(f"""
            <div class="recommendation">
                ℹ️ <b>Individual SHAP explanation unavailable.</b>
                <br><br>
                {shap_status}
            </div>
            """)


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        render_html("""
        <div class="section-title">
            📊 Model Performance
        </div>
        """)


        if model_metrics is not None:

            r2 = model_metrics.get("r2")
            mae = model_metrics.get("mae")
            rmse = model_metrics.get("rmse")


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "R² Score",
                    f"{r2:.3f}"
                    if r2 is not None
                    else "N/A"
                )


            with c2:

                st.metric(
                    "MAE",
                    f"{mae:.3f}"
                    if mae is not None
                    else "N/A"
                )


            with c3:

                st.metric(
                    "RMSE",
                    f"{rmse:.3f}"
                    if rmse is not None
                    else "N/A"
                )


        else:

            render_html("""
            <div class="recommendation">

                ℹ️ <b>Model evaluation metrics are not loaded yet.</b>

                <br><br>

                The application is successfully using the trained
                Random Forest model for prediction. To display
                R², MAE and RMSE here, save those evaluation metrics
                during model training as:

                <br><br>

                <b>models/model_metrics.pkl</b>

            </div>
            """)


        # ====================================================
        # FACTOR SNAPSHOT
        # ====================================================

        render_html("""
        <div class="section-title">
            📈 Factor Snapshot
        </div>
        """)


        snapshot_data = [

            (
                "📚 Study Time",
                studytime,
                4,
                "1 - 4"
            ),

            (
                "❌ Previous Failures",
                failures,
                4,
                "0 - 4"
            ),

            (
                "📅 Absences",
                absences,
                93,
                "0 - 93"
            ),

            (
                "❤️ Health",
                health,
                5,
                "1 - 5"
            )

        ]


        snapshot_html = """
        <div class="factor-snapshot">
        """


        for name, value, maximum, scale in snapshot_data:

            percentage = min(
                100,
                max(
                    0,
                    (value / maximum) * 100
                )
            )


            snapshot_html += f"""

            <div class="snapshot-row">

                <div class="snapshot-header">

                    <div class="snapshot-name">
                        {name}
                    </div>

                    <div class="snapshot-value">
                        {value}
                    </div>

                </div>

                <div class="snapshot-track">

                    <div
                        class="snapshot-fill"
                        style="width:{percentage:.1f}%;">
                    </div>

                </div>

                <div class="snapshot-scale">
                    Scale: {scale}
                </div>

            </div>

            """


        snapshot_html += """
        </div>
        """


        render_html(snapshot_html)


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        render_html("""
        <div class="section-title">
            🚀 Recommendations
        </div>
        """)


        recommendations = []


        if studytime <= 2:

            recommendations.append(
                "📚 Increase study time gradually and maintain "
                "a consistent study schedule."
            )

        elif prediction < 12:

            recommendations.append(
                "📚 Maintain your current study routine and "
                "focus on improving study effectiveness."
            )


        if absences >= 8:

            recommendations.append(
                "🏫 Improve attendance and avoid unnecessary absences."
            )

        else:

            recommendations.append(
                "🏫 Continue maintaining good attendance."
            )


        if failures >= 1:

            recommendations.append(
                "📝 Focus on subjects where previous academic "
                "difficulties occurred."
            )


        if health <= 2:

            recommendations.append(
                "❤️ Maintain healthy sleep, nutrition, exercise "
                "and study habits."
            )


        if prediction < 10:

            recommendations.append(
                "🎯 Consider additional academic support or tutoring."
            )


        if not recommendations:

            recommendations = [

                "📚 Maintain a consistent study schedule.",

                "🏫 Continue maintaining good attendance.",

                "🎯 Keep monitoring academic progress "
                "throughout the semester."

            ]


        for recommendation in recommendations:

            render_html(
                f"""
                <div class="recommendation">
                    ✓ {recommendation}
                </div>
                """
            )


        # ====================================================
        # SAVE CURRENT PREDICTION TO SESSION HISTORY
        # ====================================================

        from datetime import datetime

        prediction_record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Age": age,
            "Gender": sex,
            "Study Time": studytime,
            "Previous Failures": failures,
            "Absences": absences,
            "Health": health,
            "Predicted Grade": round(prediction, 2),
            "Performance Level": level,
            "Risk Level": risk
        }

        # Avoid adding the same prediction repeatedly during a rerun.
        if (
            not st.session_state.prediction_history
            or st.session_state.prediction_history[-1] != prediction_record
        ):
            st.session_state.prediction_history.append(prediction_record)


        # ====================================================
        # DOWNLOADABLE STUDENT REPORT
        # ====================================================

        render_html("""
        <div class="section-title">
            📄 Student Performance Report
        </div>
        """)

        report_lines = [
            "STUDENT PERFORMANCE AI - STUDENT REPORT",
            "=" * 55,
            "",
            f"Generated: {prediction_record['Time']}",
            "",
            "PREDICTION",
            "-" * 55,
            f"Predicted Final Grade: {prediction:.2f} / 20",
            f"Performance Level: {level}",
            f"Risk Level: {risk}",
            "",
            "STUDENT PROFILE",
            "-" * 55,
            f"Age: {age}",
            f"Gender: {sex}",
            f"School: {school}",
            f"Study Time: {studytime}",
            f"Previous Failures: {failures}",
            f"Absences: {absences}",
            f"Health: {health}",
            f"Family Relationship: {famrel}",
            f"Free Time: {freetime}",
            f"Going Out: {goout}",
            "",
            "MODEL PERFORMANCE",
            "-" * 55,
            f"Model: Random Forest Regression",
            f"Target: G3 Final Grade",
            f"R2 Score: {model_metrics.get('r2', 'N/A') if model_metrics else 'N/A'}",
            f"MAE: {model_metrics.get('mae', 'N/A') if model_metrics else 'N/A'}",
            f"RMSE: {model_metrics.get('rmse', 'N/A') if model_metrics else 'N/A'}",
            "",
            "TOP MODEL FACTORS",
            "-" * 55,
        ]

        if importance_df is not None and not importance_df.empty:
            for _, row in importance_df.head(8).iterrows():
                report_lines.append(
                    f"{clean_feature_name(row['Feature'])}: {float(row['Importance']):.3f}"
                )

        report_lines.extend(["", "INDIVIDUAL SHAP EXPLANATION", "-" * 55])

        if shap_df is not None and not shap_df.empty:
            for _, row in shap_df.head(8).iterrows():
                report_lines.append(
                    f"{clean_feature_name(row['Feature'])}: {float(row['SHAP']):+.3f}"
                )
        else:
            report_lines.append("Individual SHAP explanation unavailable.")

        report_lines.extend(["", "RECOMMENDATIONS", "-" * 55])
        report_lines.extend([f"- {item}" for item in recommendations])
        report_lines.extend(["", "Generated by Student Performance AI"])

        report_text = "\n".join(report_lines)

        st.download_button(
            "📥 Download Student Report",
            data=report_text,
            file_name="student_performance_report.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"report_{len(st.session_state.prediction_history)}"
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        render_html("""
        <div class="section-title">
            🤖 Model Information
        </div>
        """)


        m1, m2, m3 = st.columns(3)


        with m1:

            render_html("""
            <div class="model-info">

                <div class="model-label">
                    MODEL
                </div>

                <div class="model-value">
                    Random Forest
                </div>

            </div>
            """)


        with m2:

            render_html("""
            <div class="model-info">

                <div class="model-label">
                    TARGET
                </div>

                <div class="model-value">
                    G3 Final Grade
                </div>

            </div>
            """)


        with m3:

            render_html("""
            <div class="model-info">

                <div class="model-label">
                    SCALE
                </div>

                <div class="model-value">
                    0 - 20
                </div>

            </div>
            """)


        # ====================================================
        # FOOTER
        # ====================================================

        render_html("""
        <div class="divider"></div>

        <div class="footer">

            🎓 <b>Student Performance AI</b>

            <br>

            Machine Learning + Random Forest
            &nbsp; | &nbsp;
            Explainable AI
            &nbsp; | &nbsp;
            Academic Analytics

        </div>
        """)


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error("Prediction failed.")

        render_html("""
        <div class="section-title">
            🔧 What happened?
        </div>

        <div class="recommendation">

            The application could not send the entered data
            through the saved Machine Learning pipeline.

            <br><br>

            This usually means that the input columns or data
            types do not exactly match the columns used while
            training the model.

        </div>
        """)

        st.exception(e)


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    render_html("""
    <div class="section-title">
        📜 Prediction History
    </div>
    """)

    history_df = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    history_csv = history_df.to_csv(index=False).encode("utf-8")

    h1, h2 = st.columns(2)

    with h1:
        st.download_button(
            "📥 Download Prediction History (CSV)",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_history"
        )

    with h2:
        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True,
            key="clear_history"
        ):
            st.session_state.prediction_history = []
            st.rerun()

# ============================================================
# 📈 STUDENT PROGRESS TRACKER - NEW ADD-ON
# Existing prediction history is used. No model or existing feature
# is changed. This section helps compare repeated predictions for the
# same student profile during the current app session.
# ============================================================

if st.session_state.prediction_history:

    render_html("""
    <div class="section-title">
        📈 Student Progress Tracker
    </div>

    <div class="recommendation">
        📊 Track how a student's predicted performance changes when the
        profile is evaluated again after updated academic information.
        The tracker uses the existing Random Forest predictions and does
        not retrain the model.
    </div>
    """)

    progress_df = pd.DataFrame(
        st.session_state.prediction_history
    ).copy()

    progress_df["Time"] = pd.to_datetime(
        progress_df["Time"],
        errors="coerce"
    )

    progress_df["Predicted Grade"] = pd.to_numeric(
        progress_df["Predicted Grade"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # SELECT STUDENT / PROFILE
    # --------------------------------------------------------

    progress_options = [
        "Current Session (all predictions)",
        "Compare First vs Latest"
    ]

    progress_mode = st.selectbox(
        "📊 Progress View",
        progress_options,
        key="progress_tracker_mode"
    )

    # --------------------------------------------------------
    # OVERALL SESSION PROGRESS
    # --------------------------------------------------------

    if progress_mode == "Current Session (all predictions)":

        if len(progress_df) >= 2:

            first_grade = float(
                progress_df.iloc[0]["Predicted Grade"]
            )

            latest_grade = float(
                progress_df.iloc[-1]["Predicted Grade"]
            )

            grade_change = latest_grade - first_grade

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "First Prediction",
                    f"{first_grade:.2f}/20"
                )

            with p2:
                st.metric(
                    "Latest Prediction",
                    f"{latest_grade:.2f}/20"
                )

            with p3:
                st.metric(
                    "Prediction Change",
                    f"{grade_change:+.2f}"
                )

            # ----------------------------------------------------
            # PROGRESS CHART
            # ----------------------------------------------------

            chart_df = progress_df[
                ["Time", "Predicted Grade"]
            ].copy()

            chart_df = chart_df.set_index("Time")

            render_html("""
            <div class="card-title">
                📈 Predicted Grade Trend
            </div>
            """)

            st.line_chart(
                chart_df,
                y="Predicted Grade",
                use_container_width=True
            )

            if grade_change > 0.05:
                st.success(
                    f"📈 The latest prediction is "
                    f"**{grade_change:+.2f} points** higher than "
                    "the first recorded prediction."
                )
            elif grade_change < -0.05:
                st.warning(
                    f"📉 The latest prediction is "
                    f"**{abs(grade_change):.2f} points** lower than "
                    "the first recorded prediction. Review the updated "
                    "student inputs and provide appropriate support."
                )
            else:
                st.info(
                    "➡️ The recorded predictions are currently very close "
                    "to the initial prediction."
                )

        else:
            st.info(
                "📌 Make another prediction with updated student information "
                "to see a progress trend."
            )

    # --------------------------------------------------------
    # FIRST VS LATEST COMPARISON
    # --------------------------------------------------------

    else:

        first_record = progress_df.iloc[0]
        latest_record = progress_df.iloc[-1]

        comparison_progress = pd.DataFrame({
            "Metric": [
                "Predicted Grade",
                "Study Time",
                "Previous Failures",
                "Absences",
                "Health",
                "Performance Level",
                "Risk Level"
            ],
            "First Prediction": [
                first_record["Predicted Grade"],
                first_record["Study Time"],
                first_record["Previous Failures"],
                first_record["Absences"],
                first_record["Health"],
                first_record["Performance Level"],
                first_record["Risk Level"]
            ],
            "Latest Prediction": [
                latest_record["Predicted Grade"],
                latest_record["Study Time"],
                latest_record["Previous Failures"],
                latest_record["Absences"],
                latest_record["Health"],
                latest_record["Performance Level"],
                latest_record["Risk Level"]
            ]
        })

        st.dataframe(
            comparison_progress,
            use_container_width=True,
            hide_index=True
        )

        first_grade = float(
            first_record["Predicted Grade"]
        )

        latest_grade = float(
            latest_record["Predicted Grade"]
        )

        grade_change = latest_grade - first_grade

        if grade_change > 0.05:
            st.success(
                f"🏆 Improvement detected: **{grade_change:+.2f} points** "
                "from the first to the latest prediction."
            )
        elif grade_change < -0.05:
            st.warning(
                f"⚠️ The latest prediction changed by "
                f"**{grade_change:+.2f} points** compared with the first "
                "prediction."
            )
        else:
            st.info(
                "➡️ No meaningful change was detected between the first "
                "and latest predictions."
            )

    # --------------------------------------------------------
    # DOWNLOAD PROGRESS DATA
    # --------------------------------------------------------

    progress_download = progress_df.copy()
    progress_download["Time"] = progress_download["Time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.download_button(
        "📥 Download Progress History (CSV)",
        data=progress_download.to_csv(index=False).encode("utf-8"),
        file_name="student_progress_history.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_progress_history"
    )

# ============================================================
# BATCH STUDENT PREDICTION
# ============================================================

render_html("""
<div class="section-title">
    👥 Batch Student Prediction
</div>

<div class="recommendation">
    📂 Upload a CSV file containing multiple student records.
    The existing Random Forest model will predict the final grade
    for every student without changing the individual prediction system.
</div>
""")


# ------------------------------------------------------------
# CSV TEMPLATE DOWNLOAD
# ------------------------------------------------------------

batch_template_columns = [
    "student_id"
] + FEATURES

batch_template_df = pd.DataFrame(
    columns=batch_template_columns
)

template_csv = batch_template_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "📄 Download Batch CSV Template",
    data=template_csv,
    file_name="student_batch_template.csv",
    mime="text/csv",
    use_container_width=True,
    key="download_batch_template"
)


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------

batch_file = st.file_uploader(
    "📤 Upload Student CSV",
    type=["csv"],
    key="batch_student_upload"
)


if batch_file is not None:

    try:

        batch_df = pd.read_csv(batch_file)

        st.success(
            f"✅ File uploaded successfully — "
            f"{len(batch_df)} student record(s) found."
        )


        # ----------------------------------------------------
        # CHECK REQUIRED COLUMNS
        # ----------------------------------------------------

        missing_columns = [
            feature
            for feature in FEATURES
            if feature not in batch_df.columns
        ]


        if missing_columns:

            st.error(
                "❌ Missing required columns:"
            )

            st.write(
                ", ".join(missing_columns)
            )

            st.info(
                "Please download the CSV template above and "
                "use the same column names."
            )

        else:

            # ------------------------------------------------
            # KEEP ONLY MODEL FEATURES
            # ------------------------------------------------

            batch_input = batch_df[FEATURES].copy()


            # ------------------------------------------------
            # NUMERIC FEATURES
            # ------------------------------------------------

            numeric_features = [
                "age",
                "Medu",
                "Fedu",
                "traveltime",
                "studytime",
                "failures",
                "famrel",
                "freetime",
                "goout",
                "Dalc",
                "Walc",
                "health",
                "absences"
            ]


            for feature in numeric_features:

                if feature in batch_input.columns:

                    batch_input[feature] = pd.to_numeric(
                        batch_input[feature],
                        errors="coerce"
                    )


            # ------------------------------------------------
            # CHECK NUMERIC VALUES
            # ------------------------------------------------

            invalid_numeric = batch_input[
                numeric_features
            ].isnull().any(axis=1)


            if invalid_numeric.any():

                invalid_count = int(
                    invalid_numeric.sum()
                )

                st.error(
                    f"❌ {invalid_count} row(s) contain "
                    "invalid or missing numeric values."
                )

            else:

                # ------------------------------------------------
                # BATCH PREDICTION
                # ------------------------------------------------

                predictions = model.predict(
                    batch_input
                )


                predictions = np.clip(
                    predictions,
                    0.0,
                    20.0
                )


                # ------------------------------------------------
                # CREATE RESULT DATAFRAME
                # ------------------------------------------------

                results_df = batch_df.copy()

                results_df["Predicted Grade"] = np.round(
                    predictions,
                    2
                )


                # ------------------------------------------------
                # PERFORMANCE LEVEL
                # ------------------------------------------------

                def batch_performance_level(score):

                    if score >= 16:
                        return "Excellent"

                    elif score >= 14:
                        return "Good"

                    elif score >= 10:
                        return "Average"

                    else:
                        return "Needs Improvement"


                results_df["Performance Level"] = [
                    batch_performance_level(score)
                    for score in predictions
                ]


                # ------------------------------------------------
                # RISK ANALYSIS
                # SAME LOGIC AS INDIVIDUAL PREDICTION
                # ------------------------------------------------

                risk_levels = []


                for index, row in batch_input.iterrows():

                    prediction_value = float(
                        predictions[index]
                    )

                    risk_points = 0


                    # Absences
                    if row["absences"] >= 15:

                        risk_points += 2

                    elif row["absences"] >= 8:

                        risk_points += 1


                    # Previous failures
                    if row["failures"] >= 2:

                        risk_points += 2

                    elif row["failures"] == 1:

                        risk_points += 1


                    # Study time
                    if row["studytime"] == 1:

                        risk_points += 1


                    # Health
                    if row["health"] <= 2:

                        risk_points += 1


                    # Prediction
                    if prediction_value < 10:

                        risk_points += 2

                    elif prediction_value < 12:

                        risk_points += 1


                    # Risk level
                    if risk_points >= 4:

                        risk_levels.append(
                            "HIGH RISK"
                        )

                    elif risk_points >= 2:

                        risk_levels.append(
                            "MEDIUM RISK"
                        )

                    else:

                        risk_levels.append(
                            "LOW RISK"
                        )


                results_df["Risk Level"] = risk_levels


                # ------------------------------------------------
                # STUDENT ID
                # ------------------------------------------------

                if "student_id" not in results_df.columns:

                    results_df.insert(
                        0,
                        "student_id",
                        [
                            f"Student-{i + 1}"
                            for i in range(len(results_df))
                        ]
                    )


                # ------------------------------------------------
                # BATCH RESULTS
                # ------------------------------------------------

                render_html("""
                <div class="section-title">
                    📊 Batch Prediction Results
                </div>
                """)


                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )


                # ------------------------------------------------
                # SUMMARY METRICS
                # ------------------------------------------------

                total_students = len(
                    results_df
                )

                average_grade = float(
                    results_df["Predicted Grade"].mean()
                )

                high_risk_count = int(
                    (
                        results_df["Risk Level"]
                        == "HIGH RISK"
                    ).sum()
                )

                good_students = int(
                    (
                        results_df["Performance Level"]
                        .isin(
                            ["Excellent", "Good"]
                        )
                    ).sum()
                )


                render_html("""
                <div class="section-title">
                    📈 Batch Analytics
                </div>
                """)


                b1, b2, b3, b4 = st.columns(4)


                with b1:

                    st.metric(
                        "👥 Students",
                        total_students
                    )


                with b2:

                    st.metric(
                        "🎯 Average Grade",
                        f"{average_grade:.2f}"
                    )


                with b3:

                    st.metric(
                        "🟢 Good / Excellent",
                        good_students
                    )


                with b4:

                    st.metric(
                        "🔴 High Risk",
                        high_risk_count
                    )


                # ------------------------------------------------
                # PERFORMANCE DISTRIBUTION
                # ------------------------------------------------

                render_html("""
                <div class="section-title">
                    📊 Performance Distribution
                </div>
                """)


                performance_counts = (
                    results_df[
                        "Performance Level"
                    ]
                    .value_counts()
                )


                st.bar_chart(
                    performance_counts
                )


                # ------------------------------------------------
                # RISK DISTRIBUTION
                # ------------------------------------------------

                render_html("""
                <div class="section-title">
                    🛡️ Risk Distribution
                </div>
                """)


                risk_counts = (
                    results_df[
                        "Risk Level"
                    ]
                    .value_counts()
                )


                st.bar_chart(
                    risk_counts
                )


                # ------------------------------------------------
                # DOWNLOAD RESULTS
                # ------------------------------------------------

                render_html("""
                <div class="section-title">
                    📥 Download Batch Results
                </div>
                """)


                batch_csv = results_df.to_csv(
                    index=False
                ).encode("utf-8")


                st.download_button(
                    "📥 Download Batch Prediction Results",
                    data=batch_csv,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_batch_results"
                )


                # ------------------------------------------------
                # HIGH-RISK STUDENTS
                # ------------------------------------------------

                high_risk_students = results_df[
                    results_df["Risk Level"]
                    == "HIGH RISK"
                ]


                if not high_risk_students.empty:

                    render_html("""
                    <div class="section-title">
                        🚨 Students Requiring Attention
                    </div>
                    """)


                    st.warning(
                        f"{len(high_risk_students)} "
                        "student(s) are currently classified "
                        "as HIGH RISK."
                    )


                    st.dataframe(
                        high_risk_students[
                            [
                                "student_id",
                                "Predicted Grade",
                                "Performance Level",
                                "Risk Level"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )



                # ============================================================
                # 🚀 ADVANCED BATCH ANALYTICS - ADDITIONAL FEATURES
                # ============================================================

                render_html("""
                <div class="section-title">
                    🧠 Advanced Batch Analytics
                </div>
                """)

                # Create a safe numeric analysis copy. The original
                # results_df is kept unchanged for the existing system.
                analysis_df = results_df.copy()

                for analysis_feature in [
                    "age", "Medu", "Fedu", "traveltime", "studytime",
                    "failures", "famrel", "freetime", "goout",
                    "Dalc", "Walc", "health", "absences"
                ]:
                    if analysis_feature in batch_input.columns:
                        analysis_df[analysis_feature] = batch_input[
                            analysis_feature
                        ].to_numpy()

                # ------------------------------------------------------------
                # ADVANCED SUMMARY METRICS
                # ------------------------------------------------------------

                high_risk_percentage = (
                    (high_risk_count / total_students) * 100
                    if total_students > 0
                    else 0
                )

                average_study_time = (
                    float(analysis_df["studytime"].mean())
                    if "studytime" in analysis_df.columns
                    else 0
                )

                average_absences = (
                    float(analysis_df["absences"].mean())
                    if "absences" in analysis_df.columns
                    else 0
                )

                average_failures = (
                    float(analysis_df["failures"].mean())
                    if "failures" in analysis_df.columns
                    else 0
                )

                advanced_col1, advanced_col2, advanced_col3, advanced_col4 = st.columns(4)

                with advanced_col1:
                    st.metric(
                        "🚨 High-Risk %",
                        f"{high_risk_percentage:.1f}%"
                    )

                with advanced_col2:
                    st.metric(
                        "📚 Avg Study Time",
                        f"{average_study_time:.2f}"
                    )

                with advanced_col3:
                    st.metric(
                        "📅 Avg Absences",
                        f"{average_absences:.2f}"
                    )

                with advanced_col4:
                    st.metric(
                        "❌ Avg Failures",
                        f"{average_failures:.2f}"
                    )

                # ------------------------------------------------------------
                # TOP PERFORMING STUDENTS
                # ------------------------------------------------------------

                render_html("""
                <div class="section-title">
                    🏆 Top Performing Students
                </div>
                """)

                top_students = (
                    results_df
                    .sort_values("Predicted Grade", ascending=False)
                    .head(5)
                )

                top_student_columns = [
                    column
                    for column in [
                        "student_id",
                        "Predicted Grade",
                        "Performance Level",
                        "Risk Level"
                    ]
                    if column in top_students.columns
                ]

                st.dataframe(
                    top_students[top_student_columns],
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------------------
                # STUDENTS NEEDING IMPROVEMENT
                # ------------------------------------------------------------

                render_html("""
                <div class="section-title">
                    📉 Students Needing Improvement
                </div>
                """)

                bottom_students = (
                    results_df
                    .sort_values("Predicted Grade", ascending=True)
                    .head(5)
                )

                bottom_student_columns = [
                    column
                    for column in [
                        "student_id",
                        "Predicted Grade",
                        "Performance Level",
                        "Risk Level"
                    ]
                    if column in bottom_students.columns
                ]

                st.dataframe(
                    bottom_students[bottom_student_columns],
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------------------
                # DOWNLOAD HIGH-RISK STUDENTS
                # ------------------------------------------------------------

                if not high_risk_students.empty:

                    high_risk_csv = (
                        high_risk_students
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        "🚨 Download High-Risk Students",
                        data=high_risk_csv,
                        file_name="high_risk_students.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_high_risk_students"
                    )

                # ------------------------------------------------------------
                # STUDENT SEARCH
                # ------------------------------------------------------------

                render_html("""
                <div class="section-title">
                    🔎 Search Student
                </div>
                """)

                if "student_id" in results_df.columns:

                    search_student = st.text_input(
                        "Enter Student ID",
                        placeholder="Example: STU003",
                        key="batch_student_search"
                    )

                    if search_student.strip():

                        searched_students = results_df[
                            results_df["student_id"]
                            .astype(str)
                            .str.contains(
                                search_student.strip(),
                                case=False,
                                na=False,
                                regex=False
                            )
                        ]

                        if not searched_students.empty:

                            st.success(
                                f"✅ {len(searched_students)} "
                                "student record(s) found."
                            )

                            st.dataframe(
                                searched_students,
                                use_container_width=True,
                                hide_index=True
                            )

                        else:

                            st.warning(
                                "No student found with that ID."
                            )

                # ------------------------------------------------------------
                # PERFORMANCE INSIGHTS
                # ------------------------------------------------------------

                render_html("""
                <div class="section-title">
                    💡 Batch Performance Insights
                </div>
                """)

                best_grade = float(results_df["Predicted Grade"].max())
                lowest_grade = float(results_df["Predicted Grade"].min())
                median_grade = float(results_df["Predicted Grade"].median())

                insight_col1, insight_col2, insight_col3 = st.columns(3)

                with insight_col1:
                    st.metric(
                        "🏆 Highest Predicted Grade",
                        f"{best_grade:.2f}"
                    )

                with insight_col2:
                    st.metric(
                        "📉 Lowest Predicted Grade",
                        f"{lowest_grade:.2f}"
                    )

                with insight_col3:
                    st.metric(
                        "📊 Median Grade",
                        f"{median_grade:.2f}"
                    )

                # ------------------------------------------------------------
                # STUDY TIME VS PREDICTED GRADE
                # ------------------------------------------------------------

                if (
                    "studytime" in analysis_df.columns
                    and "Predicted Grade" in analysis_df.columns
                ):

                    render_html("""
                    <div class="section-title">
                        📚 Study Time vs Predicted Grade
                    </div>
                    """)

                    study_analysis = (
                        analysis_df
                        .groupby("studytime")["Predicted Grade"]
                        .mean()
                        .round(2)
                    )

                    st.bar_chart(
                        study_analysis,
                        use_container_width=True
                    )

                # ------------------------------------------------------------
                # ABSENCES VS PREDICTED GRADE
                # ------------------------------------------------------------

                if (
                    "absences" in analysis_df.columns
                    and "Predicted Grade" in analysis_df.columns
                ):

                    render_html("""
                    <div class="section-title">
                        📅 Absences vs Predicted Grade
                    </div>
                    """)

                    absence_analysis = (
                        analysis_df[
                            ["absences", "Predicted Grade"]
                        ]
                        .sort_values("absences")
                        .set_index("absences")
                    )

                    st.line_chart(
                        absence_analysis,
                        use_container_width=True
                    )

                # ------------------------------------------------------------
                # PREVIOUS FAILURES VS PREDICTED GRADE
                # ------------------------------------------------------------

                if (
                    "failures" in analysis_df.columns
                    and "Predicted Grade" in analysis_df.columns
                ):

                    render_html("""
                    <div class="section-title">
                        ❌ Previous Failures vs Predicted Grade
                    </div>
                    """)

                    failure_analysis = (
                        analysis_df
                        .groupby("failures")["Predicted Grade"]
                        .mean()
                        .round(2)
                    )

                    st.bar_chart(
                        failure_analysis,
                        use_container_width=True
                    )

                # ============================================================
                # 👥 STUDENT COMPARISON + 🔮 WHAT-IF SIMULATOR
                # NEW ADD-ON FEATURE
                # Existing prediction, batch analytics and model code remain
                # unchanged. This section only adds new functionality.
                # ============================================================

                render_html("""
                <div class="section-title">
                    👥 Student Comparison & What-If Simulator
                </div>
                """)

                # ------------------------------------------------------------
                # STUDENT COMPARISON
                # ------------------------------------------------------------

                if "student_id" in results_df.columns and len(results_df) >= 2:

                    st.markdown("### 👥 Compare Two Students")

                    comparison_students = (
                        results_df["student_id"]
                        .astype(str)
                        .tolist()
                    )

                    comparison_col1, comparison_col2 = st.columns(2)

                    with comparison_col1:
                        student_a = st.selectbox(
                            "Select Student A",
                            comparison_students,
                            key="batch_compare_student_a"
                        )

                    with comparison_col2:
                        student_b_options = [
                            student
                            for student in comparison_students
                            if student != student_a
                        ]

                        student_b = st.selectbox(
                            "Select Student B",
                            student_b_options,
                            key="batch_compare_student_b"
                        )

                    row_a = results_df[
                        results_df["student_id"].astype(str) == student_a
                    ].iloc[0]

                    row_b = results_df[
                        results_df["student_id"].astype(str) == student_b
                    ].iloc[0]

                    def safe_value(row, column):
                        if column in row.index:
                            return row[column]
                        return "N/A"

                    comparison_table = pd.DataFrame({
                        "Metric": [
                            "Predicted Grade",
                            "Performance Level",
                            "Risk Level",
                            "Study Time",
                            "Previous Failures",
                            "Absences",
                            "Health",
                            "Age"
                        ],
                        student_a: [
                            f"{float(row_a['Predicted Grade']):.2f}",
                            safe_value(row_a, "Performance Level"),
                            safe_value(row_a, "Risk Level"),
                            safe_value(row_a, "studytime"),
                            safe_value(row_a, "failures"),
                            safe_value(row_a, "absences"),
                            safe_value(row_a, "health"),
                            safe_value(row_a, "age")
                        ],
                        student_b: [
                            f"{float(row_b['Predicted Grade']):.2f}",
                            safe_value(row_b, "Performance Level"),
                            safe_value(row_b, "Risk Level"),
                            safe_value(row_b, "studytime"),
                            safe_value(row_b, "failures"),
                            safe_value(row_b, "absences"),
                            safe_value(row_b, "health"),
                            safe_value(row_b, "age")
                        ]
                    })

                    st.dataframe(
                        comparison_table,
                        use_container_width=True,
                        hide_index=True
                    )

                    grade_a = float(row_a["Predicted Grade"])
                    grade_b = float(row_b["Predicted Grade"])
                    grade_difference = abs(grade_a - grade_b)

                    if grade_a > grade_b:
                        st.info(
                            f"📊 **{student_a}** has a "
                            f"{grade_difference:.2f} point higher predicted "
                            f"grade than **{student_b}**."
                        )
                    elif grade_b > grade_a:
                        st.info(
                            f"📊 **{student_b}** has a "
                            f"{grade_difference:.2f} point higher predicted "
                            f"grade than **{student_a}**."
                        )
                    else:
                        st.info(
                            "📊 Both students have the same predicted grade."
                        )

                elif "student_id" in results_df.columns:
                    st.info(
                        "At least two students are required for comparison."
                    )

                # ------------------------------------------------------------
                # WHAT-IF PERFORMANCE SIMULATOR
                # ------------------------------------------------------------

                render_html("""
                <div class="section-title">
                    🔮 What-If Performance Simulator
                </div>
                """)

                if "student_id" in results_df.columns and len(results_df) > 0:

                    what_if_students = (
                        results_df["student_id"]
                        .astype(str)
                        .tolist()
                    )

                    selected_what_if_student = st.selectbox(
                        "🎓 Select Student for Simulation",
                        what_if_students,
                        key="batch_what_if_student"
                    )

                    selected_result_row = results_df[
                        results_df["student_id"].astype(str)
                        == selected_what_if_student
                    ].iloc[0]

                    selected_batch_index = results_df[
                        results_df["student_id"].astype(str)
                        == selected_what_if_student
                    ].index[0]

                    # batch_input uses the exact 30 model features, so the
                    # simulation uses the same Random Forest pipeline.
                    original_features = batch_input.loc[
                        selected_batch_index
                    ].copy()

                    current_prediction = float(
                        selected_result_row["Predicted Grade"]
                    )

                    st.markdown(
                        f"**Current predicted grade:** "
                        f"### {current_prediction:.2f} / 20"
                    )

                    sim_col1, sim_col2, sim_col3 = st.columns(3)

                    with sim_col1:
                        current_studytime = int(
                            float(original_features["studytime"])
                        )
                        simulated_studytime = st.slider(
                            "📚 Study Time",
                            min_value=1,
                            max_value=4,
                            value=max(1, min(4, current_studytime)),
                            key="batch_sim_studytime"
                        )

                    with sim_col2:
                        current_absences = int(
                            float(original_features["absences"])
                        )
                        simulated_absences = st.number_input(
                            "📅 Absences",
                            min_value=0,
                            max_value=93,
                            value=max(0, min(93, current_absences)),
                            step=1,
                            key="batch_sim_absences"
                        )

                    with sim_col3:
                        current_failures = int(
                            float(original_features["failures"])
                        )
                        simulated_failures = st.number_input(
                            "❌ Previous Failures",
                            min_value=0,
                            max_value=4,
                            value=max(0, min(4, current_failures)),
                            step=1,
                            key="batch_sim_failures"
                        )

                    st.caption(
                        "Change the academic indicators and see how the "
                        "existing Random Forest model responds. "
                        "This is a simulation, not a guaranteed future grade."
                    )

                    if st.button(
                        "🔮 Simulate New Performance",
                        use_container_width=True,
                        key="batch_run_what_if"
                    ):

                        simulated_features = original_features.copy()

                        simulated_features["studytime"] = simulated_studytime
                        simulated_features["absences"] = simulated_absences
                        simulated_features["failures"] = simulated_failures

                        simulation_input = pd.DataFrame(
                            [simulated_features],
                            columns=FEATURES
                        )

                        try:
                            simulated_prediction = float(
                                model.predict(simulation_input)[0]
                            )

                            simulated_prediction = float(
                                np.clip(
                                    simulated_prediction,
                                    0.0,
                                    20.0
                                )
                            )

                            prediction_change = (
                                simulated_prediction
                                - current_prediction
                            )

                            result_col1, result_col2, result_col3 = st.columns(3)

                            with result_col1:
                                st.metric(
                                    "Current Grade",
                                    f"{current_prediction:.2f}"
                                )

                            with result_col2:
                                st.metric(
                                    "Simulated Grade",
                                    f"{simulated_prediction:.2f}"
                                )

                            with result_col3:
                                st.metric(
                                    "Potential Change",
                                    f"{prediction_change:+.2f}"
                                )

                            if prediction_change > 0.10:
                                st.success(
                                    f"📈 The simulated changes increase the "
                                    f"model prediction by approximately "
                                    f"**{prediction_change:.2f} points**."
                                )
                            elif prediction_change < -0.10:
                                st.warning(
                                    f"📉 The simulated changes decrease the "
                                    f"model prediction by approximately "
                                    f"**{abs(prediction_change):.2f} points**."
                                )
                            else:
                                st.info(
                                    "➡️ The simulated changes have little "
                                    "effect on the model prediction."
                                )

                        except Exception as simulation_error:
                            st.error(
                                "❌ Unable to run the What-If simulation."
                            )
                            st.exception(simulation_error)


                # ============================================================
                # 🎯 AI INTERVENTION IMPACT PLANNER - NEW ADD-ON
                # Existing model, prediction logic, comparison and What-If
                # simulator remain unchanged. This section only adds a
                # recommendation-style intervention analysis.
                # ============================================================

                render_html("""
                <div class="section-title">
                    🎯 AI Intervention Impact Planner
                </div>

                <div class="recommendation">
                    🤖 The existing Random Forest model is used to test
                    practical academic interventions for the selected student.
                    This does not retrain or modify the model.
                </div>
                """)

                if "student_id" in results_df.columns and len(results_df) > 0:

                    intervention_students = (
                        results_df["student_id"]
                        .astype(str)
                        .tolist()
                    )

                    intervention_student = st.selectbox(
                        "🎓 Select Student for Intervention Analysis",
                        intervention_students,
                        key="batch_intervention_student"
                    )

                    intervention_row = results_df[
                        results_df["student_id"].astype(str)
                        == intervention_student
                    ].iloc[0]

                    intervention_index = results_df[
                        results_df["student_id"].astype(str)
                        == intervention_student
                    ].index[0]

                    intervention_features = batch_input.loc[
                        intervention_index
                    ].copy()

                    intervention_current_grade = float(
                        intervention_row["Predicted Grade"]
                    )

                    def intervention_risk(prediction_value, row):
                        intervention_risk_points = 0

                        if float(row["absences"]) >= 15:
                            intervention_risk_points += 2
                        elif float(row["absences"]) >= 8:
                            intervention_risk_points += 1

                        if float(row["failures"]) >= 2:
                            intervention_risk_points += 2
                        elif float(row["failures"]) == 1:
                            intervention_risk_points += 1

                        if float(row["studytime"]) == 1:
                            intervention_risk_points += 1

                        if float(row["health"]) <= 2:
                            intervention_risk_points += 1

                        if prediction_value < 10:
                            intervention_risk_points += 2
                        elif prediction_value < 12:
                            intervention_risk_points += 1

                        if intervention_risk_points >= 4:
                            return "HIGH RISK"
                        elif intervention_risk_points >= 2:
                            return "MEDIUM RISK"
                        return "LOW RISK"

                    # --------------------------------------------------------
                    # BUILD REALISTIC MODEL-BASED INTERVENTION SCENARIOS
                    # --------------------------------------------------------
                    # Instead of changing each feature by one fixed amount,
                    # search realistic values and keep the best prediction
                    # that the EXISTING Random Forest actually produces.
                    # This does NOT retrain or modify the model.

                    current_features = intervention_features.copy()

                    def predict_intervention(features_row):
                        """Run the existing model for one counterfactual row."""
                        scenario_input = pd.DataFrame(
                            [features_row],
                            columns=FEATURES
                        )

                        prediction = float(model.predict(scenario_input)[0])
                        return float(np.clip(prediction, 0.0, 20.0))

                    current_prediction = predict_intervention(current_features)

                    current_studytime = int(float(current_features["studytime"]))
                    current_absences = int(float(current_features["absences"]))
                    current_failures = int(float(current_features["failures"]))

                    # Keep values inside the original dataset's practical scales.
                    current_studytime = int(np.clip(current_studytime, 1, 4))
                    current_absences = max(0, current_absences)
                    current_failures = int(np.clip(current_failures, 0, 4))

                    def search_best_change(feature_name, candidate_values,
                                           minimum_change=True):
                        """Find the best prediction for one intervention feature."""
                        candidates = []

                        for candidate_value in candidate_values:
                            test_features = current_features.copy()
                            test_features[feature_name] = candidate_value

                            try:
                                prediction = predict_intervention(test_features)
                                candidates.append((prediction, candidate_value, test_features))
                            except Exception:
                                continue

                        if not candidates:
                            return None

                        if minimum_change:
                            changed = [
                                item for item in candidates
                                if item[1] != current_features[feature_name]
                            ]
                            if changed:
                                candidates = changed

                        return max(candidates, key=lambda item: item[0])

                    # 1. Study-time intervention: test every higher level.
                    studytime_candidates = list(
                        range(current_studytime + 1, 5)
                    )
                    best_studytime = search_best_change(
                        "studytime",
                        studytime_candidates
                    )

                    # 2. Attendance intervention: test several realistic
                    # reductions rather than only cutting absences by 50%.
                    attendance_candidates = sorted(set([
                        max(0, current_absences - 1),
                        max(0, current_absences - 2),
                        max(0, current_absences - 3),
                        max(0, current_absences - 5),
                        max(0, int(round(current_absences * 0.75))),
                        max(0, int(round(current_absences * 0.50))),
                        0
                    ]))

                    attendance_candidates = [
                        value for value in attendance_candidates
                        if value < current_absences
                    ]

                    best_attendance = search_best_change(
                        "absences",
                        attendance_candidates
                    )

                    # 3. Previous-failure intervention: test every lower
                    # failure count that is possible for this student.
                    failure_candidates = list(
                        range(current_failures - 1, -1, -1)
                    )
                    best_failures = search_best_change(
                        "failures",
                        failure_candidates
                    )

                    # 4. Combined intervention: search combinations of the
                    # three academic indicators and keep the strongest model
                    # prediction. This is the important improvement over the
                    # previous fixed scenario.
                    combined_candidates = []

                    study_values = list(range(current_studytime, 5))
                    absence_values = sorted(set([
                        current_absences,
                        *attendance_candidates
                    ]))
                    failure_values = list(range(current_failures, -1, -1))

                    for study_value in study_values:
                        for absence_value in absence_values:
                            for failure_value in failure_values:
                                # Skip the unchanged row because Current
                                # Situation already represents it.
                                if (
                                    study_value == current_studytime
                                    and absence_value == current_absences
                                    and failure_value == current_failures
                                ):
                                    continue

                                test_features = current_features.copy()
                                test_features["studytime"] = study_value
                                test_features["absences"] = absence_value
                                test_features["failures"] = failure_value

                                try:
                                    prediction = predict_intervention(test_features)
                                    combined_candidates.append(
                                        (prediction, study_value, absence_value,
                                         failure_value, test_features)
                                    )
                                except Exception:
                                    continue

                    best_combined = (
                        max(combined_candidates, key=lambda item: item[0])
                        if combined_candidates
                        else None
                    )

                    # --------------------------------------------------------
                    # CONVERT SEARCH RESULTS INTO DISPLAY SCENARIOS
                    # --------------------------------------------------------
                    intervention_scenarios = [
                        (
                            "Current Situation",
                            "No changes",
                            current_features
                        )
                    ]

                    if best_studytime is not None:
                        _, best_value, best_features = best_studytime
                        intervention_scenarios.append((
                            "Increase Study Time",
                            f"Increase study time from {current_studytime} to {int(best_value)}",
                            best_features
                        ))
                    else:
                        intervention_scenarios.append((
                            "Increase Study Time",
                            "No higher study-time level available",
                            current_features
                        ))

                    if best_attendance is not None:
                        _, best_value, best_features = best_attendance
                        intervention_scenarios.append((
                            "Improve Attendance",
                            f"Reduce absences from {current_absences} to {int(best_value)}",
                            best_features
                        ))
                    else:
                        intervention_scenarios.append((
                            "Improve Attendance",
                            "No further absence reduction available",
                            current_features
                        ))

                    if best_failures is not None:
                        _, best_value, best_features = best_failures
                        intervention_scenarios.append((
                            "Address Previous Failures",
                            f"Reduce previous failures from {current_failures} to {int(best_value)}",
                            best_features
                        ))
                    else:
                        intervention_scenarios.append((
                            "Address Previous Failures",
                            "No previous failures available to reduce",
                            current_features
                        ))

                    if best_combined is not None:
                        _, best_study, best_absences, best_failure, best_features = best_combined
                        intervention_scenarios.append((
                            "Combined Intervention",
                            (
                                f"Study time {current_studytime}→{best_study}, "
                                f"absences {current_absences}→{best_absences}, "
                                f"failures {current_failures}→{best_failure}"
                            ),
                            best_features
                        ))
                    else:
                        intervention_scenarios.append((
                            "Combined Intervention",
                            "No combined scenario available",
                            current_features
                        ))

                    intervention_results = []

                    for (
                        scenario_name,
                        scenario_action,
                        scenario_features
                    ) in intervention_scenarios:

                        try:
                            scenario_prediction = predict_intervention(
                                scenario_features
                            )

                            scenario_change = (
                                scenario_prediction
                                - intervention_current_grade
                            )

                            scenario_risk = intervention_risk(
                                scenario_prediction,
                                scenario_features
                            )

                            intervention_results.append({
                                "Intervention": scenario_name,
                                "Action": scenario_action,
                                "Predicted Grade": round(
                                    scenario_prediction,
                                    2
                                ),
                                "Change": round(
                                    scenario_change,
                                    2
                                ),
                                "Risk Level": scenario_risk
                            })

                        except Exception:
                            pass

                    if intervention_results:

                        intervention_df = pd.DataFrame(
                            intervention_results
                        )

                        st.dataframe(
                            intervention_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # ----------------------------------------------------
                        # BEST INTERVENTION
                        # ----------------------------------------------------

                        non_current = intervention_df[
                            intervention_df["Intervention"]
                            != "Current Situation"
                        ]

                        if not non_current.empty:

                            best_intervention = non_current.sort_values(
                                "Predicted Grade",
                                ascending=False
                            ).iloc[0]

                            best_change = float(
                                best_intervention["Change"]
                            )

                            if best_change > 0:

                                st.success(
                                    f"🎯 **Recommended intervention:** "
                                    f"{best_intervention['Intervention']} "
                                    f"could produce the strongest simulated "
                                    f"prediction of **"
                                    f"{float(best_intervention['Predicted Grade']):.2f}/20** "
                                    f"({best_change:+.2f} points compared "
                                    f"with the current prediction)."
                                )

                            else:

                                st.info(
                                    "ℹ️ The tested interventions produced "
                                    "little or no improvement in the current "
                                    "model prediction. This shows that the "
                                    "model should be used as a decision-support "
                                    "tool rather than a guaranteed outcome."
                                )

                        # ----------------------------------------------------
                        # VISUAL INTERVENTION COMPARISON
                        # ----------------------------------------------------

                        render_html("""
                        <div class="section-title">
                            📊 Intervention Comparison
                        </div>
                        """)

                        intervention_chart = (
                            intervention_df[
                                ["Intervention", "Predicted Grade"]
                            ]
                            .set_index("Intervention")
                        )

                        st.bar_chart(
                            intervention_chart,
                            use_container_width=True
                        )

                        # ----------------------------------------------------
                        # RISK CHANGE
                        # ----------------------------------------------------

                        current_intervention_risk = intervention_risk(
                            intervention_current_grade,
                            current_features
                        )

                        combined_result = intervention_df[
                            intervention_df["Intervention"]
                            == "Combined Intervention"
                        ]

                        if not combined_result.empty:

                            combined_risk = combined_result.iloc[0][
                                "Risk Level"
                            ]

                            render_html(f"""
                            <div class="recommendation">
                                🚦 <b>Risk Projection:</b>
                                Current risk is
                                <b>{current_intervention_risk}</b>.
                                After the combined simulated intervention,
                                the projected risk is
                                <b>{combined_risk}</b>.
                                <br><br>
                                ⚠️ This is a model simulation and should not
                                be interpreted as a guaranteed academic result.
                            </div>
                            """)


                # ============================================================
                # 🎯 AI STUDENT ACTION PLAN - NEW ADD-ON
                # Existing prediction, SHAP, batch analytics, comparison,
                # What-If simulator and Intervention Planner remain unchanged.
                # This section only converts the intervention result into a
                # practical 30-day academic action plan.
                # ============================================================

                if intervention_results and intervention_student:

                    render_html("""
                    <div class="section-title">
                        🎯 AI Student Action Plan
                    </div>

                    <div class="recommendation">
                        🧠 This action plan converts the existing model
                        simulation into practical academic steps for the
                        selected student. It does not retrain or modify the
                        Random Forest model.
                    </div>
                    """)

                    action_current_grade = float(
                        intervention_current_grade
                    )

                    action_current_risk = intervention_risk(
                        action_current_grade,
                        current_features
                    )

                    # Use the best POSITIVE intervention for the action plan.
                    # If none of the tested interventions improves the model
                    # prediction, keep the current grade as the planning target
                    # instead of presenting a negative simulation as a goal.
                    action_candidates = intervention_df[
                        intervention_df["Intervention"]
                        != "Current Situation"
                    ].copy()

                    if not action_candidates.empty:
                        action_candidates = action_candidates.sort_values(
                            "Predicted Grade",
                            ascending=False
                        )
                        action_best_row = action_candidates.iloc[0]
                        action_best_change = float(action_best_row["Change"])

                        if action_best_change > 0:
                            action_recommended_intervention = str(
                                action_best_row["Intervention"]
                            )
                            action_target_grade = float(
                                action_best_row["Predicted Grade"]
                            )
                            action_expected_change = action_best_change
                        else:
                            action_recommended_intervention = (
                                "No positive intervention identified"
                            )
                            action_target_grade = action_current_grade
                            action_expected_change = 0.0
                    else:
                        action_recommended_intervention = (
                            "No intervention scenario available"
                        )
                        action_target_grade = action_current_grade
                        action_expected_change = 0.0

                    action_absences = int(
                        float(intervention_features["absences"])
                    )
                    action_failures = int(
                        float(intervention_features["failures"])
                    )
                    action_studytime = int(
                        float(intervention_features["studytime"])
                    )
                    action_health = int(
                        float(intervention_features["health"])
                    )

                    # --------------------------------------------------------
                    # PRIORITY DETECTION
                    # --------------------------------------------------------

                    action_priorities = []

                    if action_absences >= 15:
                        action_priorities.append(
                            "🔴 Attendance is a critical priority."
                        )
                    elif action_absences >= 8:
                        action_priorities.append(
                            "🟠 Attendance should be improved."
                        )
                    else:
                        action_priorities.append(
                            "🟢 Attendance is currently within a manageable range."
                        )

                    if action_failures >= 2:
                        action_priorities.append(
                            "🔴 Previous failures require focused academic support."
                        )
                    elif action_failures == 1:
                        action_priorities.append(
                            "🟠 Review the subject areas associated with the previous failure."
                        )
                    else:
                        action_priorities.append(
                            "🟢 No previous failures are currently recorded."
                        )

                    if action_studytime <= 1:
                        action_priorities.append(
                            "🔴 Study-time routine should be strengthened."
                        )
                    elif action_studytime == 2:
                        action_priorities.append(
                            "🟠 Study time can be increased gradually."
                        )
                    else:
                        action_priorities.append(
                            "🟢 Study-time level is relatively strong."
                        )

                    if action_health <= 2:
                        action_priorities.append(
                            "🟠 Maintain healthy daily routines and seek appropriate support when needed."
                        )

                    # --------------------------------------------------------
                    # DASHBOARD METRICS
                    # --------------------------------------------------------

                    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

                    with action_col1:
                        st.metric(
                            "Current Grade",
                            f"{action_current_grade:.2f}/20"
                        )

                    with action_col2:
                        st.metric(
                            "Simulated Target",
                            f"{action_target_grade:.2f}/20",
                            f"{action_expected_change:+.2f}"
                        )

                    with action_col3:
                        st.metric(
                            "Current Risk",
                            action_current_risk
                        )

                    with action_col4:
                        st.metric(
                            "Planning Horizon",
                            "30 Days"
                        )

                    # --------------------------------------------------------
                    # PRIORITY SUMMARY
                    # --------------------------------------------------------

                    render_html("""
                    <div class="section-title">
                        🚦 Priority Assessment
                    </div>
                    """)

                    priority_df = pd.DataFrame({
                        "Priority Area": [
                            "Attendance",
                            "Previous Failures",
                            "Study Time",
                            "Health"
                        ],
                        "Current Value": [
                            action_absences,
                            action_failures,
                            action_studytime,
                            action_health
                        ],
                        "Recommended Focus": [
                            "Maintain / improve attendance",
                            "Target weak academic areas",
                            "Build a consistent study routine",
                            "Maintain healthy routines"
                        ]
                    })

                    st.dataframe(
                        priority_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # --------------------------------------------------------
                    # 30-DAY PLAN
                    # --------------------------------------------------------

                    render_html("""
                    <div class="section-title">
                        🗓️ 30-Day Intervention Roadmap
                    </div>
                    """)

                    action_plan = pd.DataFrame({
                        "Period": [
                            "Days 1-7",
                            "Days 8-14",
                            "Days 15-21",
                            "Days 22-30"
                        ],
                        "Primary Goal": [
                            "Build a consistent academic routine",
                            "Strengthen weak subject areas",
                            "Monitor attendance and study consistency",
                            "Review progress and repeat prediction"
                        ],
                        "Recommended Action": [
                            "Set a daily study schedule and minimize avoidable absences.",
                            "Spend additional study time on difficult subjects and review previous mistakes.",
                            "Track study sessions, attendance and academic progress each week.",
                            "Re-enter updated student indicators into the system and compare the new prediction."
                        ],
                        "Success Check": [
                            "Study routine established",
                            "Weak topics identified and practiced",
                            "Attendance and study habits maintained",
                            "Progress reviewed with the model and academic support"
                        ]
                    })

                    st.dataframe(
                        action_plan,
                        use_container_width=True,
                        hide_index=True
                    )

                    # --------------------------------------------------------
                    # PERSONALIZED RECOMMENDATIONS
                    # --------------------------------------------------------

                    render_html("""
                    <div class="section-title">
                        💡 Personalized Recommendations
                    </div>
                    """)

                    for priority_message in action_priorities:
                        st.write(f"• {priority_message}")

                    # --------------------------------------------------------
                    # MODEL-BASED GOAL
                    # --------------------------------------------------------

                    if action_expected_change > 0:
                        st.success(
                            f"🎯 **Model-based planning goal:** "
                            f"the best positive simulated intervention is "
                            f"**{action_recommended_intervention}**, changing "
                            f"the prediction from **{action_current_grade:.2f}** "
                            f"to **{action_target_grade:.2f}/20**, a simulated "
                            f"change of **{action_expected_change:+.2f} points**."
                        )
                    else:
                        st.info(
                            "ℹ️ None of the tested interventions produced a "
                            "positive model change for this student. The "
                            "action plan therefore uses the current prediction "
                            "as the planning baseline instead of presenting a "
                            "negative simulation as a target."
                        )

                    # --------------------------------------------------------
                    # DOWNLOADABLE ACTION PLAN
                    # --------------------------------------------------------

                    download_plan = action_plan.copy()

                    download_plan.insert(
                        0,
                        "student_id",
                        intervention_student
                    )

                    download_plan.insert(
                        1,
                        "current_predicted_grade",
                        round(action_current_grade, 2)
                    )

                    download_plan.insert(
                        2,
                        "simulated_target_grade",
                        round(action_target_grade, 2)
                    )

                    download_plan.insert(
                        3,
                        "simulated_change",
                        round(action_expected_change, 2)
                    )

                    download_plan.insert(
                        4,
                        "recommended_intervention",
                        action_recommended_intervention
                    )

                    download_plan.insert(
                        5,
                        "current_risk",
                        action_current_risk
                    )

                    st.download_button(
                        "📥 Download 30-Day Student Action Plan (CSV)",
                        download_plan.to_csv(index=False).encode("utf-8"),
                        file_name=f"{intervention_student}_30_day_action_plan.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    render_html("""
                    <div class="recommendation">
                        ⚠️ <b>Important:</b> This is a machine-learning
                        decision-support plan. It is not a guaranteed grade
                        prediction, diagnosis, or substitute for teacher,
                        parent, counselor, or institutional judgment.
                    </div>
                    """)


                # ------------------------------------------------------------
                # FINAL BATCH INSIGHT
                # ------------------------------------------------------------

                if high_risk_count > 0:

                    render_html(f"""
                    <div class="risk-high">

                        <div class="risk-title">
                            🚨 Intervention Alert
                        </div>

                        <div class="risk-description">

                            {high_risk_count} out of {total_students}
                            students ({high_risk_percentage:.1f}%)
                            are currently classified as HIGH RISK.

                            <br><br>

                            These students should be prioritized for
                            academic monitoring and additional support.

                        </div>

                    </div>
                    """)

                else:

                    render_html("""
                    <div class="risk-low">

                        <div class="risk-title">
                            🟢 Batch Status
                        </div>

                        <div class="risk-description">

                            No students in this batch are currently
                            classified as HIGH RISK.

                        </div>

                    </div>
                    """)

    except Exception as e:

        st.error(
            "❌ Batch prediction failed."
        )

        st.write(
            "Please verify that the CSV columns and "
            "data types match the provided template."
        )

        st.exception(e)

# ============================================================
# BEFORE PREDICTION FOOTER
# ============================================================

if not st.session_state.prediction_history:

    render_html("""
    <div class="divider"></div>

    <div class="footer">

        🎓 <b>Student Performance AI</b>

        <br>

        Enter student information above and click
        <b>Predict Student Performance</b>.

    </div>
    """)
