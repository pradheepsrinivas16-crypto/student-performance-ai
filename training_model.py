"""
Student Performance AI
Professional Model Training Pipeline

Run from the project root:

    python train_model.py

This script:
1. Loads student-mat.csv
2. Removes target leakage (G1, G2, G3, final_score)
3. Builds preprocessing + regression pipelines
4. Compares Linear Regression, Random Forest and Gradient Boosting
5. Uses 5-fold cross-validation
6. Tunes Random Forest and Gradient Boosting
7. Selects the best model using test RMSE
8. Saves the selected model and evaluation metrics for Streamlit
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "student-mat.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "student_performance_model.pkl"
)

FEATURES_PATH = os.path.join(
    MODELS_DIR,
    "model_features.pkl"
)

METRICS_PATH = os.path.join(
    MODELS_DIR,
    "model_metrics.pkl"
)

METRICS_JSON_PATH = os.path.join(
    MODELS_DIR,
    "model_metrics.json"
)

COMPARISON_PATH = os.path.join(
    MODELS_DIR,
    "model_comparison.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "RMSE": float(
            np.sqrt(mse)
        ),

        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),

        "MSE": float(mse),
    }


def build_preprocessor(X):
    """Create a robust numerical + categorical preprocessor."""

    numerical_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return (
        preprocessor,
        numerical_features,
        categorical_features
    )


def build_pipeline(preprocessor, estimator):
    """Create the complete preprocessing + model pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                estimator
            )
        ]
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("🎓 STUDENT PERFORMANCE AI")
print("🚀 PROFESSIONAL MODEL TRAINING PIPELINE")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n📂 Loading dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure student-mat.csv is inside the data folder."
    )


# Original UCI Student Performance dataset is tab-separated.
df = pd.read_csv(
    DATA_PATH,
    sep="\t"
)


print(
    f"✅ Dataset loaded: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)


# ============================================================
# VALIDATE TARGET
# ============================================================

if "G3" not in df.columns:

    raise ValueError(
        "Target column 'G3' was not found in the dataset."
    )


# ============================================================
# REMOVE TARGET LEAKAGE
# ============================================================

print("\n🔒 Removing target leakage...")

leakage_columns = [
    "G3",
    "G1",
    "G2",
    "final_score"
]

X = df.drop(
    columns=leakage_columns,
    errors="ignore"
)

y = df["G3"]


print(
    "Removed columns:",
    [
        column
        for column in leakage_columns
        if column in df.columns
    ]
)

print(
    f"Remaining prediction features: {X.shape[1]}"
)


# ============================================================
# FEATURE LIST
# ============================================================

model_features = X.columns.tolist()

print("\n📋 Features used by the model:")

for index, feature in enumerate(
    model_features,
    start=1
):

    print(
        f"{index:2}. {feature}"
    )


# ============================================================
# PREPROCESSING
# ============================================================

print("\n⚙️ Building preprocessing pipeline...")

(
    preprocessor,
    numerical_features,
    categorical_features
) = build_preprocessor(X)


print(
    f"Numerical features: "
    f"{len(numerical_features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
)


print("\n📊 Dataset split:")

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# BASELINE MODELS
# ============================================================

print("\n" + "=" * 70)
print("1️⃣ BASELINE MODEL COMPARISON")
print("=" * 70)


baseline_models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_depth=8,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
}


baseline_results = []


for name, estimator in (
    baseline_models.items()
):

    print(
        f"\n🔄 Evaluating {name}..."
    )

    pipeline = build_pipeline(
        preprocessor,
        estimator
    )

    # Cross-validation using negative RMSE.
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = -cv_scores

    cv_mae_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    cv_mae = -cv_mae_scores

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        predictions
    )

    result = {

        "Model": name,

        "CV_RMSE_Mean":
            float(cv_rmse.mean()),

        "CV_RMSE_Std":
            float(cv_rmse.std()),

        "CV_MAE_Mean":
            float(cv_mae.mean()),

        "Test_RMSE":
            test_metrics["RMSE"],

        "Test_MAE":
            test_metrics["MAE"],

        "Test_R2":
            test_metrics["R2"]
    }

    baseline_results.append(
        result
    )

    print(
        f"   CV RMSE : "
        f"{result['CV_RMSE_Mean']:.4f}"
    )

    print(
        f"   Test RMSE: "
        f"{result['Test_RMSE']:.4f}"
    )

    print(
        f"   Test MAE : "
        f"{result['Test_MAE']:.4f}"
    )

    print(
        f"   Test R²  : "
        f"{result['Test_R2']:.4f}"
    )


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("2️⃣ HYPERPARAMETER OPTIMIZATION")
print("=" * 70)


# ------------------------------------------------------------
# Random Forest tuning
# ------------------------------------------------------------

print("\n🌲 Tuning Random Forest...")

rf_pipeline = build_pipeline(
    preprocessor,
    RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
)


rf_parameters = {

    "model__n_estimators": [
        200,
        400
    ],

    "model__max_depth": [
        5,
        8,
        None
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ],

    "model__max_features": [
        "sqrt",
        0.7
    ]
}


rf_search = GridSearchCV(

    estimator=rf_pipeline,

    param_grid=rf_parameters,

    scoring="neg_root_mean_squared_error",

    cv=cv,

    n_jobs=-1,

    refit=True,

    verbose=0
)


rf_search.fit(
    X_train,
    y_train
)


best_rf = rf_search.best_estimator_

rf_predictions = best_rf.predict(
    X_test
)

rf_metrics = calculate_metrics(
    y_test,
    rf_predictions
)


print(
    "\n🌲 Best Random Forest parameters:"
)

print(
    rf_search.best_params_
)

print(
    f"Best CV RMSE: "
    f"{-rf_search.best_score_:.4f}"
)

print(
    f"Test RMSE: "
    f"{rf_metrics['RMSE']:.4f}"
)

print(
    f"Test MAE: "
    f"{rf_metrics['MAE']:.4f}"
)

print(
    f"Test R²: "
    f"{rf_metrics['R2']:.4f}"
)


# ------------------------------------------------------------
# Gradient Boosting tuning
# ------------------------------------------------------------

print("\n🚀 Tuning Gradient Boosting...")


gb_pipeline = build_pipeline(
    preprocessor,
    GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )
)


gb_parameters = {

    "model__n_estimators": [
        100,
        200
    ],

    "model__learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "model__max_depth": [
        2,
        3
    ],

    "model__min_samples_leaf": [
        1,
        3
    ]
}


gb_search = GridSearchCV(

    estimator=gb_pipeline,

    param_grid=gb_parameters,

    scoring="neg_root_mean_squared_error",

    cv=cv,

    n_jobs=-1,

    refit=True,

    verbose=0
)


gb_search.fit(
    X_train,
    y_train
)


best_gb = gb_search.best_estimator_

gb_predictions = best_gb.predict(
    X_test
)

gb_metrics = calculate_metrics(
    y_test,
    gb_predictions
)


print(
    "\n🚀 Best Gradient Boosting parameters:"
)

print(
    gb_search.best_params_
)

print(
    f"Best CV RMSE: "
    f"{-gb_search.best_score_:.4f}"
)

print(
    f"Test RMSE: "
    f"{gb_metrics['RMSE']:.4f}"
)

print(
    f"Test MAE: "
    f"{gb_metrics['MAE']:.4f}"
)

print(
    f"Test R²: "
    f"{gb_metrics['R2']:.4f}"
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("3️⃣ FINAL MODEL COMPARISON")
print("=" * 70)


comparison = []


# Add baseline models
for result in baseline_results:

    comparison.append({

        "Model":
            result["Model"],

        "CV RMSE":
            result["CV_RMSE_Mean"],

        "Test RMSE":
            result["Test_RMSE"],

        "Test MAE":
            result["Test_MAE"],

        "Test R2":
            result["Test_R2"]
    })


# Add tuned models
comparison.append({

    "Model":
        "Tuned Random Forest",

    "CV RMSE":
        float(
            -rf_search.best_score_
        ),

    "Test RMSE":
        rf_metrics["RMSE"],

    "Test MAE":
        rf_metrics["MAE"],

    "Test R2":
        rf_metrics["R2"]
})


comparison.append({

    "Model":
        "Tuned Gradient Boosting",

    "CV RMSE":
        float(
            -gb_search.best_score_
        ),

    "Test RMSE":
        gb_metrics["RMSE"],

    "Test MAE":
        gb_metrics["MAE"],

    "Test R2":
        gb_metrics["R2"]
})


comparison_df = pd.DataFrame(
    comparison
)


comparison_df = comparison_df.sort_values(
    by="Test RMSE",
    ascending=True
).reset_index(
    drop=True
)


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("4️⃣ BEST MODEL SELECTION")
print("=" * 70)


if (
    comparison_df.iloc[0]["Model"]
    == "Tuned Random Forest"
):

    best_model = best_rf

elif (
    comparison_df.iloc[0]["Model"]
    == "Tuned Gradient Boosting"
):

    best_model = best_gb

else:

    best_model_name = (
        comparison_df.iloc[0]["Model"]
    )

    baseline_pipeline = build_pipeline(
        preprocessor,
        baseline_models[
            best_model_name
        ]
    )

    baseline_pipeline.fit(
        X_train,
        y_train
    )

    best_model = baseline_pipeline


best_model_name = (
    comparison_df.iloc[0]["Model"]
)


best_predictions = best_model.predict(
    X_test
)


best_metrics = calculate_metrics(
    y_test,
    best_predictions
)


print(
    f"\n🏆 Selected model: "
    f"{best_model_name}"
)

print(
    f"R²   : "
    f"{best_metrics['R2']:.4f}"
)

print(
    f"MAE  : "
    f"{best_metrics['MAE']:.4f}"
)

print(
    f"RMSE : "
    f"{best_metrics['RMSE']:.4f}"
)


# ============================================================
# FEATURE IMPORTANCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("5️⃣ FEATURE IMPORTANCE")
print("=" * 70)


final_estimator = best_model.named_steps[
    "model"
]


if hasattr(
    final_estimator,
    "feature_importances_"
):

    preprocessor_final = (
        best_model
        .named_steps["preprocessor"]
    )

    transformed_features = (
        preprocessor_final
        .get_feature_names_out()
    )

    importances = (
        final_estimator
        .feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature":
            transformed_features,

        "Importance":
            importances

    })


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    print(
        "\nTop 15 transformed features:"
    )

    print(
        importance_df
        .head(15)
        .to_string(index=False)
    )

else:

    importance_df = pd.DataFrame()


# ============================================================
# SAVE MODELS
# ============================================================

print("\n" + "=" * 70)
print("6️⃣ SAVING DEPLOYMENT FILES")
print("=" * 70)


os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# Save final pipeline
joblib.dump(
    best_model,
    MODEL_PATH
)


# Save original raw feature names
joblib.dump(
    model_features,
    FEATURES_PATH
)


# Save complete metrics
metrics = {

    "model":
        best_model_name,

    "r2":
        best_metrics["R2"],

    "mae":
        best_metrics["MAE"],

    "rmse":
        best_metrics["RMSE"],

    "mse":
        best_metrics["MSE"],

    "training_samples":
        int(len(X_train)),

    "testing_samples":
        int(len(X_test)),

    "number_of_features":
        int(len(model_features)),

    "cv_folds":
        int(CV_FOLDS),

    "random_state":
        int(RANDOM_STATE),

    "test_size":
        float(TEST_SIZE),

    "best_parameters":
        (
            rf_search.best_params_
            if best_model_name
            == "Tuned Random Forest"

            else
            gb_search.best_params_
            if best_model_name
            == "Tuned Gradient Boosting"

            else {}
        )
}


joblib.dump(
    metrics,
    METRICS_PATH
)


with open(
    METRICS_JSON_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# Save comparison table
comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\n🏆 Best model: "
    f"{best_model_name}"
)

print(
    f"📈 R²: "
    f"{best_metrics['R2']:.4f}"
)

print(
    f"📉 MAE: "
    f"{best_metrics['MAE']:.4f}"
)

print(
    f"📉 RMSE: "
    f"{best_metrics['RMSE']:.4f}"
)

print(
    "\n📁 Deployment files:"
)

print(
    "✅ models/student_performance_model.pkl"
)

print(
    "✅ models/model_features.pkl"
)

print(
    "✅ models/model_metrics.pkl"
)

print(
    "✅ models/model_metrics.json"
)

print(
    "✅ models/model_comparison.csv"
)

print(
    "\n🚀 Your Streamlit app can now use the selected model."
)
