"""
Case Study 1: Hospital 30-Day Readmission Prediction
----------------------------------------------------
Logistic Regression with L2 regularization.

Expected dataset:
Diabetes 130-US hospitals for years 1999-2008
CSV filename: data/diabetic_data.csv

Target:
readmitted == "<30" -> 1 (readmitted within 30 days)
otherwise -> 0

The script:
1. Cleans the hospital dataset.
2. Uses diagnosis codes, age, admission/discharge information,
   prior inpatient/emergency visits and other patient-record features.
3. One-hot encodes categorical variables and imputes missing values.
4. Trains Logistic Regression with L2 regularization.
5. Evaluates using ROC-AUC.
6. Shows the effect of changing the decision threshold.
7. Discusses false-negative vs false-positive clinical cost.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/diabetic_data.csv")
RANDOM_STATE = 42


def load_and_prepare_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {path}\n"
            "Download the Diabetes 130-US hospitals dataset and place "
            "diabetic_data.csv inside the data/ folder.\n"
        )

    df = pd.read_csv(path)

    # Remove identifiers and columns that can leak the outcome.
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # The assignment is specifically about 30-day readmission.
    if "readmitted" not in df.columns:
        raise ValueError("Column 'readmitted' was not found in the dataset.")

    df["target"] = (df["readmitted"].astype(str).str.strip() == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Treat '?' as missing.
    df = df.replace("?", pd.NA)

    # Remove columns with too much missing data.
    missing_ratio = df.isna().mean()
    too_missing = missing_ratio[missing_ratio > 0.90].index.tolist()
    df = df.drop(columns=too_missing)

    # Common numeric hospital-utilization features.
    numeric_candidates = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]

    # Diagnosis columns are useful predictors for clinical risk.
    diagnosis_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]

    # Age is categorical in this dataset and will be one-hot encoded.
    # Other categorical patient/admission variables are also retained.
    feature_cols = numeric_cols + diagnosis_cols

    categorical_candidates = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "max_glu_serum",
        "A1Cresult",
        "change",
        "diabetesMed",
    ]
    feature_cols += [c for c in categorical_candidates if c in df.columns]

    # Keep only columns available in this dataset.
    feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # IDs are categorical codes rather than continuous measurements.
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_cols = [c for c in X.columns if c in numeric_cols]

    return X, y, numeric_cols, categorical_cols


def build_model(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    # L2 is the default regularization penalty for this solver.
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def evaluate_at_threshold(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )

    print(f"\nThreshold = {threshold:.2f}")
    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")

    return predictions


def main():
    X, y, numeric_cols, categorical_cols = load_and_prepare_data(DATA_PATH)

    print("Dataset shape:", X.shape)
    print("30-day readmission rate:", f"{y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model(numeric_cols, categorical_cols)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    print("\n========== MODEL RESULT ==========")
    print(f"ROC-AUC: {auc:.4f}")

    # Standard 0.50 threshold.
    predictions_50 = evaluate_at_threshold(
        y_test, probabilities, threshold=0.50
    )

    print("\nClassification report at threshold 0.50:")
    print(
        classification_report(
            y_test,
            predictions_50,
            target_names=["No 30-day readmission", "30-day readmission"],
            zero_division=0,
        )
    )

    # Lower threshold: useful when missing a high-risk patient is costly.
    predictions_30 = evaluate_at_threshold(
        y_test, probabilities, threshold=0.30
    )

    # ROC curve.
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - 30-Day Hospital Readmission")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=160)
    plt.show()

    # Confusion matrix for the lower threshold.
    cm = confusion_matrix(y_test, predictions_30, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No readmission", "Readmitted <30d"],
    )
    disp.plot()
    plt.title("Confusion Matrix at Threshold 0.30")
    plt.tight_layout()
    plt.savefig("confusion_matrix_threshold_30.png", dpi=160)
    plt.show()

    print("\n========== CLINICAL COST DISCUSSION ==========")
    print(
        "False negative: the model predicts low risk but the patient is "
        "readmitted within 30 days. In a clinical screening workflow, "
        "this can be costly because an at-risk patient may not receive "
        "additional follow-up or discharge planning."
    )
    print(
        "False positive: the model predicts high risk but the patient "
        "is not readmitted. This may consume extra staff time, follow-up "
        "resources, or monitoring capacity."
    )
    print(
        "Therefore, a threshold below 0.50 can increase recall and reduce "
        "false negatives, but it generally creates more false positives. "
        "The final threshold should be selected with clinicians and based "
        "on the relative costs of the two error types."
    )


if __name__ == "__main__":
    main()
