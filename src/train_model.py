from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


TARGET = "perf_band_next"

# These columns are identifiers or audit-only fields, not model inputs.
EXCLUDED_COLUMNS = [
    "employee_id",
    "gender_audit_only",
    TARGET
]


def train():
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    df = pd.read_csv("data/employee_performance.csv")

    X = df.drop(columns=EXCLUDED_COLUMNS)
    y = df[TARGET]

    # 60% train, 20% validation, 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=42, stratify=y_temp
    )

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2
        )
    }

    comparison = []
    best_name = None
    best_score = -1

    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)

        score = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        comparison.append({
            "model": name,
            "validation_macro_f1": score
        })

        print(f"{name}: validation macro-F1 = {score:.3f}")

        if score > best_score:
            best_score = score
            best_name = name

    pd.DataFrame(comparison).to_csv(
        "outputs/model_comparison.csv", index=False
    )

    # Refit selected model on train + validation data.
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", candidates[best_name])
    ])

    final_model.fit(X_train_full, y_train_full)

    # Evaluate the final model once on held-out test data.
    test_predictions = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, test_predictions)
    macro_f1 = f1_score(
        y_test, test_predictions, average="macro", zero_division=0
    )

    report = classification_report(
        y_test, test_predictions, output_dict=True, zero_division=0
    )

    metrics = {
        "selected_model": best_name,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_classification_report": report,
        "note": "Metrics are from a synthetic educational dataset."
    }

    with open("outputs/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nFinal test results")
    print(f"Selected model: {best_name}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test macro-F1: {macro_f1:.3f}")
    print(classification_report(
        y_test, test_predictions, zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test, test_predictions, xticks_rotation=30
    )
    plt.title("Confusion Matrix - Synthetic Test Data")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Permutation importance is calculated on held-out test features.
    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    }).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(
        "outputs/feature_importance.csv", index=False
    )

    plt.figure(figsize=(9, 6))
    top = importance_df.head(12).sort_values("importance_mean")
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Decrease in macro-F1 after permutation")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    joblib.dump(final_model, "models/employee_performance_model.joblib")
    print("\nModel saved in models/employee_performance_model.joblib")


if __name__ == "__main__":
    train()