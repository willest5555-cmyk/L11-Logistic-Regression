"""
Iris classification with scikit-learn following the CRISP-DM process.

Run:
    python regression_crisp_dm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def print_step(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    # 1. Business Understanding
    print_step("1. Business Understanding")
    print(
        "Goal: classify iris flowers into setosa, versicolor, or virginica "
        "using sepal and petal measurements."
    )
    print(
        "Success criterion: build an interpretable baseline model with strong "
        "test accuracy and clear evaluation metrics."
    )

    # 2. Data Understanding
    print_step("2. Data Understanding")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    df = X.copy()
    df["species"] = y.map(dict(enumerate(target_names)))

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFeature preview:")
    print(df.head())
    print("\nClass distribution:")
    print(df["species"].value_counts())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nFeature summary:")
    print(X.describe())

    # 3. Data Preparation
    print_step("3. Data Preparation")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print(
        "Preparation choices: stratified train/test split and feature "
        "standardization inside a pipeline."
    )

    # 4. Modeling
    print_step("4. Modeling")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(X_train, y_train)
    print("Model trained: StandardScaler + LogisticRegression")

    # 5. Evaluation
    print_step("5. Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = list(range(len(target_names)))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"Test accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=target_names, columns=target_names))

    OUTPUT_DIR.mkdir(exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=target_names,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Iris Classification Confusion Matrix")
    plt.tight_layout()
    matrix_path = OUTPUT_DIR / "iris_confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {matrix_path}")

    # 6. Deployment
    print_step("6. Deployment")
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=iris.feature_names,
    )
    predicted_class = model.predict(sample)[0]
    predicted_probabilities = model.predict_proba(sample)[0]

    print("Example prediction input:")
    print(sample)
    print(f"Predicted species: {target_names[predicted_class]}")
    print("\nPrediction probabilities:")
    for name, probability in zip(target_names, predicted_probabilities):
        print(f"  {name}: {probability:.3f}")
    print(
        "\nDeployment note: this pipeline object can be saved with joblib and "
        "loaded later for batch or API predictions."
    )


if __name__ == "__main__":
    main()
