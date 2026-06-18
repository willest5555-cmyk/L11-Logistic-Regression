"""
Generate data for an interactive 3D Iris PCA demo.

Run:
    python iris_pca_3d_demo.py

Then open:
    http://127.0.0.1:8000/iris_pca_3d_demo.html
"""

import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "iris_pca_points.json"


def print_step(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    print_step("1. Business Understanding")
    print("Goal: create an interactive 3D PCA view of the Iris classification data.")

    print_step("2. Data Understanding")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    print(f"Loaded {len(X)} samples with {X.shape[1]} numeric features.")

    print_step("3. Data Preparation")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print("Standardized features and projected them onto the first three PCs.")

    print_step("4. Modeling")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Baseline logistic regression test accuracy: {accuracy:.3f}")

    print_step("5. Evaluation")
    explained = pca.explained_variance_ratio_
    print(
        "Explained variance: "
        + ", ".join(f"PC{i + 1}={value:.3f}" for i, value in enumerate(explained))
    )

    print_step("6. Deployment")
    points = []
    for row_index, (coords, target) in enumerate(zip(X_pca, y)):
        points.append(
            {
                "id": int(row_index),
                "species": str(target_names[target]),
                "classIndex": int(target),
                "pc1": round(float(coords[0]), 6),
                "pc2": round(float(coords[1]), 6),
                "pc3": round(float(coords[2]), 6),
            }
        )

    payload = {
        "title": "First three PCA dimensions",
        "featureNames": list(iris.feature_names),
        "targetNames": list(target_names),
        "explainedVarianceRatio": [round(float(value), 6) for value in explained],
        "modelAccuracy": round(float(accuracy), 6),
        "points": points,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote demo data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
