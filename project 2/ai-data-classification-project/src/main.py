"""Beginner-friendly supervised classification project using the Iris dataset."""

from pathlib import Path
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"


def main() -> None:
    # 1. Load and understand the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("=" * 60)
    print("AI PROJECT 2: DATA CLASSIFICATION USING AI")
    print("=" * 60)
    print(f"Dataset: Iris")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {', '.join(iris.feature_names)}")
    print(f"Classes: {', '.join(iris.target_names)}")

    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 3. Train a simple supervised classification algorithm
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # 4. Test and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )

    # 5. Predict a new sample (measurements in cm)
    sample_flower = [[5.1, 3.5, 1.4, 0.2]]
    predicted_class = model.predict(sample_flower)[0]
    predicted_name = iris.target_names[predicted_class]

    print("\nMODEL EVALUATION")
    print("-" * 60)
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)
    print("\nSAMPLE PREDICTION")
    print("-" * 60)
    print(f"Measurements: {sample_flower[0]}")
    print(f"Predicted flower species: {predicted_name}")

    # Save reproducible results to a text file
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_path = OUTPUT_DIR / "evaluation.txt"
    result_path.write_text(
        "AI Project 2: Data Classification Using AI\n"
        f"Run time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Dataset: Iris ({X.shape[0]} samples, {X.shape[1]} features)\n"
        f"Algorithm: K-Nearest Neighbors (k=3)\n"
        f"Train/test split: {len(X_train)}/{len(X_test)} (80%/20%)\n"
        f"Test accuracy: {accuracy:.2%}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix:\n{matrix}\n\n"
        f"Sample measurements: {sample_flower[0]}\n"
        f"Predicted species: {predicted_name}\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to: {result_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
