from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "processed" / "cleaned_ids2018.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "nids_model.joblib"


def main():
    print(f"Loading cleaned dataset: {DATASET_PATH}")
    data = pd.read_csv(DATASET_PATH)

    if "Label" not in data.columns:
        raise ValueError("The dataset must contain a 'Label' column")

    data["Label"] = data["Label"].astype(str).str.strip()
    features = data.drop(columns=["Label"])
    labels = data["Label"]

    print(f"Rows: {len(data):,}")
    print(f"Features: {features.shape[1]}")
    print("Label counts:")
    print(labels.value_counts().to_string())

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, predictions))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
