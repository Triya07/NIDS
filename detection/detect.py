import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "processed" / "cleaned_ids2018.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "nids_model.joblib"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "detection_results.csv"


def attack_probabilities(model, features):
    probabilities = model.predict_proba(features)
    attack_columns = [
        index for index, label in enumerate(model.classes_)
        if str(label).strip().lower() != "benign"
    ]
    return probabilities[:, attack_columns].sum(axis=1)


def load_features(data, expected_features):
    features = data.drop(columns=["Label", "Timestamp"], errors="ignore")
    missing_features = [column for column in expected_features if column not in features]
    if missing_features:
        raise ValueError(
            "Input is missing model features: " + ", ".join(missing_features[:10])
        )
    features = features[expected_features].apply(pd.to_numeric, errors="coerce")
    features = features.mask(~np.isfinite(features), np.nan)
    valid_rows = features.notna().all(axis=1)
    return features.loc[valid_rows].astype(float), valid_rows


def detect(input_path, output_path, limit=None):
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(input_path, low_memory=False)

    if limit is not None:
        data = data.head(limit).copy()

    expected_features = list(model.feature_names_in_)
    features, valid_rows = load_features(data, expected_features)
    data = data.loc[valid_rows].copy()
    predictions = model.predict(features)
    probabilities = attack_probabilities(model, features)

    results = pd.DataFrame(
        {
            "prediction": predictions,
            "attack_probability": probabilities.round(6),
        }
    )

    if "Label" in data.columns:
        results.insert(0, "original_label", data["Label"].astype(str).values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Processed rows: {len(results):,}")
    print("Predictions:")
    print(results["prediction"].value_counts())
    print(f"Detection results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Classify network-flow records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    detect(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
