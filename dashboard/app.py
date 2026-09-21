from pathlib import Path
import sys

import joblib
import pandas as pd
from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.detect import load_features


MODEL_PATH = PROJECT_ROOT / "models" / "nids_model.joblib"
DEFAULT_RESULTS = PROJECT_ROOT / "logs" / "detection_results.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
model = joblib.load(MODEL_PATH)
expected_features = list(model.feature_names_in_)


def attack_probabilities(features):
    probabilities = model.predict_proba(features)
    attack_columns = [
        index for index, label in enumerate(model.classes_)
        if str(label).strip().lower() != "benign"
    ]
    return probabilities[:, attack_columns].sum(axis=1)


def predict_data(data):
    features, valid_rows = load_features(data, expected_features)
    data = data.loc[valid_rows].copy()
    predictions = model.predict(features)
    probabilities = attack_probabilities(features)

    results = pd.DataFrame(
        {
            "prediction": [
                "Attack" if str(value).strip().lower() != "benign" else "Benign"
                for value in predictions
            ],
            "attack_type": [
                str(value) if str(value).strip().lower() != "benign" else "Benign"
                for value in predictions
            ],
            "attack_probability": probabilities,
        }
    )
    if "Label" in data.columns:
        results.insert(0, "original_label", data["Label"].astype(str).values)
    return results


def dashboard_data(results):
    attack_count = int((results["prediction"] == "Attack").sum())
    total_count = len(results)
    benign_count = total_count - attack_count
    attack_rate = (attack_count / total_count * 100) if total_count else 0

    display_results = results.copy()
    display_results["attack_probability"] = (
        display_results["attack_probability"] * 100
    ).round(2)
    return {
        "total_count": total_count,
        "attack_count": attack_count,
        "benign_count": benign_count,
        "attack_rate": round(attack_rate, 2),
        "results": display_results.tail(25).iloc[::-1].to_dict("records"),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if not uploaded_file or not uploaded_file.filename:
            error = "Choose a CSV file before submitting."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only CSV files are supported."
        else:
            try:
                data = pd.read_csv(uploaded_file, low_memory=False)
                results = dashboard_data(predict_data(data))
            except (ValueError, pd.errors.ParserError) as exc:
                error = str(exc)

    if results is None and DEFAULT_RESULTS.exists():
        saved_results = pd.read_csv(DEFAULT_RESULTS)
        results = dashboard_data(saved_results)

    return render_template("index.html", results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)
