from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTEN
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split

import wandb


def train_ada_model(cleaned_indian_liver_patient):
    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_FILE = BASE_DIR / cleaned_indian_liver_patient
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / "ada_model.pkl"

    # Initialize W&B
    wandb.init(project="LiverDiseasePrediction")

    # Hyperparameters
    config = wandb.config
    config.n_estimators = 250
    config.learning_rate = 1.0

    # Load data
    df = pd.read_csv(CSV_FILE)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    X = df.drop(columns=["Dataset"])
    y = df["Dataset"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=8
    )

    smoten = SMOTEN(random_state=8)
    X_resampled, y_resampled = smoten.fit_resample(X_train, y_train)

    ada_model = AdaBoostClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        random_state=42,
    )
    ada_model.fit(X_resampled, y_resampled)

    y_test_predict = ada_model.predict(X_test)
    recall = recall_score(y_test, y_test_predict)
    print(f"Recall: {recall}")
    print(classification_report(y_test, y_test_predict))

    wandb.log({"recall": recall})
    joblib.dump(ada_model, MODEL_DIR / "ada_model.pkl")
    artifact = wandb.Artifact(name="ada_model", type="model")
    artifact.add_file(MODEL_DIR / "ada_model.pkl")
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    train_ada_model("cleaned_indian_liver_patient.csv")
