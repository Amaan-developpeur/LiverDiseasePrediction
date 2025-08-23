import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTEN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
import joblib
import wandb
from pathlib import Path

def train_rf_model(cleaned_indian_liver_patient):
    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_FILE = BASE_DIR / cleaned_indian_liver_patient
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / "rf_model.pkl"

    # Initialize W&B
    wandb.init(project="LiverDiseasePrediction")

    # Hyperparameters
    config = wandb.config
    config.n_estimators = 100
    config.max_depth = 5

    # Load data
    df = pd.read_csv(CSV_FILE)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    X = df.drop(columns=["Dataset"])
    y = df["Dataset"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smoten = SMOTEN(random_state=8)
    X_resampled, y_resampled = smoten.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=config.n_estimators,
                                      max_depth=config.max_depth,
                                      random_state=42)
    rf_model.fit(X_resampled, y_resampled)

    y_test_predict = rf_model.predict(X_test)
    recall = recall_score(y_test, y_test_predict)
    print(f"Recall: {recall}")
    print(classification_report(y_test, y_test_predict))

    wandb.log({"recall": recall})
    joblib.dump(rf_model, MODEL_DIR / "rf_model.pkl")
    wandb.save(str(MODEL_DIR / "rf_model.pkl"))

if __name__ == "__main__":
    train_rf_model("cleaned_indian_liver_patient.csv")
