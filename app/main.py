from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# --- Setup ---
app = FastAPI(title="Liver Disease Prediction Using Machine Learning")
BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# --- Load model and data ---
# Load model first
model = joblib.load(BASE_DIR / "Ada_Model.pkl")

# Load train and test data
X_train = joblib.load(BASE_DIR / "X_resampled.pkl")
y_train = joblib.load(BASE_DIR / "y_resampled.pkl")
X_test = joblib.load(BASE_DIR / "X_test_lime.pkl")
y_test = joblib.load(BASE_DIR / "y_test_lime.pkl")

model.fit(X_train, y_train)

# Feature names
feature_names = X_train.columns.tolist()

# Predict test set
y_pred = model.predict(X_test)

# Evaluation metrics
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame for rendering
cm_df = pd.DataFrame(
    cm, index=["Actual [No Liver Disease]", "Actual [Liver Disease]"], columns=["Predicted [No Liver Disease]", "Predicted [Liver Disease]"]
    )
report_df = pd.DataFrame(report).transpose()

# Setup LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["No Liver Disease", "Liver Disease"],
    mode="classification",
    discretize_continuous=True
)



# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "features": feature_names
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    try:
        input_data = [float(form[feat]) for feat in feature_names]
    except ValueError:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "features": feature_names,
            "error": "Please enter valid numeric values."
        })

    input_array = np.array(input_data).reshape(1, -1)
    pred_label = model.predict(input_array)[0]
    pred_proba = model.predict_proba(input_array)[0]
    pred_class = "Liver Disease" if pred_label == 1 else "No Liver Disease"
    confidence = f"{pred_proba[pred_label]:.2f}"

    explanation = explainer.explain_instance(
        data_row=input_array[0],
        predict_fn=model.predict_proba,
        num_features=10
    )

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": pred_class,
        "probability": confidence,
        "lime_explanation": explanation.as_list(),
        "input_data": dict(zip(feature_names, input_data))
    })



@app.get("/metrics", response_class=HTMLResponse)
async def show_metrics(request: Request):
    return templates.TemplateResponse("metrics.html", {
        "request": request,
        "confusion_matrix": cm_df.to_html(classes="table-auto border"),
        "classification_report": report_df.to_html(classes="table-auto border")
    })
