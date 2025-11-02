"""
phishing_api.py

Author: tauimonen
Date: 2025-11-02
Description:
    FastAPI REST API for phishing site detection using the Decision Tree model
    defined in decision_tree.py.

    The API provides endpoints to:
        - Train a model on a CSV dataset
        - Make predictions using the trained model
        - Retrieve training metrics

Dependencies:
    - fastapi
    - uvicorn
    - pandas
    - scikit-learn
"""

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import pandas as pd
import tempfile
from decision_tree import run_decision_tree

app = FastAPI(
    title="Phishing Site Detection API",
    description="API for training and using a Decision Tree model to detect phishing websites.",
    version="1.0.0",
)

# Global references to model and metrics
trained_model = None
last_metrics = None

# Default feature list (must match decision_tree.py defaults)
DEFAULT_FEATURES = [
    "Shortining_Service",
    "HTTPS_token",
    "Abnormal_URL",
    "SSLfinal_State",
    "Favicon",
    "port",
    "on_mouseover",
    "Submitting_to_email",
]


# ---------------------- TRAIN ENDPOINT ----------------------
@app.post("/train")
async def train_model(
    file: UploadFile = File(...), max_depth: int = Form(3), cv_folds: int = Form(5)
):
    """
    Train a Decision Tree model using a CSV file upload.
    """
    global trained_model, last_metrics

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Train the model using the existing decision_tree.py function
    result = run_decision_tree(
        csv_path=tmp_path, max_depth=max_depth, cv_folds=cv_folds
    )

    trained_model = result["model"]
    last_metrics = result

    return {
        "message": "Model trained successfully.",
        "accuracy": result["accuracy"],
        "cv_mean": result["cv_mean"],
        "features_used": DEFAULT_FEATURES,
    }


# ---------------------- PREDICT ENDPOINT ----------------------
class PhishingInput(BaseModel):
    Shortining_Service: int
    HTTPS_token: int
    Abnormal_URL: int
    SSLfinal_State: int
    Favicon: int
    port: int
    on_mouseover: int
    Submitting_to_email: int


@app.post("/predict")
def predict_site(data: PhishingInput):
    """
    Predict whether a website is phishing or not.
    """
    if trained_model is None:
        return {"error": "Model not trained yet. Please call /train first."}

    X = pd.DataFrame([data.dict()])
    prediction = trained_model.predict(X)[0]
    label = "Phishing" if prediction == 1 else "Not phishing"

    return {"prediction": label, "raw_value": int(prediction)}


# ---------------------- METRICS ENDPOINT ----------------------
@app.get("/metrics")
def get_metrics():
    """
    Retrieve the latest model performance metrics.
    """
    if last_metrics is None:
        return {"error": "No model has been trained yet."}

    return {
        "accuracy": last_metrics["accuracy"],
        "cv_mean": last_metrics["cv_mean"],
        "confusion_matrix": last_metrics["confusion_matrix"].tolist(),
        "classification_report": last_metrics["classification_report"],
    }
