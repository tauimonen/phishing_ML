# Phishing Website Detection Package

## Description
This Python package provides two methods for detecting phishing websites using the phishing dataset:

1. **Decision Tree Classifier** (`decision_tree.py`)  
   A classical machine learning approach using scikit-learn's DecisionTreeClassifier with cross-validation and evaluation metrics.

2. **H2O AutoML** (`automl_model.py`)  
   An automated machine learning pipeline that trains multiple models (GBM, Stacked Ensembles, etc.) and selects the best performing model.

This package allows data scientists to easily experiment with phishing detection models and compare classical ML with AutoML approaches.

---

## Dataset

The dataset is **phishing.csv**, which contains features extracted from websites. Each row corresponds to a website and is labeled as:

- `1` - Phishing site  
- `-1` - Legitimate site  

### Note on Features

The phishing dataset contains over 30 features extracted from websites.  
In the current decision tree module, only 8 selected features are used for simplicity and faster training:  

- Shortining_Service  
- HTTPS_token  
- Abnormal_URL  
- SSLfinal_State  
- Favicon  
- port  
- on_mouseover  
- Submitting_to_email  

H2O AutoML, on the other hand, automatically uses all features and performs internal feature selection.

This dataset is based on the study: [Mohammad et al., 2015](http://eprints.hud.ac.uk/id/eprint/24330/6/MohammadPhishing14July2015.pdf)

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/tauimonen/phishing_ML
cd phishing_ml
pip install -r requirements.txt
```

## Usage
### 1. Decision Tree Classifier
```python
from phishing_ml import decision_tree

# Train and evaluate Decision Tree
results = decision_tree.run_decision_tree("data/phishing.csv")

print("Test Accuracy:", results["accuracy"])
print("Cross-validation Mean Accuracy:", results["cv_mean"])
print("Classification Report:\n", results["classification_report"])
```
This will train a decision tree, print accuracy, cross-validation scores, and a classification report.

### 2. H2O AutoML
```python
from phishing_ml import automl_model

# Train AutoML models
aml_results = automl_model.run_automl("data/phishing.csv")

print("Test Accuracy:", aml_results["test_accuracy"])
print("Test AUC:", aml_results["test_auc"])
print("Best Model ID:", aml_results["best_model"].model_id)
```
This will run H2O AutoML, train multiple models, and return the best model along with test performance metrics.

***A demonstration of how to use the package and its modules can be found in the project folder under phishing_ml_demo/usage.ipynb.***

## Notes
### Notes

- Ensure **H2O** is installed and **Java** is available. You can check with:

```bash
java -version
```

- H2O AutoML may require more memory; adjust the mem_limit parameter if needed.
- This package allows flexible usage for both classical ML and AutoML, enabling experimentation and research on phishing detection.

## Example Usage — API Endpoints (`/train`, `/predict`, `/metrics`)

The API endpoints can be accessed using any HTTP client or method of your choice (e.g., curl, Postman, Python requests, etc.).  
For clarity and convenience, the examples below are provided using **PowerShell** (`Invoke-RestMethod`).

### 1. POST /train — Train Model from CSV

Description: Upload a CSV file via multipart/form-data to the /train endpoint. Form fields: file (CSV), max_depth (integer), and cv_folds (integer).

PowerShell example (assuming .\phishing.csv exists in the current directory):

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/train" `
  -Method Post `
  -Form @{
    file      = Get-Item ".\phishing.csv"
    max_depth = "3"
    cv_folds  = "5"
  }
  ```
Expected response (example JSON):
```json
{
  "message": "Model trained successfully.",
  "accuracy": 0.95,
  "cv_mean": 0.94,
  "features_used": [ "...list..." ]
}
  ```
**Common errors:**

Connection refused → server not started or incorrect host/port.

415 Unsupported Media Type → PowerShell version does not support -Form or the request did not form multipart correctly.

500 server errors → check decision_tree.py and CSV column names.

## 2. POST /predict — Make a Prediction
Description: Send a single observation as JSON. Use the same fields expected by the model (PhishingInput). Ensure the model has been trained with /train first.

PowerShell example:
```powershell
$body = @{
  Shortining_Service   = 0
  HTTPS_token          = 1
  Abnormal_URL         = 0
  SSLfinal_State       = 1
  Favicon              = 0
  port                 = 0
  on_mouseover         = 0
  Submitting_to_email  = 0
}

$response = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body ($body | ConvertTo-Json -Depth 5)

# Display the prediction
$response
  ```
Expected response (example):
```json
{
  "prediction": "Not phishing",
  "raw_value": 0
}
  ```
**Common errors:**

{"error":"Model not trained yet. Please call /train first."} → run /train first.

422 Unprocessable Entity → JSON does not match Pydantic model (missing or misnamed fields).

## 3. GET /metrics — Retrieve Model Metrics
Description: Returns metrics from the most recent training, including accuracy, cross-validation mean, confusion matrix, and classification report.

PowerShell example:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metrics" -Method Get
  ```
Expected response (example JSON):
```json
{
  "accuracy": 0.95,
  "cv_mean": 0.94,
  "confusion_matrix": [[50, 2], [3, 45]],
  "classification_report": {
    "0": { "precision": 0.94, "recall": 0.96, "f1-score": 0.95, "support": 52 },
    "1": { "precision": 0.96, "recall": 0.94, "f1-score": 0.95, "support": 48 }
  }
}

  ```
Error if no model exists:
```json
{"error":"No model has been trained yet."}
  ```
### Best Practices & Tips
- Use a virtual environment to isolate project dependencies:
   python -m venv .venv
   pip install -r requirements.txt
- Validate CSV before /train: column names must match the features expected by decision_tree.py.
- Check logs: uvicorn console output provides detailed information for failed requests.
- Security: This example assumes a local development environment. For production, add authentication, HTTPS, and rate-limiting.
- PowerShell version: PowerShell Core / 7+ is recommended to support Invoke-RestMethod -Form for file uploads.
## Author and License

**Author:** tauimonen
**Date:** 2025-10-30
**License:** MIT License  
You are free to use, modify, and distribute this software under the terms of the MIT License.  see the [LICENSE](LICENSE.txt) file for details.
