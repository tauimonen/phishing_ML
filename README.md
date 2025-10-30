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

**Features include:**

- `Shortining_Service` – Whether URL shortening service is used  
- `HTTPS_token` – Suspicious HTTPS token in URL  
- `Abnormal_URL` – Abnormal URL pattern  
- `SSLfinal_State` – SSL certificate information  
- `Favicon` – Favicon status  
- `port` – Non-standard port usage  
- `on_mouseover` – JavaScript mouseover events  
- `Submitting_to_email` – Whether form submits to email

### Note on Features

The phishing dataset contains over 50 features extracted from websites.  
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
git clone https://github.com/yourusername/phishing_ml.git
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

## Notes
### Notes

- Ensure **H2O** is installed and **Java** is available. You can check with:

```bash
java -version
```

- H2O AutoML may require more memory; adjust the mem_limit parameter if needed.
- This package allows flexible usage for both classical ML and AutoML, enabling experimentation and research on phishing detection.

## Author and License

**Author:** tauimonen
**Date:** 2025-10-30
**License:** MIT License  
You are free to use, modify, and distribute this software under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.
