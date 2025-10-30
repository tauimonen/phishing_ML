"""
decision_tree.py

Author: tauimonen
Date: 2025-10-30
Description: 
    This module implements a Decision Tree classifier for phishing website detection.
    It includes functions to train the model, evaluate accuracy, generate a confusion matrix,
    produce classification reports, and perform cross-validation.

Dependencies:
    - pandas
    - scikit-learn
    - matplotlib (optional, for plotting confusion matrix)
    - seaborn (optional, for visualization)

Functions:
    - run_decision_tree(csv_path, features=None, target="Result", max_depth=3, cv_folds=5)
        Trains a Decision Tree classifier on the given CSV dataset and returns performance metrics.
"""
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

def run_decision_tree(csv_path, features=None, target="Result", max_depth=3, cv_folds=5):
    """
    Train a Decision Tree classifier on the given dataset.

    Parameters:
        csv_path (str): Path to the CSV dataset
        features (list, optional): List of feature column names. Defaults to a predefined list.
        target (str, optional): Name of the target column. Defaults to "Result".
        max_depth (int, optional): Maximum depth of the decision tree. Default is 3.
        cv_folds (int, optional): Number of folds for cross-validation. Default is 5.

    Returns:
        dict: Contains trained model, confusion matrix, accuracy, classification report,
              cross-validation scores, and mean CV accuracy.
    """
    df = pd.read_csv(csv_path, sep=";")
    
    if features is None:
        # Default feature list
        features = [
            "Shortining_Service", "HTTPS_token", "Abnormal_URL", "SSLfinal_State",
            "Favicon", "port", "on_mouseover", "Submitting_to_email"
        ]
    
    X = df[features]
    y = df[target]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train decision tree
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Not phishing", "Phishing"])
    
    # Cross-validation
    scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv_folds)
    
    return {
        "model": clf,
        "confusion_matrix": cm,
        "accuracy": acc,
        "classification_report": cr,
        "cv_scores": scores,
        "cv_mean": scores.mean()
    }
