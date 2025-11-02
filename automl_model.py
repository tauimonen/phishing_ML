"""
automl_model.py

Author: tauimonen
Date: 2025-10-30
Description:
    This module implements an H2O AutoML pipeline for phishing website detection.
    It trains multiple models automatically, including GBM, Stacked Ensembles, and more.
    Returns the best model along with test set performance metrics.

Dependencies:
    - h2o

Functions:
    - run_automl(csv_path, target="Result", max_models=10, max_runtime_secs=90, mem_limit="4G")
        Trains H2O AutoML on the given dataset and returns the leaderboard, best model,
        test accuracy, and test AUC.
"""
import h2o
from h2o.automl import H2OAutoML

def run_automl(csv_path, target="Result", max_models=10, max_runtime_secs=90, mem_limit="4G"):
    """
    Train H2O AutoML models on the dataset.

    Parameters:
        csv_path (str): Path to the CSV dataset
        target (str, optional): Name of the target column. Defaults to "Result".
        max_models (int, optional): Maximum number of models to train. Default is 10.
        max_runtime_secs (int, optional): Maximum runtime in seconds. Default is 90.
        mem_limit (str, optional): Max memory allocation for H2O, e.g., "4G".

    Returns:
        dict: Contains the AutoML object, best model, test accuracy, and test AUC.
    """
    h2o.init(max_mem_size=mem_limit)
    
    data = h2o.import_file(csv_path, sep=";")
    data[target] = data[target].asfactor()  # Ensure binary classification
    
    x = list(data.columns)
    x.remove(target)
    
    train, test = data.split_frame(ratios=[0.8], seed=42)
    
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=42,
        balance_classes=True
    )
    aml.train(x=x, y=target, training_frame=train)
    
    best_model = aml.leader
    perf = best_model.model_performance(test)
    
    # Save the best model
    model_path = h2o.save_model(best_model, path="./models", force=True)
    print(f"Model saved: {model_path}")
    
    return {
        "aml": aml,
        "best_model": best_model,
        "test_accuracy": perf.accuracy()[0][1],
        "test_auc": perf.auc()
    }
