from setuptools import setup, find_packages

setup(
    name="phishing_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "h2o"
    ],
    author="Your Name",
    description="Phishing detection models: Decision Tree & H2O AutoML",
)
