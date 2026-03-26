# Code for Explainable Detection of Severe Hypoglycemia in Type-1 Diabetes

This folder contains the main implementation used in the paper:

**Explainable Detection of Severe Hypoglycemia in Type-1 Diabetes Patients with Multimodal Data, Optimized Static and Dynamic Ensemble Frameworks**

## Overview

Severe hypoglycemia (SH) in Type 1 Diabetes (T1D) is a life-threatening complication that remains difficult to predict using conventional clinical approaches. This code implements an **explainable multimodal machine learning framework** for SH risk prediction using heterogeneous patient information, including **clinical/tabular data, medical text, psychological and behavioral assessments, and continuous glucose monitoring (CGM) time-series data**. The framework supports both **early fusion** and **late fusion**, and evaluates **classical machine learning models, static ensemble methods, and Dynamic Ensemble Selection (DES)** methods. 

The study processes data from **187 adults with T1D** and reports that the **best late-fusion DES model achieved ROC-AUC ≈ 0.88**, while early-fusion models reached **ROC-AUC ≈ 0.86**. The framework also incorporates **SHAP-based explainability** to identify clinically meaningful risk factors. 

## Main contributions implemented in this code

- A **multimodal predictive pipeline** integrating structured clinical and demographic variables, psychological and behavioral assessments, medical text, and CGM time-series features. 
- Support for both **feature-level (early) fusion** and **prediction-level (late) fusion**. 
- Comparative evaluation of:
  - **Classical ML models**
  - **Static ensemble models**
  - **Dynamic Ensemble Selection (DES)** methods, including FIRE-enhanced variants. 
- **Bayesian hyperparameter optimization** and **feature selection** for robust model development. 
- **Explainable AI (XAI)** using **SHAP** for global and local interpretation. 

## Data modalities

The full framework is designed around the following modalities:

- **Tabular / structured data**
  - attitude
  - blood test
  - demographic/lifestyle variables
  - depression-related variables
  - fear-related variables
  - MoCA
  - total scores
  - hypoglycemia unawareness
  - medication chart
- **Text**
  - medical conditions
  - medications
- **Time series**
  - CGM recordings 

## Methodology implemented in this folder

### 1. Modality-specific encoding

The framework uses specialized encoders for each modality:

- **TabularEncoder**
  - categorical encoding
  - missing-value imputation
  - normalization / standardization
  - statistical feature selection
- **TextEncoder**
  - based on **Clinical BioBERT**
  - patient-level text embedding extraction
- **TS2VecEncoder**
  - time-series representation learning for CGM
  - extraction of glucose-related temporal representations and glycemic features. 

### 2. Feature selection

A **NeuralFeatureSelector** is used for dimensionality reduction with attention-based weighting and regularization. In the paper setting, target dimensionalities were configured to reduce tabular, text, and time-series representations before fusion. 

### 3. Fusion strategies

The code supports two main multimodal learning settings:

- **Early Fusion**
  - modality embeddings are concatenated into a single representation before classification
- **Late Fusion**
  - each modality is modeled independently and predictions are aggregated using a meta-classifier. :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}

### 4. Prediction layer

The framework evaluates three families of predictors:

#### Classical machine learning
- Decision Tree
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors
- Multi-Layer Perceptron
- Support Vector Classifier. 

#### Static ensemble models
- Random Forest
- XGBoost
- Gradient Boosting
- AdaBoost
- CatBoost
- optional LightGBM extension.

#### Dynamic Ensemble Selection (DES)
Representative DES methods include:
- KNORA-E
- KNORA-U
- KNOP
- DES-MI
- META-DES
- DES-KNN
- DES-P  
along with **FIRE-based variants**. 

### 5. Evaluation and explainability

The codebase supports:
- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- ROC curve generation
- Friedman test
- Nemenyi post-hoc analysis
- SHAP-based feature importance and interpretation.

## Environment

The experimental environment reported in the paper used:

- **Python 3.11**
- Pandas 1.5.3
- NumPy 1.20.3
- Optuna 4.3.0
- Scikit-learn 1.2.2
- Matplotlib 3.7.1
- Deslib 0.3.7. 

Depending on which parts of the pipeline you run, you may also need packages commonly used by this implementation, such as:
- PyTorch
- Transformers
- SHAP
- CatBoost
- XGBoost
- imbalanced-learn
- scikit-optimize

