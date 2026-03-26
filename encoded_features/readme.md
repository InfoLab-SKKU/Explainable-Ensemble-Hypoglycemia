# encoded_features

This folder contains the **precomputed encoded feature representations** used in the paper:

**“Explainable Detection of Severe Hypoglycemia in Type-1 Diabetes Patients with Multimodal Data, Optimized Static and Dynamic Ensemble Frameworks.”**

These files are the outputs of the modality-specific encoders and feature-reduction steps applied before training the classification, static ensemble, and dynamic ensemble selection (DES) models.

## Purpose

The goal of this directory is to provide reusable feature representations for the multimodal severe hypoglycemia prediction pipeline. Storing encoded features separately helps:

- avoid recomputing expensive embeddings,
- keep experiments reproducible,
- simplify early-fusion and late-fusion training,
- support ablation and comparison studies across modalities.

## Modalities represented in this folder

The encoded features are derived from the three main modalities used in the paper:

1. **Tabular / structured data**
   - demographic variables
   - blood test values
   - psychological and behavioral assessments
   - cognitive scores
   - medication chart variables

2. **Text encoded data**
   - medical conditions
   - medication-related free text

3. **Time-series data**
   - continuous glucose monitoring (CGM) sequences
   - CGM-derived glycemic metrics

## Encoding pipeline

### 1) Tabular features
Structured clinical and assessment data are processed through a tabular encoder that includes:
- categorical encoding,
- missing-value imputation,
- feature scaling / normalization,
- statistical feature selection.

### 2) Text features
Clinical text is encoded using a biomedical language model to obtain contextual patient-level embeddings.

### 3) Time-series features
CGM sequences are encoded into compact representations using a time-series representation learning pipeline, together with clinically relevant glucose variability metrics.

### 4) Feature reduction
After modality-specific encoding, a neural feature-selection stage reduces each representation to a compact embedding suitable for downstream training.

## Feature dimensions

The paper uses the following reduced feature sizes:

- **Tabular embedding:** 50 dimensions
- **Text embedding:** 100 dimensions
- **Time-series embedding:** 32 dimensions
- **Early-fusion representation:** 182 dimensions total

## How these features are used

### Early fusion
The reduced embeddings from all modalities are concatenated into a single feature vector and used to train:
- classical machine learning models,
- static ensemble models,
- DES models.

### Late fusion
Each modality is modeled independently first. Their modality-specific predictions are then combined at the decision level using a meta-classifier.



