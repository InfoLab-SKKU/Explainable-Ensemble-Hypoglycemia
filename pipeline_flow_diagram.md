# Multi-Modal Medical Data Processing Pipeline Flow

## EXECUTION ORDER & METHODOLOGY DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN PIPELINE EXECUTION                           │
└─────────────────────────────────────────────────────────────────────────────┘

1. INITIALIZATION & SETUP
   ├── Create output directory: "encoded_features/"
   ├── Initialize encoders:
   │   ├── TabularEncoder(n_features=112, output_dim=112)
   │   ├── TextEncoder(model="Bio_ClinicalBERT", output_dim=768)
   │   └── TimeSeriesTransformerEncoder(output_dim=64)
   └── Load data by type

┌─────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: DATA LOADING                          │
└─────────────────────────────────────────────────────────────────────────────┘

2. DATA LOADING (load_data_by_type function)
   ├── Load Tabular Data
   │   ├── Read all CSV files from "preprocessed/tabular/"
   │   ├── Concatenate files by PtID
   │   └── Result: tabular_data DataFrame
   │
   ├── Load Text Data  
   │   ├── Read all CSV files from "preprocessed/text/"
   │   ├── Concatenate files by PtID
   │   └── Result: text_data DataFrame
   │
   ├── Load Time Series Data
   │   ├── Read all CSV files from "preprocessed/time_series/"
   │   ├── Aggregate by patient (mean, std, count per patient)
   │   └── Result: ts_data DataFrame
   │
   └── Load Patient Roster
       ├── Read "BPtRoster.txt" 
       └── Result: roster_data (PtID + BCaseControlStatus labels)

┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2: INITIAL ENCODING                         │
└─────────────────────────────────────────────────────────────────────────────┘

3. TABULAR DATA ENCODING (if tabular_data exists)
   ├── Preprocessing:
   │   ├── Categorical encoding (LabelEncoder)
   │   ├── Remove empty columns
   │   ├── Missing value imputation (median strategy)
   │   └── StandardScaler normalization
   │
   ├── Feature Selection:
   │   ├── Join with roster for labels (if available)
   │   ├── SelectKBest with f_classif
   │   └── Keep top 112 features
   │
   ├── Output Processing:
   │   ├── Pad/truncate to output_dim=112
   │   └── Save: "encoded_features/tabular_encoded.csv"
   │
   └── Result: tabular_embeddings (patients × 112 features)

4. TEXT DATA ENCODING (if text_data exists)
   ├── Model Loading:
   │   ├── Load Clinical BioBERT tokenizer
   │   ├── Load Clinical BioBERT model
   │   └── Move to GPU/CPU device
   │
   ├── Text Processing:
   │   ├── Clean text (remove newlines, normalize spaces)
   │   ├── Combine all text columns per patient
   │   ├── Tokenize with max_length=512
   │   └── Extract CLS token embeddings
   │
   ├── Output Processing:
   │   ├── Truncate/pad to output_dim=768
   │   └── Save: "encoded_features/text_encoded.csv"
   │
   └── Result: text_embeddings (patients × 768 features)

5. TIME SERIES ENCODING (if ts_data exists)
   ├── Data Processing:
   │   ├── Convert to sequences per patient
   │   ├── Create (Hypo, Number) pairs
   │   ├── Pad sequences to max_seq_length=1000
   │   └── StandardScaler normalization
   │
   ├── Transformer Encoding:
   │   ├── Build TimeSeriesTransformer model
   │   ├── Process with multi-head attention
   │   ├── Global average pooling
   │   └── Extract fixed-size representation
   │
   ├── Output Processing:
   │   ├── Resize to output_dim=64
   │   └── Save: "encoded_features/timeseries_encoded.csv"
   │
   └── Result: ts_embeddings (patients × 64 features)

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: NEURAL FEATURE SELECTION                        │
└─────────────────────────────────────────────────────────────────────────────┘

6. USER FUSION METHOD SELECTION
   ├── Display fusion options:
   │   ├── 1. Early Fusion (concatenation)
   │   ├── 2. Cross-Attention Fusion
   │   └── 3. CLS Transformer Fusion
   └── User chooses method

7. NEURAL FEATURE SELECTION PER MODALITY
   ├── Tabular Neural Selection:
   │   ├── Target: 112 → 50 features
   │   ├── NeuralFeatureSelector with supervised learning
   │   ├── Architecture: [256, 128, 64] hidden layers
   │   ├── Training: 80 epochs, CrossEntropyLoss
   │   ├── Attention-based feature importance
   │   └── Save: "encoded_features/tabular_reduced.csv"
   │
   ├── Text Neural Selection:
   │   ├── Target: 768 → 100 features  
   │   ├── NeuralFeatureSelector with supervised learning
   │   ├── Architecture: [512, 256, 128] hidden layers
   │   ├── Training: 60 epochs, CrossEntropyLoss
   │   ├── Attention-based feature importance
   │   └── Save: "encoded_features/text_reduced.csv"
   │
   └── Time Series Neural Selection:
       ├── Target: 64 → 32 features
       ├── NeuralFeatureSelector with supervised learning
       ├── Architecture: [128, 64, 32] hidden layers
       ├── Training: 100 epochs, CrossEntropyLoss
       ├── Attention-based feature importance
       └── Save: "encoded_features/timeseries_reduced.csv"

┌─────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 4: FUSION                                  │
└─────────────────────────────────────────────────────────────────────────────┘

8. FUSION EXECUTION (based on user choice)

   A. EARLY FUSION (if selected)
      ├── Simple concatenation of reduced features
      ├── Merge all modalities on PtID
      ├── Result: (tabular_50 + text_100 + ts_32) = 182 features
      └── Save: "encoded_features/fused_multimodal_dataset.csv"

   B. CROSS-ATTENTION FUSION (if selected)
      ├── Initialize MultiModalCrossAttention
      │   ├── d_model=256, n_heads=8
      │   ├── Project each modality to d_model
      │   └── Cross-attention between modalities
      │
      ├── Processing:
      │   ├── Each modality attends to others
      │   ├── Feed-forward networks per modality
      │   ├── Layer normalization
      │   └── Final fusion layer
      │
      ├── Post-processing:
      │   ├── MinMaxScaler normalization
      │   ├── Zero-value mitigation
      │   └── Result: 256 fused features
      │
      └── Save: "encoded_features/fused_multimodal_dataset.csv"

   C. CLS TRANSFORMER FUSION (if selected)
      ├── Initialize CLSTransformerFusion
      │   ├── Learnable CLS token
      │   ├── Positional embeddings per modality
      │   ├── Transformer encoder layers
      │   └── Classification head
      │
      ├── Processing:
      │   ├── Add CLS token to sequence
      │   ├── Project modalities to d_model=256
      │   ├── Add positional embeddings
      │   ├── Transformer encoding
      │   └── Extract CLS representation
      │
      ├── Result: 256 fused features
      └── Save: "encoded_features/fused_multimodal_dataset.csv"

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 5: FINAL OUTPUT                              │
└─────────────────────────────────────────────────────────────────────────────┘

9. FINAL PROCESSING & SAVING
   ├── Merge with roster labels (BCaseControlStatus)
   ├── Create complete dataset (patients with all modalities)
   ├── Generate statistics and summaries
   ├── Save final files:
   │   ├── "encoded_features/fused_multimodal_dataset.csv"
   │   ├── "encoded_features/complete_fused_dataset.csv"
   │   ├── "encoded_features/tabular_reduced.csv"
   │   ├── "encoded_features/text_reduced.csv"
   │   └── "encoded_features/timeseries_reduced.csv"
   └── Display final statistics

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARCHITECTURAL FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

Raw Data (3 modalities)
         ↓
   Initial Encoding
    ┌─────────────────┐
    │ Tabular: 112D   │
    │ Text: 768D      │ 
    │ Time Series: 64D│
    └─────────────────┘
         ↓
  Neural Feature Selection
    ┌─────────────────┐
    │ Tabular: 50D    │
    │ Text: 100D      │
    │ Time Series: 32D│
    └─────────────────┘
         ↓
      Fusion Layer
    ┌─────────────────┐
    │ Early: 182D     │
    │ Cross-Att: 256D │
    │ CLS Trans: 256D │
    └─────────────────┘
         ↓
    Final Dataset
  (Ready for ML tasks)

EXECUTION ENTRY POINT: 
- Script starts with `if __name__ == "__main__": main()`
- Creates output directory
- Executes the complete pipeline sequentially
```

## Key Execution Characteristics:

1. **Sequential Processing**: Each phase must complete before the next begins
2. **Conditional Execution**: Each modality is processed only if data exists  
3. **User Interaction**: Fusion method selection happens mid-pipeline
4. **Error Handling**: Graceful handling of missing modalities
5. **Incremental Saving**: Results saved at each major step
6. **Memory Management**: Models moved to the appropriate device (GPU/CPU)

## Critical Dependencies:
- PyTorch + CUDA (for neural networks)
- HuggingFace Transformers (for BioBERT)
- scikit-learn (for classical ML preprocessing)
- pandas/numpy (for data manipulation)
