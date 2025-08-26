# Explainable Detection of Severe Hypoglycemia in Type-1 Diabetes Patients with Multimodal Data, Optimized Static and Dynamic Ensemble Frameworks

<p align="justify"> Severe hypoglycemia remains a critical and life-threatening complication for patients with Type 1
Diabetes, necessitating accurate and early risk prediction. In this study, we propose a comprehensive
machine learning framework that leverages multimodal patient data—including clinical, psychological, and cognitive features—to predict severe hypoglycemic episodes. We evaluate and compare
classical machine learning models, static ensemble classifiers, and dynamic ensemble selection (DES)
techniques, all optimized through feature selection and hyperparameter tuning. Both early fusion and
late fusion strategies are investigated. Our results indicate that DES and static ensembles achieve
comparable performance, improving upon the baseline reported in Francisco et al. (2025) with the
same data source. The best model in the late fusion setting reaches an AUC-ROC of 𝟎.𝟖𝟕𝟕𝟐, while
the best early fusion model achieves an Accuracy of 𝟎.𝟕𝟗𝟖𝟑. Statistical tests confirm the robustness
and consistency of these models across multiple metrics. Ultimately, this work highlights the clinical
relevance of advanced ensemble methods for improving predictive accuracy in high-risk diabetes care,
and underscores their potential for integration into decision support systems aimed at proactive inter-
vention and personalized treatment. Dataset : https://figshare.com/articles/dataset/Multimodal-fusion-severe-hypo-data/25136942/1 </p>

---
## Dataset used 
Data raw and preprocessed from Jaeb Center for Health Research (Tampa, Florida, United States): https://figshare.com/articles/dataset/Multimodal-fusion-severe-hypo-data/25136942/2  
The preprocessed data has been concatenated into a single CSV file, fusing their text, time series, and tabular data.  

Filename in the project: complete_fused_dataset.csv

---
## Roadmap and Frameworks
### Complete and summarised road map : 
<img width="570" height="645" alt="ROADMAP" src="https://github.com/user-attachments/assets/144f5a04-b745-4157-bea4-9b3dc8bc606a" />

### Early Fusion (detailed): 
<img width="239" height="652" alt="frameworkearly" src="https://github.com/user-attachments/assets/7b3ef6e3-9ecd-4e71-9a6d-b4e178a081f5" />

### Late Fusion (detailed): 
<img width="239" height="652" alt="frameworklate (1)" src="https://github.com/user-attachments/assets/993f8f74-f9cc-4717-a26d-0d878559d367" />

### Data preprocessing (detailed) : 
<img width="444" height="456" alt="datapreprocessing (2)" src="https://github.com/user-attachments/assets/03a2dd96-786b-4c13-aa44-a93fcf7b2ee9" />

### Early Fusion / Late Fusion representation :
<img width="482" height="449" alt="late early (1)" src="https://github.com/user-attachments/assets/a7063214-3498-4268-bee9-9899963b6df4" />
