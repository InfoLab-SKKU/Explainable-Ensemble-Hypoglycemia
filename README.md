# Explainable Detection of Severe Hypoglycemia in Type-1 Diabetes Patients with Optimized Static and Dynamic Ensemble Frameworks

<p align="justify"> Type 1 diabetes is a chronic autoimmune disease that affects individuals of all ages and requires lifelong insulin therapy. One of its most serious complications is severe hypoglycemia (SH), a condition where blood glucose levels fall dangerously low, often requiring external assistance. If not properly detected and treated, SH can lead to seizures, unconsciousness, or even death. This study introduces an explainable ensemble-based framework designed to predict the risk of SH among individuals with Type 1 diabetes. We evaluate and compare both static ensemble strategies (SES) and dynamic ensemble selection (DES) methods, focusing on their effectiveness in identifying patients at high risk. Using a multimodal dataset originally introduced in previous work, our approach includes model performance evaluation supported by detailed visual analyses. Furthermore, explainable AI (XAI) techniques—specifically SHAP values—are integrated to provide interpretability and insights into the most influential features driving the model's predictions. Our results demonstrate that both methods, particularly CatBoost/GradientBoosting (SES) and FIRE-KNORA-U/KNORAU, can achieve strong predictive performance, and the use of XAI enables better understanding of clinical and behavioral factors associated with SH. These findings suggest that ensemble learning, combined with explainability, can offer valuable decision-support tools in managing diabetes-related risks, showing satisfying precision, accuracy, and AUCROC scores. </p>

---
## Dataset used 
Data raw and preprocessed from Jaeb Center for Health Research (Tampa, Florida, United States): https://figshare.com/articles/dataset/Multimodal-fusion-severe-hypo-data/25136942/2  
The preprocessed data has been concatenated into a single CSV file, fusing their text, time series, and tabular data.  

Filename in the project: complete_fused_dataset.csv

---
## Roadmap and Frameworks
Complete and summarised road map : 
<img width="570" height="645" alt="ROADMAP" src="https://github.com/user-attachments/assets/144f5a04-b745-4157-bea4-9b3dc8bc606a" />
Early Fusion (detailed): 
<img width="239" height="652" alt="frameworkearly" src="https://github.com/user-attachments/assets/7b3ef6e3-9ecd-4e71-9a6d-b4e178a081f5" />
Late Fusion (detailed): 
<img width="239" height="652" alt="frameworklate (1)" src="https://github.com/user-attachments/assets/993f8f74-f9cc-4717-a26d-0d878559d367" />
Data preprocessing (detailed) : 
<img width="444" height="456" alt="datapreprocessing (2)" src="https://github.com/user-attachments/assets/03a2dd96-786b-4c13-aa44-a93fcf7b2ee9" />
Early Fusion / Late Fusion representation 
<img width="482" height="449" alt="late early (1)" src="https://github.com/user-attachments/assets/a7063214-3498-4268-bee9-9899963b6df4" />
