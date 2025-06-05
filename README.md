# Explainable Detection of Severe Hypoglycemia in Type-1 Diabetes Patients with Optimized Static and Dynamic Ensemble Frameworks

<p align="justify"> Type 1 diabetes is a chronic autoimmune disease that affects individuals of all ages and requires lifelong insulin therapy. One of its most serious complications is severe hypoglycemia (SH), a condition where blood glucose levels fall dangerously low, often requiring external assistance. If not properly detected and treated, SH can lead to seizures, unconsciousness, or even death. This study introduces an explainable ensemble-based framework designed to predict the risk of SH among individuals with Type 1 diabetes. We evaluate and compare both static ensemble strategies (SES) and dynamic ensemble selection (DES) methods, focusing on their effectiveness in identifying patients at high risk. Using a multimodal dataset originally introduced in previous work, our approach includes model performance evaluation supported by detailed visual analyses. Furthermore, explainable AI (XAI) techniques—specifically SHAP values—are integrated to provide interpretability and insights into the most influential features driving the model's predictions. Our results demonstrate that both methods, particularly CatBoost/GradientBoosting (SES) and FIRE-KNORA-U/KNORAU, can achieve strong predictive performance, and the use of XAI enables better understanding of clinical and behavioral factors associated with SH. These findings suggest that ensemble learning, combined with explainability, can offer valuable decision-support tools in managing diabetes-related risks, showing satisfying precision, accuracy, and AUCROC scores. </p>

---
## Dataset used 
Data raw and preprocessed from Jaeb Center for Health Research (Tampa, Florida, United States): https://figshare.com/articles/dataset/Multimodal-fusion-severe-hypo-data/25136942/2
The preprocessed data has been concatenated into a single CSV file, fusing their text, time series, and tabular data. 
Filename in the project: 3labelv4Classification.csv


## The Architecture of the Proposed Framework
<div align="center">
  <img src="https://github.com/InfoLab-SKKU/DES4Depression/blob/main/Proposed%20Architecture_page-0001.jpg" alt="Proposed Architecture" width="500"/>
</div>

---

## Guideline for Reproducing Results

### Obtaining and Processing Data

1. The data can be found on [NSHAP - ICPSR20541](https://doi.org/10.3886/ICPSR20541.v10) under the **Data & Documentation** tab. Download **DS1 Core Data, Public-Use**.

2. Run the following lines in R to create a CSV file:

    ```r
    # Load the .rda file and export it to CSV
    my_data_frame = load("./20541-0001-Data.rda")
    ls()
    write.csv(my_data_frame, file = "data.csv", row.names = FALSE)
    ```

3. Data can be processed in `dataPreprocessing.ipynb` in the code folder. Run the code sequentially. At the end, two outputs will be generated:
   - **3labelv4Classification.csv** for Detection and Severity Prediction tasks.
   - **3labelv4Regression.csv** for Scale Prediction task.

### Detection and Severity Prediction Task

- **Metrics Results**:

  The `detectionLayer.ipynb` and `severityPredictionLayer.ipynb` files in the code folder follow similar experimental methods. The code needs to be run sequentially from **Start** up to **Training (classic/static)**. At this point, you may choose to:

  1. **Train with classical/static classifiers**.

     - To train with classic models, uncomment the models you want to use in the **Classic Classifiers** code cell and run it.
     - To train with static models, uncomment the models you want to use in the **Static Classifiers** code cell and run it.
     - To get the average over 10 states, ROC curves, and FN curve, run the **Post Training (classic/static)** code cells only after running either the **Classic Classifiers** or **Static Classifiers** code cell.
     - Below the **Post Training (classic/static)** code cells, you'll find hyperparameter optimization cells.

  2. **Train with DES methods**.

     - Go immediately to **DES Training (all)**, uncomment the base classifiers that you want to use for DES, and run the code cells.

- **SHAP Results**:

  - SHAP results can be generated in the **Shap (will mostly be exported files)** section, which is located towards the bottom of the `.ipynb` file.

### Scale Prediction Task

- **Metrics Results**:

  - The `regressionLayer.ipynb` file in the code folder can be used for the Scale Prediction task.
  - Run the code sequentially from **Start** up to before **Regression Training**.
  - Then, run **Regression Training** to get the metrics.

  - Hyperparameter optimization can be found towards the bottom of the `.ipynb` file.

- **SHAP and FN**:

  - Run the code in the **SHAP and FN** sections to generate the FN diagram and SHAP files.

---

## Citation

We would appreciate it if you consider citing our work when using our code.

```bibtex
@article{imans2024explainable,
  title={Explainable Multi-Layer Dynamic Ensemble Framework Optimized for Depression Detection and Severity Assessment},
  author={Imans, Dillan and Abuhmed, Tamer and Alharbi, Meshal and El-Sappagh, Shaker},
  journal={Diagnostics},
  volume={14},
  number={21},
  pages={2385},
  year={2024}
}
```
