# AUTISM_python
AUTISM_python
Autism Spectrum Disorder — Early Detection Model
AI-driven screening support tool for early ASD identification
![Python](https://img.shields.io/badge/Python-3.8+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)


Overview
This project builds a machine learning model to support early detection of Autism Spectrum Disorder (ASD) using behavioral screening data. The goal is to simulate a clinical decision-support tool that can assist in identifying individuals at risk and prioritizing further diagnostic evaluation.
> This is not just a classification problem — it is a healthcare problem.  
> The real objective is not prediction accuracy, but **reducing missed diagnoses**.


Why This Matters
Early ASD detection is critical for improving long-term outcomes. However:
Many cases remain undiagnosed or significantly delayed
Screening tools must balance sensitivity and specificity
Missing positive cases (false negatives) carries real clinical consequences
This project prioritizes recall over raw accuracy — reflecting how clinical screening tools should be evaluated.


Dataset
Property	Value
Total samples	800
Features	21 (behavioral scores, demographics, prior screening results)
Target	ASD classification (0 = No ASD, 1 = ASD)
Class distribution	Non-ASD ~80% / ASD ~20%
> The dataset is imbalanced by design — reflecting real-world ASD prevalence rates.


ML Pipeline

Raw Data
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Model Training (Decision Tree → Random Forest → XGBoost)
   ↓
Evaluation (Recall-focused metrics)
   ↓
Threshold Optimization
```
---
Models
Model	Status
Decision Tree	Baseline
Random Forest	Improved baseline
XGBoost	Planned — minority class optimization
---
Current Results
Metrics (Random Forest — baseline)
Metric	Score
Accuracy	0.80
Precision (ASD class)	0.56
Recall (ASD class)	0.53
F1-score (ASD class)	0.54
Confusion Matrix
```
                Predicted: No ASD    Predicted: ASD
Actual: No ASD       109                  15
Actual: ASD           17                  19
```
Key Insight
While overall accuracy is 80%, the model currently misses 47% of true ASD cases (17 false negatives out of 36).
> In a clinical screening context, false negatives are the critical failure mode.  
> A missed diagnosis means delayed intervention — the highest-cost outcome.
This highlights a core principle in healthcare ML:  
Sensitivity (recall) must be prioritized over accuracy when detecting disease.
---
Challenges
Class imbalance — 80/20 non-ASD/ASD split causes model bias toward majority class
Majority class dominance — high accuracy masks poor minority class performance
Precision–recall trade-off — reducing false negatives increases false positives; the clinical context determines acceptable balance
---
Planned Improvements
[ ] Apply SMOTE to synthetically oversample the ASD minority class
[ ] Lower decision threshold (0.3–0.35) to reduce false negatives
[ ] Implement XGBoost with `scale_pos_weight` for imbalanced class handling
[ ] Cross-validation with stratified folds to preserve class distribution
[ ] Add SHAP feature importance for clinical interpretability and trust
[ ] Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
---
Future Direction
Integrate multimodal data (behavioral signals, structured clinical notes)
Deploy as a real-time Streamlit screening dashboard for clinician use
Add model interpretability layer (SHAP / LIME) for clinical transparency
Evaluate fairness metrics across demographic subgroups
---
What This Project Demonstrates
End-to-end ML pipeline: data → model → clinical evaluation
Handling imbalanced healthcare datasets
Understanding clinical impact beyond standard metrics
Ability to translate ML outputs into real-world health applications
---
Tech Stack
`Python` `scikit-learn` `pandas` `NumPy` `Matplotlib` `Seaborn` `imbalanced-learn (SMOTE)`
---
Author
Prince Kwarteng Amaning  
MS Data Science (in progress) — University of Michigan–Dearborn  
Background: Dentistry | Global Public Health | Healthcare Analytics
Interested in building AI systems that improve early detection, reduce diagnostic gaps, and support clinical decision-making — particularly in underserved communities.
---
License
MIT License — open for educational and research use.
