# KMITL AI4BA · Data Science Projects

> 🎓 Coursework · M.Sc. Artificial Intelligence for Business Analytics (AI4BA), King Mongkut's Institute of Technology Ladkrabang (KMITL)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)

Two end-to-end classification projects: **data cleansing → EDA → preprocessing → model comparison → evaluation**.

| # | Project | Business question | Best model |
|---|---|---|---|
| 1 | [👥 HR Attrition Prediction](#1--hr-attrition-prediction) | Which employees are likely to leave? | Random Forest / KNN (tuned) |
| 2 | [🚗 Insurance Cross-Sell Prediction](#2--insurance-cross-sell-prediction) | Which health-insurance customers will buy car insurance? | XGBoost (AUC 0.83) |

---

## 1 · HR Attrition Prediction

📓 [`hr-attrition/HR_Attrition.ipynb`](hr-attrition/HR_Attrition.ipynb) · 📄 [`IBM HR Data3.csv`](hr-attrition/IBM%20HR%20Data3.csv)

**Data:** IBM HR Analytics dataset, extended version: 23,532 rows × 37 columns

**Workflow**

1. **Cleansing:** dropped missing values and duplicates, trimmed whitespace, fixed data types, removed ID and constant columns, cleaned invalid `"Test"` values
2. **Labeling:** `Termination` and `Voluntary Resignation` → **Inactive**; `Current employee` → **Active**
3. **EDA:** distributions of age, income, tenure and promotion; attrition by department, education field, gender, job level and marital status; correlation heatmap
4. **Feature selection:** one-hot encoding, then an Extra Trees feature-importance ranking. Kept the **16 of 32** features with more than 2.5% importance
5. **Modeling:** 6 algorithms, each tuned with Grid Search and Randomized Search

| Model | Accuracy (test) |
|---|---|
| Logistic Regression | 83.4% |
| Support Vector Classifier | 88.1% |
| Neural Network (MLP) | 96.1% → 99.5% (tuned) |
| KNN | 98.1% → 99.5% (tuned) |
| Random Forest | 99.8% |

> **💡 Lessons learned:** Accuracy above 99% is a red flag. The extended IBM dataset contains many near-duplicate employee records, so a random train/test split lets near-identical rows land on both sides (**data leakage**). Next steps: de-duplicate before splitting, use a group-aware split, and report **F1 / ROC-AUC** for the minority (attrition) class rather than accuracy alone.

---

## 2 · Insurance Cross-Sell Prediction

📓 [`insurance-cross-sell/insurance.ipynb`](insurance-cross-sell/insurance.ipynb)

**Business case:** An insurer wants to predict which of last year's health-insurance policyholders will be interested in its new **vehicle insurance** product, so that sales outreach can be targeted.

**Data:** 113,252 customers × 11 features (age, gender, driving license, region, previously insured, vehicle age, vehicle damage, annual premium, sales channel, vintage)
**Target:** `Response` is imbalanced: **12.3% Yes** (13,945) vs 87.7% No

**Key EDA insights**

- Customers with **past vehicle damage** make up almost all positive responses (13,666 of 13,945)
- **Vehicle age 1–2 years** is the largest interested segment
- Men respond slightly more than women; certain age bands matter strongly

**Workflow**

1. **Cleansing:** dropped missing values, checked for duplicates, removed the ID column
2. **Preprocessing:** label-encoded Gender, Vehicle Age and Vehicle Damage; StandardScaler; 80/20 train/test split
3. **Visualization:** PCA 2-D projection (Plotly)
4. **Model comparison:** Random Forest, Logistic Regression (2 solvers), Gradient Boosting, Gaussian Naive Bayes and XGBoost, compared with **Stratified 5-Fold CV on ROC-AUC**
5. **Evaluation:** learning curves (log-loss, error), ROC curve, classification report (Yellowbrick)

**Result:** **XGBoost** gave the best balance. Random Forest overfit.
Test **ROC-AUC = 0.834** (from predicted probabilities)

---

## 📁 Repository Structure

```
kmitl-ai4ba-data-science/
├── hr-attrition/
│   ├── HR_Attrition.ipynb
│   └── IBM HR Data3.csv
├── insurance-cross-sell/
│   ├── insurance.ipynb
│   └── insurance_data.csv
└── README.md
```

## 📚 Data Sources

- IBM HR Analytics Employee Attrition & Performance ([Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)): fictional dataset created by IBM data scientists
- Health Insurance Cross Sell Prediction ([Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction))

---

👤 **Surasak Chantarach** · [LinkedIn](https://linkedin.com/in/surasak-ch/) · [Medium](https://medium.com/@surasak.chantarach) · [GitHub](https://github.com/llexpertll)
