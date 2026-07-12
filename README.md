# 📊 Customer Churn Prediction Dashboard

> Predict customer churn using XGBoost and Explainable AI with an interactive Streamlit dashboard

[Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
[Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit)
[XGBoost](https://img.shields.io/badge/XGBoost-2.0-0066CC?style=flat-square)
[License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

### 🚀 Live Demo
**[View Deployed App](https://vyshnavi-customer-churn-prediction.streamlit.app)**

---

## 📌 Problem Statement
Telecom companies lose **15-25%** of customers annually to churn. Identifying high-risk customers early enables proactive retention strategies, saving **₹24,000/year** per customer.

Customer acquisition costs 5x more than retention. This tool helps businesses move from reactive to proactive churn management.

## 🎯 Solution
This dashboard uses **XGBoost + SMOTE + SHAP** to:

1. **Predict churn probability** for each customer with 82% ROC-AUC
2. **Segment customers** into Low/Medium/High risk buckets automatically
3. **Explain *why*** a customer might churn using SHAP values - full transparency
4. **Export high-risk lists** for retention teams to take immediate action

---

## 🛠 Tech Stack

| Category | Technology | Why This Choice |
| --- | --- | --- |
| **ML Model** | XGBoost Classifier | Best performance on tabular data, handles non-linear patterns |
| **Imbalance Handling** | SMOTE Oversampling | Churn rate ~27%, SMOTE balances training data |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Industry standard for model interpretability |
| **Frontend** | Streamlit | Rapid deployment, Python-native, interactive |
| **Visualization** | Plotly, Matplotlib | Interactive charts + static SHAP plots |
| **Data Processing** | Pandas, Scikit-learn | Standard data manipulation & preprocessing |

---

## 📈 Model Performance

| Metric | Score | Business Interpretation |
| --- | --- | --- |
| **ROC-AUC** | 0.82 | Strong ability to distinguish churners from non-churners |
| **Accuracy** | 79.2% | 8 out of 10 predictions are correct overall |
| **Precision** | 0.65 | 65% of customers flagged as "will churn" actually churn |
| **Recall** | 0.58 | Model captures 58% of all actual churners |
| **F1-Score** | 0.61 | Balanced measure of precision & recall |

> **Note on Imbalanced Data:** Churn datasets have inherent class imbalance (27% churn vs 73% no-churn). These metrics reflect real-world performance. The precision/recall trade-off can be tuned via threshold adjustment based on business cost of false positives vs false negatives.

**Training Details:**
- Dataset: 7,043 Telco customers
- Train/Test Split: 80/20 stratified
- Cross-Validation: 5-Fold CV
- Hyperparameter Tuning: GridSearchCV

---

## 🎨 Key Features

- ✅ **Bulk Prediction:** Upload CSV with 1000s of customers and get instant churn scores
- ✅ **Risk Segmentation:** Automatic Low (<30%), Medium (30-70%), High (>70%) categorization
- ✅ **Explainable AI:** SHAP bar plots show top 3 features driving each prediction
    - Example: `Contract=Month-to-month` + `Tenure<12 months` + `MonthlyCharges>₹70` = High Risk
- ✅ **Business Intelligence:** KPI cards showing Total Customers, Predicted Churn Count, Churn Rate %
- ✅ **Interactive Visualizations:** Plotly charts for churn distribution, feature importance
- ✅ **Export Functionality:** Download filtered high-risk customer list as CSV for CRM
- ✅ **Dark Mode UI:** Professional corporate theme, works in light/dark browser modes
- ✅ **Data Validation:** Handles missing values, encoding, and feature scaling automatically

---

## 📊 Sample Insights from Model

Top 3 Churn Drivers identified by SHAP:
1. **Contract Type:** Month-to-month customers 3x more likely to churn vs 2-year contracts
2. **Tenure:** Customers < 12 months have 65% churn probability
3. **Tech Support:** No tech support increases churn risk by 40%

---

## 💻 Run Locally

### Prerequisites
- Python 3.11
- pip package manager
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/EvuriVyshnavi/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment - recommended
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
