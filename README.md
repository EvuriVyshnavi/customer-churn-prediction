# 📊 Customer Churn Prediction Dashboard

> Predict customer churn using XGBoost and Explainable AI with an interactive Streamlit dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-0066CC?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

### 🚀 Live Demo
**[View Deployed App](https://your-username-customer-churn-prediction.streamlit.app)** ← Deploy ayyaka ee link marchu

---

## 📌 Problem Statement
Telecom companies lose **15-25%** of customers annually to churn. Identifying high-risk customers early enables proactive retention strategies, saving **₹24,000/year** per customer.

## 🎯 Solution
This dashboard uses **XGBoost + SMOTE + SHAP** to:
1. Predict churn probability for each customer
2. Segment customers into Low/Medium/High risk
3. Explain *why* a customer might churn using SHAP values
4. Export high-risk lists for retention teams

---

## 🛠️ Tech Stack

| Category | Technology |
| --- | --- |
| **ML Model** | XGBoost Classifier |
| **Imbalance Handling** | SMOTE Oversampling |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Frontend** | Streamlit |
| **Visualization** | Plotly, Matplotlib |
| **Data Processing** | Pandas, Scikit-learn |

---

## 📈 Model Performance

| Metric | Score | Interpretation |
| --- | --- | --- |
| **ROC-AUC** | 0.82 | Strong discriminative power |
| **Accuracy** | 79.2% | 8/10 predictions correct |
| **Precision** | 0.65 | 65% of flagged churners actually churn |
| **Recall** | 0.58 | Captures 58% of actual churners |

> **Note:** Churn datasets have inherent class imbalance (27% churn vs 73% no-churn). These metrics reflect real-world performance. Precision/Recall trade-off can be adjusted via threshold tuning for business needs.

---

## 🎨 Features

- ✅ **Bulk Prediction:** Upload CSV and get instant churn scores for thousands of customers
- ✅ **Risk Segmentation:** Automatic Low/Medium/High categorization based on probability
- ✅ **Explainable AI:** SHAP bar plots show top features driving churn (Contract, Tenure, MonthlyCharges)
- ✅ **Business Metrics:** KPI cards showing Total Customers, Predicted Churn, Churn Rate
- ✅ **Export Functionality:** Download high-risk customer list for CRM integration
- ✅ **Dark Mode UI:** Professional corporate theme compatible with light/dark modes

---

## 📊 Screenshots

| Dashboard Overview | SHAP Feature Importance |
| --- | --- |
| ![Dashboard](https://via.placeholder.com/400x250.png?text=Add+Screenshot+Here) | ![SHAP](https://via.placeholder.com/400x250.png?text=Add+SHAP+Plot) |

*Replace placeholder images with actual screenshots after deployment*

---

## 💻 Run Locally

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
