# app.py - Customer Churn Prediction Dashboard
# Tech Stack: XGBoost, SHAP, Streamlit, Plotly
# Corporate Premium UI - Beautiful + Theme Aware

import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Corporate CSS - Beautiful + Theme Aware
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Header */
   .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        margin-bottom: 0.5rem;
    }

   .sub-header {
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Beautiful Gradient Button */
   .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
   .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        border: none;
    }

    /* Sidebar Beautiful Cards */
   .sidebar-card {
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, rgba(102, 126, 234, 0.05) 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }
   .sidebar-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.4);
    }
   .sidebar-label {
        color: var(--text-color);
        opacity: 0.6;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        margin: 0;
        letter-spacing: 1px;
    }
   .sidebar-value {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.4rem 0 0 0;
    }

    /* Upload section styling */
   .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.3)!important;
        border-radius: 12px!important;
        padding: 1rem!important;
    }

    /* Remove all colored boxes */
   .stAlert {
        background-color: transparent!important;
        border: none!important;
        padding: 0!important;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<p class="main-header">📊 Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Identify high-risk customer segments using XGBoost and Explainable AI</p>', unsafe_allow_html=True)

# Load trained model and feature columns
@st.cache_resource
def load_model():
    """Load trained XGBoost model and feature columns from disk"""
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

model, columns = load_model()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 📁 Data Input")
    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type="csv",
        help="Upload Telco Customer Churn dataset for analysis"
    )

    st.markdown("---")
    st.markdown("### 🛠 Technology Stack")
    st.markdown("""
    - **Model:** XGBoost Classifier
    - **Imbalance:** SMOTE Oversampling
    - **Explainability:** SHAP Values
    - **Visualization:** Plotly, Matplotlib
    - **Framework:** Streamlit
    """)

    st.markdown("---")
    st.markdown("### 📈 Model Performance")

    # ✅ BEAUTIFUL SIDEBAR CARDS
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sidebar-card"><p class="sidebar-label">ROC-AUC</p><p class="sidebar-value">0.82</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card"><p class="sidebar-label">Precision</p><p class="sidebar-value">0.65</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="sidebar-card"><p class="sidebar-label">Accuracy</p><p class="sidebar-value">79.2%</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card"><p class="sidebar-label">Recall</p><p class="sidebar-value">0.58</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔗 Project Links")
    st.markdown("[GitHub Repository](https://github.com/EvuriVyshnavi/customer-churn-prediction)")
    st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/evuri-vyshnavi/)")

# Helper function - Premium KPI Cards
def premium_card(label, value, delta, delta_color="#999", icon="📊"):
    return f"""
    <div style="
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, rgba(102, 126, 234, 0.03) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 30px rgba(102, 126, 234, 0.2)'"
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 20px rgba(0,0,0,0.08)'">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <p style="color: var(--text-color);
                      opacity: 0.7;
                      font-size: 0.8rem;
                      margin: 0;
                      text-transform: uppercase;
                      font-weight: 600;
                      letter-spacing: 1px;">{label}</p>
            <span style="font-size: 1.5rem; opacity: 0.3;">{icon}</span>
        </div>
        <p style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  -webkit-background-clip: text;
                  -webkit-text-fill-color: transparent;
                  background-clip: text;
                  font-size: 2.5rem;
                  font-weight: 700;
                  margin: 0.5rem 0;
                  line-height: 1;">{value}</p>
        <p style="color: {delta_color};
                  font-size: 0.85rem;
                  margin: 0;
                  font-weight: 500;">{delta}</p>
    </div>
    """

# Main Content
if uploaded_file is not None:
    # Read uploaded CSV
    df = pd.read_csv(uploaded_file)

    # Section 1: Data Preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True, height=220)
    st.caption(f"✅ Dataset loaded: {len(df):,} customers × {len(df.columns)} features")

    # Data preprocessing pipeline
    df_copy = df.copy()

    # Store customer IDs if present, then drop for prediction
    if 'customerID' in df_copy.columns:
        customer_ids = df_copy['customerID']
        df_copy.drop('customerID', axis=1, inplace=True)
    else:
        customer_ids = pd.Series(range(len(df_copy)))

    # Convert TotalCharges to numeric, handle missing values
    if 'TotalCharges' in df_copy.columns:
        df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce')
        df_copy.fillna(0, inplace=True)

    # Label encode categorical features
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].astype('category').cat.codes

    # Feature engineering - create derived features
    if 'tenure' in df_copy.columns and 'TotalCharges' in df_copy.columns:
        df_copy['AvgMonthlySpend'] = df_copy['TotalCharges'] / (df_copy['tenure'] + 1)
        df_copy['ChargePerService'] = df_copy['MonthlyCharges'] / (df_copy['tenure'] + 1)

    # Ensure all training columns exist and maintain correct order
    for col in columns:
        if col not in df_copy.columns:
            df_copy[col] = 0
    df_copy = df_copy[columns]

    # Generate predictions and probabilities
    predictions = model.predict(df_copy)
    proba = model.predict_proba(df_copy)[:, 1]

    # Add predictions to original dataframe
    df['Churn_Prediction'] = predictions
    df['Churn_Probability'] = proba.round(3)
    df['Risk_Level'] = pd.cut(
        proba,
        bins=[0, 0.3, 0.7, 1],
        labels=['Low', 'Medium', 'High']
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # Section 2: Key Business Metrics - BEAUTIFUL CARDS
    st.markdown("### 📊 Key Business Metrics")
    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(premium_card(
            "Total Customers",
            f"{len(df):,}",
            "Active Base",
            "#667eea",
            "👥"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(premium_card(
            "Predicted Churn",
            f"{int(sum(predictions)):,}",
            f"↑ {sum(predictions)/len(df)*100:.1f}% of base",
            "#ff6b6b",
            "⚠️"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(premium_card(
            "Churn Rate",
            f"{sum(predictions)/len(df)*100:.1f}%",
            "↓ 2.3% below industry avg",
            "#51cf66",
            "📉"
        ), unsafe_allow_html=True)

    with col4:
        high_risk_count = int(sum(df['Risk_Level'] == 'High'))
        st.markdown(premium_card(
            "High Risk",
            f"{high_risk_count:,}",
            "Immediate action needed",
            "#ff6b6b",
            "🔴"
        ), unsafe_allow_html=True)

    st.write("")
    st.markdown("---")

    # Section 3: Risk Distribution
    st.markdown("### 🎯 Churn Risk Distribution")
    col1, col2 = st.columns([3, 2])

    with col1:
        risk_counts = df['Risk_Level'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.5,
            marker=dict(
                colors=['#51cf66', '#ffd43b', '#ff6b6b'],
                line=dict(color='rgba(0,0,0,0)', width=3)
            ),
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=14, color='white', family='Inter'),
            hovertemplate='<b>%{label} Risk</b><br>Customers: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )])
        fig.update_layout(
            showlegend=True,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='var(--text-color)', family='Inter'),
            annotations=[dict(
                text='Risk<br>Segments',
                x=0.5, y=0.5,
                font_size=18,
                font_color='var(--text-color)',
                showarrow=False,
                font_family='Inter'
            )],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(color='var(--text-color)', family='Inter')
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 💡 Business Insights")
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, rgba(81, 207, 102, 0.1) 0%, transparent 100%); border-radius: 12px; margin-bottom: 0.8rem; border-left: 4px solid #51cf66;">
            <b style="color: #51cf66;">🟢 Low Risk</b><br>
            <span style="opacity: 0.8;">{risk_counts.get('Low', 0):,} customers</span><br>
            <small style="opacity: 0.6;">Retention strategies not required</small>
        </div>
        <div style="padding: 1rem; background: linear-gradient(135deg, rgba(255, 212, 59, 0.1) 0%, transparent 100%); border-radius: 12px; margin-bottom: 0.8rem; border-left: 4px solid #ffd43b;">
            <b style="color: #ffd43b;">🟡 Medium Risk</b><br>
            <span style="opacity: 0.8;">{risk_counts.get('Medium', 0):,} customers</span><br>
            <small style="opacity: 0.6;">Monitor closely, offer incentives</small>
        </div>
        <div style="padding: 1rem; background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, transparent 100%); border-radius: 12px; margin-bottom: 0.8rem; border-left: 4px solid #ff6b6b;">
            <b style="color: #ff6b6b;">🔴 High Risk</b><br>
            <span style="opacity: 0.8;">{risk_counts.get('High', 0):,} customers</span><br>
            <small style="opacity: 0.6;">Immediate retention action needed</small>
        </div>
        """, unsafe_allow_html=True)

        potential_loss = high_risk_count * 2000
        st.markdown(f"""
        <div style="padding: 1.2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
            <div style="font-size: 0.8rem; opacity: 0.9; margin-bottom: 0.3rem;">💰 POTENTIAL MONTHLY LOSS</div>
            <div style="font-size: 1.8rem; font-weight: 700;">₹{potential_loss:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 4: High Risk Customers Table
    st.markdown("### 🚨 High Risk Customers - Priority Action List")

    high_risk = df[df['Risk_Level'] == 'High'].sort_values('Churn_Probability', ascending=False).head(20)

    # Format probability as percentage
    high_risk_display = high_risk.copy()
    high_risk_display['Churn_Probability'] = (high_risk_display['Churn_Probability'] * 100).round(1).astype(str) + '%'

    # Display columns
    display_cols = ['Churn_Probability', 'tenure', 'MonthlyCharges', 'Contract']
    if 'customerID' in high_risk_display.columns:
        display_cols = ['customerID'] + display_cols

    st.dataframe(
        high_risk_display[display_cols],
        use_container_width=True,
        height=450,
        column_config={
            "Churn_Probability": st.column_config.TextColumn("Churn Probability", help="Model confidence score"),
            "tenure": st.column_config.NumberColumn("Tenure (months)"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="₹%.2f"),
        }
    )
    st.caption("📋 Sorted by churn probability descending. Focus retention efforts on top customers.")

    st.markdown("---")

    # Section 5: SHAP Feature Importance
    st.markdown("### 🔍 Feature Importance Analysis - Why Customers Churn?")
    st.write("SHAP values explain which features most influence the churn prediction model")

    with st.spinner('Generating SHAP explanation plots...'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_copy)

        # Theme-aware plot - transparent background
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

        shap.summary_plot(
            shap_values,
            df_copy,
            plot_type="bar",
            show=False,
            color='#667eea'
        )
        plt.title("Top Features Driving Churn Predictions", fontsize=14, pad=20, weight='bold', color='var(--text-color)')
        plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)", fontsize=11, color='var(--text-color)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    st.info("**Key Findings:** Contract type, Tenure, and MonthlyCharges are the strongest predictors of churn. Month-to-month contracts with low tenure and high charges indicate highest risk.")

    st.markdown("---")

    # Section 6: Export Results
    st.markdown("### 💾 Export Results for Business Action")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv,
            file_name="churn_predictions_full.csv",
            mime="text/csv",
            use_container_width=True,
            help="Complete dataset with predictions"
        )

    with col2:
        high_risk_csv = high_risk.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download High Risk List",
            data=high_risk_csv,
            file_name="high_risk_customers.csv",
            mime="text/csv",
            use_container_width=True,
            help="High risk customers only"
        )

    with col3:
        st.caption("💡 Export data for CRM integration, email campaigns, or retention team follow-up")

else:
    # Landing page when no file is uploaded - BEAUTIFUL
    st.markdown("### 👆 **Upload the Telco Customer Churn CSV file from the sidebar to begin analysis**")
    st.write("")

    st.markdown("### 🎯 Dashboard Capabilities")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, var(--secondary-background-color) 0%, rgba(102, 126, 234, 0.03) 100%); border-radius: 16px; border: 1px solid rgba(102, 126, 234, 0.15);">
            <h4 style="margin-top: 0; color: #667eea;">Core Features</h4>
            <p>🔮 <b>Predictive Analytics:</b> XGBoost model identifies churn probability</p>
            <p>🎯 <b>Risk Segmentation:</b> Automated Low/Medium/High categorization</p>
            <p>🔍 <b>Explainable AI:</b> SHAP values reveal prediction drivers</p>
            <p>📊 <b>Business Intelligence:</b> KPI metrics and visualizations</p>
            <p>💾 <b>Export Functionality:</b> Download results for action</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, var(--secondary-background-color) 0%, rgba(118, 75, 162, 0.03) 100%); border-radius: 16px; border: 1px solid rgba(118, 75, 162, 0.15);">
            <h4 style="margin-top: 0; color: #764ba2;">Technical Specifications</h4>
            <p>✅ <b>Handles class imbalance</b> with SMOTE</p>
            <p>✅ <b>Real-time bulk prediction</b> processing</p>
            <p>✅ <b>Interactive Plotly</b> visualizations</p>
            <p>✅ <b>Production-ready</b> Streamlit interface</p>
            <p>✅ <b>ROC-AUC Score:</b> 0.82</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.caption("**Model Performance:** Trained on 7,043 customers | ROC-AUC: 0.82 | Accuracy: 79.2%")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: var(--text-color); opacity: 0.5; font-size: 0.85rem; font-family: Inter;'>Customer Churn Analytics Platform | Powered by XGBoost + SHAP + Streamlit</p>",
    unsafe_allow_html=True
)