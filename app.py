import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# page settings
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

# load model and encoders
@st.cache_resource
def load_model():
    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# title
st.title("🔮 Customer Churn Predictor")
st.caption("Predict if a customer will leave — powered by XGBoost + SHAP")
st.divider()

# ---- TABS ----
tab1, tab2 = st.tabs([
    "👤 Single Customer",
    "📋 Batch Upload (Who is leaving?)"
])

# ---- ENCODE FUNCTION ----
def encode_input(df):
    text_cols = [
        'gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            le = encoders[col]
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_ else 0
            )
    return df

# ==============================
# TAB 1 - SINGLE CUSTOMER
# ==============================
with tab1:
    with st.sidebar:
        st.header("👤 Customer Details")
        st.caption("Fill in customer information")

        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?",
            ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

        st.divider()
        st.subheader("📱 Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines",
            ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service",
            ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security",
            ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup",
            ["Yes", "No", "No internet service"])
        protection = st.selectbox("Device Protection",
            ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support",
            ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV",
            ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies",
            ["Yes", "No", "No internet service"])

        st.divider()
        st.subheader("💳 Billing")
        contract = st.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing",
            ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly = st.slider("Monthly Charges ($)",
            18.0, 120.0, 65.0, step=0.5)
        total = st.number_input("Total Charges ($)",
            min_value=0.0,
            value=float(monthly * tenure))

        predict_btn = st.button(
            "🔮 Predict Churn",
            type="primary",
            use_container_width=True
        )

    if not predict_btn:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", "75.4%")
        with col2:
            st.metric("AUC Score", "0.839")
        with col3:
            st.metric("Training Customers", "5,634")

        st.divider()
        st.markdown("### How to use")
        st.markdown("1. Fill in customer details in sidebar")
        st.markdown("2. Click **Predict Churn** button")
        st.markdown("3. See churn probability and reasons")

        st.divider()
        st.markdown("### Top Churn Drivers")
        drivers = {
            "Contract Type": "Month-to-month = highest risk",
            "Tenure": "New customers churn more",
            "Monthly Charges": "Higher bills = higher risk",
            "Online Security": "No security = higher churn",
            "Total Charges": "Low total = newer customer"
        }
        for k, v in drivers.items():
            st.markdown(f"**{k}** — {v}")

    else:
        data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': support,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        input_df = pd.DataFrame([data])
        encoded_df = encode_input(input_df)
        proba = model.predict_proba(encoded_df)[0][1]

        if proba >= 0.7:
            risk = "🔴 HIGH RISK"
            color = "error"
            rec = "Act immediately! Offer annual contract with 20% discount."
        elif proba >= 0.4:
            risk = "🟡 MEDIUM RISK"
            color = "warning"
            rec = "Monitor closely. Consider loyalty rewards."
        else:
            risk = "🟢 LOW RISK"
            color = "success"
            rec = "Customer likely to stay. Maintain service quality."

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Churn Probability",
                f"{proba*100:.1f}%",
                delta=f"{(proba-0.265)*100:.1f}% vs average"
            )
        with col2:
            st.metric("Risk Level", risk)

        if color == "error":
            st.error(f"💡 **Recommendation:** {rec}")
        elif color == "warning":
            st.warning(f"💡 **Recommendation:** {rec}")
        else:
            st.success(f"💡 **Recommendation:** {rec}")

        st.divider()
        st.subheader("🧠 Why this prediction?")

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(encoded_df)

        shap_df = pd.DataFrame({
            'Feature': encoded_df.columns,
            'Impact': shap_vals[0],
            'Value': encoded_df.iloc[0].values
        })
        shap_df['Abs'] = abs(shap_df['Impact'])
        shap_df = shap_df.sort_values(
            'Abs', ascending=False
        ).head(5)

        for _, row in shap_df.iterrows():
            direction = "⬆️ increases" if row['Impact'] > 0 \
                else "⬇️ decreases"
            st.markdown(
                f"**{row['Feature']}** = `{row['Value']:.2f}` "
                f"— {direction} churn risk"
            )

        st.divider()
        st.subheader("📊 SHAP Chart")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [
            '#E24B4A' if x > 0 else '#1D9E75'
            for x in shap_df['Impact']
        ]
        ax.barh(
            shap_df['Feature'],
            shap_df['Impact'],
            color=colors
        )
        ax.set_xlabel("SHAP Value")
        ax.set_title("Top 5 Factors for This Customer")
        ax.axvline(x=0, color='black', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ==============================
# TAB 2 - BATCH UPLOAD
# ==============================
with tab2:
    st.subheader("📋 Who is leaving? — Batch Analysis")
    st.markdown("Upload your customer CSV file and see everyone's churn risk at once!")

    # download sample template
    sample = pd.DataFrame([{
        'customerID': 'CUST-001',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.0,
        'TotalCharges': 780.0
    }])

    st.download_button(
        "⬇️ Download Sample CSV Template",
        sample.to_csv(index=False),
        "sample_customers.csv",
        "text/csv"
    )

    st.divider()

    # file upload
    uploaded = st.file_uploader(
        "Upload customer CSV file",
        type="csv"
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df)} customers!")

        # keep customerID if exists
        has_id = 'customerID' in df.columns
        if has_id:
            customer_ids = df['customerID'].copy()
            df_model = df.drop('customerID', axis=1)
        else:
            customer_ids = pd.Series(
                range(1, len(df)+1)
            ).astype(str)
            df_model = df.copy()

      # encode and predict
        with st.spinner("Analyzing all customers..."):
            # drop Churn column if exists
            if 'Churn' in df_model.columns:
                df_model = df_model.drop('Churn', axis=1)

            # fix TotalCharges
            df_model['TotalCharges'] = pd.to_numeric(
                df_model['TotalCharges'], errors='coerce'
            ).fillna(0)

            # fix SeniorCitizen
            if df_model['SeniorCitizen'].dtype == object:
                df_model['SeniorCitizen'] = df_model[
                    'SeniorCitizen'
                ].map({'Yes': 1, 'No': 0}).fillna(0)

            # make all columns numeric
            encoded = encode_input(df_model)
            for col in encoded.columns:
                encoded[col] = pd.to_numeric(
                    encoded[col], errors='coerce'
                ).fillna(0)

            probas = model.predict_proba(encoded)[:,1]

            # drop Churn column if exists
            if 'Churn' in df_model.columns:
                df_model = df_model.drop('Churn', axis=1)

            # fix SeniorCitizen column
            if df_model['SeniorCitizen'].dtype == object:
                df_model['SeniorCitizen'] = df_model[
                    'SeniorCitizen'
                ].map({'Yes': 1, 'No': 0}).fillna(0)

            encoded = encode_input(df_model)
            probas = model.predict_proba(encoded)[:,1]

        # build results
        results = pd.DataFrame({
            'Customer ID': customer_ids,
            'Churn Probability': (probas * 100).round(1),
            'Risk Level': [
                '🔴 HIGH' if p >= 0.7
                else '🟡 MEDIUM' if p >= 0.4
                else '🟢 LOW'
                for p in probas
            ],
            'Recommendation': [
                'Call immediately — offer discount'
                if p >= 0.7
                else 'Monitor — consider loyalty offer'
                if p >= 0.4
                else 'Stable — maintain service'
                for p in probas
            ]
        }).sort_values(
            'Churn Probability', ascending=False
        )

        # summary metrics
        high = (probas >= 0.7).sum()
        medium = ((probas >= 0.4) & (probas < 0.7)).sum()
        low = (probas < 0.4).sum()

        st.divider()
        st.subheader("📊 Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("🔴 High Risk", high)
        with col3:
            st.metric("🟡 Medium Risk", medium)
        with col4:
            st.metric("🟢 Low Risk", low)

        # estimated revenue at risk
        avg_monthly = 65
        revenue_risk = high * avg_monthly * 12
        st.error(
            f"💰 Estimated Annual Revenue at Risk: "
            f"**${revenue_risk:,}** "
            f"({high} high-risk customers × $65/month × 12)"
        )

        st.divider()
        st.subheader("🔴 High Risk Customers — Act Now!")
        high_risk = results[
            results['Risk Level'] == '🔴 HIGH'
        ]
        if len(high_risk) > 0:
            st.dataframe(
                high_risk,
                use_container_width=True
            )
        else:
            st.success("No high risk customers found!")

        st.subheader("All Customers — Full List")
        st.dataframe(results, use_container_width=True)

        # download results
        st.download_button(
            "⬇️ Download Full Results CSV",
            results.to_csv(index=False),
            "churn_predictions.csv",
            "text/csv"
        )