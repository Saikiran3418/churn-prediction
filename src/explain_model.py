import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

# load model and data
print("Loading model and data...")
with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

print("✅ Model loaded!")

# create SHAP explainer
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("✅ SHAP values calculated!")

# create output folder for plots
os.makedirs('models', exist_ok=True)

# Plot 1 - Feature importance summary
print("\nGenerating feature importance plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)
plt.title("Top Features Driving Customer Churn")
plt.tight_layout()
plt.savefig('models/shap_importance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: models/shap_importance.png")

# Plot 2 - Detailed SHAP dot plot
print("\nGenerating detailed SHAP plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test,
    show=False
)
plt.title("How Each Feature Affects Churn Prediction")
plt.tight_layout()
plt.savefig('models/shap_detailed.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: models/shap_detailed.png")

# show top 5 insights in plain English
print("\n" + "="*50)
print("TOP CHURN INSIGHTS (Plain English)")
print("="*50)

feature_impact = pd.DataFrame({
    'feature': X_test.columns,
    'impact': abs(shap_values).mean(0)
}).sort_values('impact', ascending=False)

insights = {
    'Contract': 'Month-to-month contracts churn most',
    'OnlineSecurity': 'No online security = higher churn',
    'InternetService': 'Fiber optic customers churn more',
    'TechSupport': 'No tech support = higher churn',
    'tenure': 'New customers churn more than loyal ones',
    'MonthlyCharges': 'Higher bills = higher churn risk',
    'TotalCharges': 'Low total charges = newer customer',
}

for i, row in feature_impact.head(5).iterrows():
    feat = row['feature']
    note = insights.get(feat, 'Significant churn predictor')
    print(f"\n#{list(feature_impact['feature']).index(feat)+1} {feat}")
    print(f"   → {note}")

print("\n✅ SHAP analysis complete!")
print("📊 Check models/ folder for plots!")