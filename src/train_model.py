import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# load prepared data
print("Loading prepared data...")
with open('models/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('models/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('models/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('models/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

print(f"✅ Data loaded successfully!")

# Step 1 - train XGBoost model
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=200,      # number of trees
    max_depth=4,           # depth of each tree
    learning_rate=0.1,     # how fast it learns
    scale_pos_weight=2.7,  # handles imbalanced data
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Step 2 - make predictions
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# Step 3 - show performance metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("MODEL PERFORMANCE RESULTS")
print("="*50)
print(f"✅ Accuracy:  {accuracy*100:.1f}%")
print(f"✅ AUC Score: {auc_score:.3f}")
print("\n--- Detailed Report ---")
print(classification_report(
    y_test, y_pred,
    target_names=['Stayed', 'Churned']
))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives  (correctly predicted stayed):  {cm[0][0]}")
print(f"False Positives (wrongly predicted churned):   {cm[0][1]}")
print(f"False Negatives (missed actual churners):      {cm[1][0]}")
print(f"True Positives  (correctly predicted churned): {cm[1][1]}")

# Step 4 - show top important features
print("\n--- Top 10 Most Important Features ---")
feature_names = list(X_train.columns)
importances = model.feature_importances_
feat_imp = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)
for feat, imp in feat_imp[:10]:
    bar = "█" * int(imp * 100)
    print(f"  {feat:<20} {imp:.3f} {bar}")

# Step 5 - save the trained model
with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model saved to models/churn_model.pkl")
print("\n🎉 Training complete!")