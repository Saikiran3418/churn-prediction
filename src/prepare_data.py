import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# load dataset
print("Loading dataset...")
df = pd.read_csv(
    r"C:\Users\saiki\churn-prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
print(f"✅ Loaded {len(df)} customers")

# Step 1 - drop customerID (not useful for prediction)
df = df.drop('customerID', axis=1)
print("✅ Dropped customerID column")

# Step 2 - fix TotalCharges column
# it has spaces instead of numbers for new customers
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce'
)
# fill missing with 0 (new customers have no charges yet)
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print("✅ Fixed TotalCharges column")

# Step 3 - convert Churn to numbers
# Yes = 1 (churned), No = 0 (stayed)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("✅ Converted Churn to numbers (Yes=1, No=0)")

# Step 4 - convert all text columns to numbers
# machine learning only understands numbers!
text_columns = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

label_encoders = {}
for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print(f"✅ Converted {len(text_columns)} text columns to numbers")

# Step 5 - split into features and target
X = df.drop('Churn', axis=1)  # everything except Churn
y = df['Churn']                # only Churn column

print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# Step 6 - split into train and test sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps same churn ratio in both sets
)

print(f"\n✅ Training set: {len(X_train)} customers")
print(f"✅ Testing set:  {len(X_test)} customers")

# Step 7 - save everything for next step
os.makedirs('models', exist_ok=True)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('models/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('models/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('models/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('models/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\n✅ All data saved to models/ folder!")
print("\n--- Final Dataset Summary ---")
print(f"Total features used: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")
print(f"\nChurn distribution in training set:")
print(y_train.value_counts())