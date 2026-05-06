import pandas as pd

# load dataset
df = pd.read_csv(r"C:\Users\saiki\churn-prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# basic info
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Total customers: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# show column names
print("\n--- Column Names ---")
for col in df.columns:
    print(f"  {col}")

# show churn distribution
print("\n--- Churn Distribution ---")
print(df['Churn'].value_counts())
print(f"\nChurn rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")

# show first 5 rows
print("\n--- First 3 Customers ---")
print(df.head(3).to_string())

# check missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values!")