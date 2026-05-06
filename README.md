# 🔮 Customer Churn Prediction App

ML-powered web app that predicts customer churn 
using XGBoost and SHAP explainability.

## 🚀 Live Demo
👉 [Click here to try it live](https://churn-prediction-6ffy9mg3ycyzhd2jtsagwv.streamlit.app/)

## 📊 Features
- Single customer churn prediction
- Batch CSV upload — see ALL customers at risk
- SHAP explainability — shows WHY customer will churn
- Revenue at risk calculation
- Downloadable results CSV

## 🛠️ Tech Stack
- XGBoost — ML model (75.4% accuracy, AUC 0.839)
- SHAP — Model explainability
- Streamlit — Web interface
- Scikit-learn — Data preprocessing
- Pandas — Data handling

## 📈 Model Performance
- Accuracy: 75.4%
- AUC Score: 0.839
- Trained on: 5,634 customers
- Dataset: Telco Customer Churn (Kaggle)

## ⚙️ How to Run

1. Clone the repo
   git clone https://github.com/Saikiran3418/churn-prediction.git
   cd churn-prediction

2. Create virtual environment
   python -m venv venv

3. Activate virtual environment
   venv\Scripts\activate

4. Install libraries
   pip install -r requirements.txt

5. Download dataset
   - Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   - Download and place CSV in data/ folder

6. Prepare and train the model
   python src/prepare_data.py
   python src/train_model.py

7. Run the app
   streamlit run app.py

## 🧠 Key Insights
- Contract type is #1 churn predictor
- Month-to-month customers churn 3x more
- New customers (low tenure) are highest risk
- Higher monthly charges = higher churn risk

## 👤 Author
Sai Kiran Reddy