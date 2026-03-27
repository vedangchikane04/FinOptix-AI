# 🏦 FinOptix AI

**FinOptix AI** is an intelligent financial risk assessment and EMI prediction platform built with Streamlit. It uses dual machine learning models — **classification** for loan eligibility and **regression** for maximum EMI calculation — tracked via MLflow for experiment management.

---

## 🚀 Features

- 🎯 **EMI Calculator** — Precise monthly EMI computation with total interest & repayment
- ⚠️ **Financial Risk Assessment** — 3-tier risk classification: Low, Medium, High
- 📊 **Exploratory Data Analysis (EDA)** — Distribution plots, correlation matrix, outlier detection
- 📈 **MLflow Experiment Tracking** — Monitor, compare and audit all ML training runs
- 🏦 **Dual ML Pipeline** — Classification (eligible/not) + Regression (max EMI amount)
- 💡 **Affordability Insights** — Maximum affordable EMI vs. requested EMI analysis
- 📋 **Loan Summary Report** — Detailed breakdown of principal, interest, and repayment schedule

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Models | Scikit-learn (Classification + Regression) |
| Experiment Tracking | MLflow |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Persistence | Joblib |

---

## 📁 Project Structure

```
Finoptix_AI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── data/
│   └── raw/
│       └── emi_prediction_dataset.csv   # Raw EMI dataset
├── models/                         # Saved trained ML models
├── src/
│   ├── config.py                   # Application configuration
│   ├── data_preprocessing.py       # Data cleaning & encoding
│   ├── feature_engineering.py      # Feature creation logic
│   ├── model_training.py           # ML model training scripts
│   └── mlflow_utils.py             # MLflow experiment utilities
└── mlruns/                         # MLflow run history
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vedangchikane04/FinOptix-AI.git
cd FinOptix-AI
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🧭 How to Use

### 🏠 Home Page
- Overview of the platform with dataset quick stats
- Age and income distribution charts

### 🎯 EMI Calculator & Risk Assessment
1. Enter **Personal Info** (age, gender, marital status, dependents)
2. Fill **Employment & Income** (type, monthly income, tenure)
3. Enter **Loan Details** (amount, tenure, interest rate, credit score)
4. Click **🚀 Calculate EMI & Assess Risk**
5. View results: EMI amount, risk level, factors & loan summary

### 📊 EDA Page
- Load and analyze the EMI dataset
- View distributions, box plots, correlation matrix
- Detect and drill into outliers

### 📈 MLflow Dashboard
- Track all experiment runs with model names and statuses
- Compare classification vs. regression model histories
- Launch MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlruns.db`

---

## 🧮 Risk Assessment Logic

| Risk Factor | Threshold |
|-------------|-----------|
| Debt-to-Income Ratio | > 40% |
| EMI-to-Income Ratio | > 50% |
| Total Debt Ratio | > 60% |
| Credit Score | < 600 |
| Previous Defaults | Any |
| Age | < 25 years |
| Employment Length | < 2 years |

**Risk Score:**
- 0 factors → ✅ Eligible (Low Risk)
- 1–2 factors → ⚠️ Conditional Approval (Medium Risk)
- 3+ factors → ❌ Not Eligible (High Risk)

---

## 📊 Key Input Features

| Feature | Range |
|---------|-------|
| Age | 18–70 |
| Monthly Income | ₹10,000 – ₹5,00,000 |
| Loan Amount | ₹10,000 – ₹50,00,000 |
| Loan Tenure | 6–84 months |
| Interest Rate | 5%–20% p.a. |
| Credit Score | 300–850 |
| Active Loans | 0–10 |

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
mlflow
joblib
```

---

## 👤 Author

**Vedang Chikane**
- GitHub: [@vedangchikane04](https://github.com/vedangchikane04)
- Email: vedangchikane@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
