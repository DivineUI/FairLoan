# FairLoan

A machine learning application that supports loan approval decisions using historical application data.

---

## Problem Statement

Loan officers must evaluate applications and decide whether to approve or reject loans—often under limited review time, leading to potential inconsistency or delays. **FairLoan** serves as an intelligent decision-support system that provides data-driven recommendations without replacing human judgment. 

Additionally, the system incorporates fairness audits and SHAP explainability to evaluate potential algorithmic bias across protected attributes (e.g., gender, race) and maintain transparency in decision-making.

---

## Key Features

* **Loan Approval Prediction:** Machine learning models trained to evaluate risk and predict loan application outcomes.
* **Explainable AI (SHAP):** Clear breakdown of feature contributions for each prediction to assist loan officers in understanding decision factors.
* **Fairness & Bias Auditing:** Evaluation of model performance across demographic sub-groups.
* **Applicant Routing Logic:** Rule-based fallback/routing for applicants with prior loan defaults.
* **Interactive Web App:** A Streamlit interface with dynamic evaluation logic and custom report generation (PDF).

---

##  Dataset

* **Source:** [Loan Approval Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
* **Size:** ~45,000 records, 14 variables

---

## Tech Stack & Tools

* **Language:** Python
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Logistic Regression, Random Forest), `shap`
* **Web App & UI:** `Streamlit`
* **Document Generation:** `reportlab`
* **Development Environment:** Google Colab / Jupyter Notebooks

---

##  Project Structure

```text
├── data/                       # Raw and processed datasets
├── docs/                       # Project documentation & assets
├── notebooks/                  # Experimental Jupyter notebooks
│   ├── FairLoan baseline workflow.ipynb  # Model training, SHAP
│   ├── LLM_API.ipynb                    # API integration experiments
│   └── Pipeline.ipynb                   # Full pipeline/reusable functions
├── app.py                      # Main Streamlit web application
├── credit_cleaned (2).csv      # Cleaned dataset
├── credit_encoded.csv          # Encoded dataset for modeling
├── model.pkl                   # Serialized Random Forest model
├── scaler.pkl                  # Serialized feature scaler
├── background.jpg              # App UI background asset
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```
---
## How to run the app

Make sure the pre-trained artifacts (`model.pkl` and `scaler.pkl`) are in the same folder as `app.py` so the app can load feature transformations and predictions properly.

Launch the Streamlit interface:
```bash
   streamlit run app.py
```
The app is already hosted and can be accessed here: 
``` bash 
https://fairloan-6v3c5kch2pozua4hggzeuv.streamlit.app/
```
Or scan the QR code:
<img width="1147" height="1147" alt="WhatsApp Image 2026-08-17 at 10 27 46 PM" src="https://github.com/user-attachments/assets/b581cd5e-4ec2-4b9f-a26f-5626071657c5" />

