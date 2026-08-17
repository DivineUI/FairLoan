import streamlit as st
import joblib
import numpy as np
from groq import Groq

# Page configuration for a professional look
st.set_page_config(
    page_title="FairLoan - Decision Support Tool",
    page_icon="🏦",
    layout="wide"
)

# Load trained model, scaler, and Groq client
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# Explanation function
def generate_explanation(decision, income, credit_score, loan_amount, prior_default):
    prompt = f"""
You are a lending analyst writing a brief internal note for a loan officer, summarizing why a model reached a loan recommendation.

Recommendation: {decision}
Applicant's income: GHS {income:,.2f}
Applicant's credit score: {credit_score}
Loan amount requested: GHS {loan_amount:,.2f}
Prior loan default on file: {"Yes" if prior_default else "No"}

Write 3-4 sentences in a natural, conversational tone, as if briefing a colleague directly. Avoid starting with generic phrases like "This model suggests," "Based on the data," or "The recommendation is." Just get straight to the point.
Base your explanation only on the information given above. Do not invent or assume any information not provided.
This is a recommendation to support the officer's decision, not a final or automatic decision.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# App Header
st.title("🏦 FairLoan — Loan Officer Decision Support")
st.markdown("Use this workspace to evaluate applicant profiles, review automated model recommendations, and read quick AI-generated summary notes.")
st.divider()

# Sidebar Layout for Inputs (Mimics a dashboard control panel)
with st.sidebar:
    st.header("📋 Applicant Information")
    
    person_age = st.number_input("Age", min_value=18, max_value=100, value=25)

    person_education_input = st.selectbox(
        "Education Level", 
        ["Not educated", "Primary", "High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )
    education_fallback = {"Not educated": "High School", "Primary": "High School"}
    person_education = education_fallback.get(person_education_input, person_education_input)

    person_income = st.number_input("Annual Income (GHS)", min_value=0, value=50000, step=1000)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, max_value=60, value=5)
    
    loan_amnt = st.number_input("Loan Amount Requested (GHS)", min_value=0, value=10000, step=500)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

    home_ownership = st.selectbox(
        "Home Ownership", 
        ["MORTGAGE", "OTHER", "OWN", "RENT"]
    )
    
    # Mapping raw loan intents to natural, human-readable labels
    intent_mapping = {
        "Debt Consolidation": "DEBTCONSOLIDATION",
        "Education": "EDUCATION",
        "Home Improvement": "HOMEIMPROVEMENT",
        "Medical": "MEDICAL",
        "Personal": "PERSONAL",
        "Venture": "VENTURE"
    }
    
    selected_intent_label = st.selectbox("Loan Purpose", list(intent_mapping.keys()))
    loan_intent = intent_mapping[selected_intent_label]

    prior_default = st.selectbox("Prior Loan Default on File", ["No", "Yes"])
    
    st.divider()
    evaluate_btn = st.button("🚀 Evaluate Loan Application", type="primary", use_container_width=True)

# Main Content Area for Results
if evaluate_btn:
    # Encoding logic
    education_order = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
    person_education_encoded = education_order[person_education]

    home_MORTGAGE = 1 if home_ownership == "MORTGAGE" else 0
    home_OTHER = 1 if home_ownership == "OTHER" else 0
    home_OWN = 1 if home_ownership == "OWN" else 0
    home_RENT = 1 if home_ownership == "RENT" else 0

    intent_DEBTCONSOLIDATION = 1 if loan_intent == "DEBTCONSOLIDATION" else 0
    intent_EDUCATION = 1 if loan_intent == "EDUCATION" else 0
    intent_HOMEIMPROVEMENT = 1 if loan_intent == "HOMEIMPROVEMENT" else 0
    intent_MEDICAL = 1 if loan_intent == "MEDICAL" else 0
    intent_PERSONAL = 1 if loan_intent == "PERSONAL" else 0
    intent_VENTURE = 1 if loan_intent == "VENTURE" else 0

    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0

    features = np.array([[
        person_age,
        person_education_encoded,
        person_income,
        person_emp_exp,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        home_MORTGAGE,
        home_OTHER,
        home_OWN,
        home_RENT,
        intent_DEBTCONSOLIDATION,
        intent_EDUCATION,
        intent_HOMEIMPROVEMENT,
        intent_MEDICAL,
        intent_PERSONAL,
        intent_VENTURE
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Evaluation Results")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        if prior_default == "Yes":
            decision = "Flagged for Manual Review"
            st.warning(f"**Status:**\n\n{decision}")
            st.caption("Requires manual review due to a prior loan default on file.")
        else:
            if prediction == 1:
                decision = "Approved"
                st.success(f"**Recommendation:**\n\nApprove")
                st.metric(label="Model Confidence", value=f"{probability:.0%}")
            else:
                decision = "Not Approved"
                st.error(f"**Recommendation:**\n\nDo Not Approve")
                st.metric(label="Model Confidence", value=f"{(1-probability):.0%}")

    with col2:
        with st.spinner("Synthesizing analyst briefing note..."):
            explanation = generate_explanation(
                decision=decision,
                income=person_income,
                credit_score=credit_score,
                loan_amount=loan_amnt,
                prior_default=(prior_default == "Yes")
            )

        st.markdown("#### 📝 Internal Analyst Note")
        st.info(explanation)
else:
    st.info("👈 Adjust the applicant details in the sidebar and click **Evaluate Loan Application** to view the model's decision support breakdown.")
