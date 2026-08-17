import streamlit as st
import joblib
import numpy as np
import base64
import random
from datetime import datetime
from groq import Groq
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page configuration for a professional look
st.set_page_config(
    page_title="FairLoan - Decision Support Tool",
    page_icon="🏦",
    layout="wide"
)

# Helper function to convert local image to base64 for reliable CSS injection
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

img_base64 = get_base64_image("background.jpg")

# Adding a dark semi-transparent overlay on top of the background image to dim its intensity
if img_base64:
    bg_css = f"linear-gradient(rgba(10, 15, 30, 0.85), rgba(10, 15, 30, 0.85)), url('data:image/jpeg;base64,{img_base64}')"
else:
    bg_css = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"

# Custom CSS for styling, banking-grade cards, and gold buttons
st.markdown(f"""
<style>
    .stApp {{
        background-image: {bg_css};
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    
    div.stButton > button:first-child {{
        background-color: #D4AF37 !important;
        color: #000000 !important;
        font-weight: bold;
        border: none !important;
        border-radius: 6px;
        transition: 0.3s;
    }}
    
    div.stButton > button:first-child:hover {{
        background-color: #C5A028 !important;
        color: #000000 !important;
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.4);
    }}
    
    .banking-card {{
        background-color: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(212, 175, 55, 0.3);
        padding: 20px;
        border-radius: 8px;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }}
</style>
""", unsafe_allow_html=True)

# Load trained model, scaler, and Groq client
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# Generate a unique reference ID for the auditing trail if not already present
if 'ref_id' not in st.session_state:
    st.session_state['ref_id'] = f"FL-{random.randint(100000, 999999)}"

# Initialize session history log if not present
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Function to generate a formatted PDF report using ReportLab
def create_pdf_report(session_ref, history_logs):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15
    )
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    # Document Header
    story.append(Paragraph("FairLoan — Credit Risk Evaluation Report", title_style))
    story.append(Paragraph(f"Session Reference: <b>{session_ref}</b> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    
    if not history_logs:
        story.append(Paragraph("No loan evaluations recorded in this session.", body_style))
    else:
        for idx, item in enumerate(history_logs, 1):
            story.append(Paragraph(f"Evaluation #{idx} — [{item['timestamp']}]", section_style))
            
            # Summary Table of Parameters
            data = [
                [Paragraph(f"<b>Income:</b> GHS {item['income']:,.2f}", body_style),
                 Paragraph(f"<b>Loan Amount:</b> GHS {item['loan_amount']:,.2f}", body_style)],
                [Paragraph(f"<b>Credit Score:</b> {item['credit_score']}", body_style),
                 Paragraph(f"<b>Decision:</b> {item['decision']}", body_style)]
            ]
            t = Table(data, colWidths=[250, 250])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
            
            # Analyst Note Section
            story.append(Paragraph(f"<b>Analyst Briefing Note:</b><br/>{item['explanation']}", body_style))
            story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

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

# Sidebar Layout with Categorized Sections
with st.sidebar:
    st.markdown(f" **Session Ref:** `{st.session_state['ref_id']}`")
    st.divider()
    
    st.header("📋 Applicant Profile")

    with st.expander("👤 Personal & Demographics", expanded=True):
        person_age = st.number_input("Age:", min_value=18, max_value=100, value=25)
        person_education_input = st.selectbox("Education Level:", ["Not educated", "Primary", "High School", "Associate", "Bachelor", "Master", "Doctorate"])
        education_fallback = {"Not educated": "High School", "Primary": "High School"}
        person_education = education_fallback.get(person_education_input, person_education_input)
        home_ownership = st.selectbox("Home Ownership:", ["MORTGAGE", "OTHER", "OWN", "RENT"])

    with st.expander("💰 Financials & Employment", expanded=False):
        person_income = st.number_input("Annual Income (GHS):", min_value=0, value=50000, step=1000)
        person_emp_exp = st.number_input("Years of Employment Experience:", min_value=0, max_value=60, value=5)

    with st.expander("📝 Loan Parameters", expanded=False):
        loan_amnt = st.number_input("Loan Amount Requested (GHS):", min_value=0, value=10000, step=500)
        loan_int_rate = st.number_input("Loan Interest Rate (%):", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        
        intent_mapping = {
            "Debt Consolidation": "DEBTCONSOLIDATION",
            "Education": "EDUCATION",
            "Home Improvement": "HOMEIMPROVEMENT",
            "Medical": "MEDICAL",
            "Personal": "PERSONAL",
            "Venture": "VENTURE"
        }
        selected_intent_label = st.selectbox("Loan Purpose:", list(intent_mapping.keys()))
        loan_intent = intent_mapping[selected_intent_label]

    with st.expander("🛡️ Credit & Risk History", expanded=False):
        cb_person_cred_hist_length = st.number_input("Credit History Length (years):", min_value=0, value=5)
        credit_score = st.number_input("Credit Score:", min_value=300, max_value=850, value=650)
        prior_default = st.selectbox("Prior Loan Default on File:", ["No", "Yes"])

    st.divider()
    evaluate_btn = st.button("🚀 Evaluate Loan Application", type="primary", use_container_width=True)

    # Session Log & PDF Report Download Section in Sidebar
    st.divider()
    st.write(f"Evaluations logged: **{len(st.session_state['history'])}**")
    
    # Generate PDF binary for download
    pdf_buffer = create_pdf_report(st.session_state['ref_id'], st.session_state['history'])
        
    st.download_button(
        label="Download Session Report (PDF)",
        data=pdf_buffer,
        file_name=f"FairLoan_Audit_Report_{st.session_state['ref_id']}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# Main Content Area for Results
if evaluate_btn:
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

    st.subheader("Evaluation Results & Risk Tier")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container():
            if prior_default == "Yes":
                decision = "Flagged for Manual Review"
                st.error(f"🔴 **Tier 3 Risk: Flagged**")
                st.warning("Requires manual review due to a prior loan default on file. The model has not generated an automated approval.")
            else:
                if prediction == 1:
                    decision = "Approved"
                    st.success(f"🟢 **Tier 1 Risk: Low Risk**\n\nModel Recommendation: **Approve**")
                    st.metric(label="Model Confidence", value=f"{probability:.0%}")
                    st.progress(float(probability), text="Approval Strength")
                else:
                    decision = "Not Approved"
                    st.error(f"🔴 **Tier 2 Risk: Elevated Risk**\n\nModel Recommendation: **Do Not Approve**")
                    st.metric(label="Rejection Confidence", value=f"{(1-probability):.0%}")
                    st.progress(float(1 - probability), text="Risk Index")

    with col2:
        with st.spinner("Synthesizing analyst briefing note..."):
            explanation = generate_explanation(
                decision=decision,
                income=person_income,
                credit_score=credit_score,
                loan_amount=loan_amnt,
                prior_default=(prior_default == "Yes")
            )

        st.markdown("#### 📝 Internal Analyst Briefing Note")
        st.info(explanation)
        
        # Automatically append this evaluation to the session audit log history
        current_eval = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "income": person_income,
            "loan_amount": loan_amnt,
            "credit_score": credit_score,
            "decision": decision,
            "explanation": explanation
        }
        if not st.session_state['history'] or st.session_state['history'][-1] != current_eval:
            st.session_state['history'].append(current_eval)
            st.rerun()
else:
    st.info("👈 Adjust the applicant details in the sidebar and click **Evaluate Loan Application** to view the model's decision support breakdown.")
