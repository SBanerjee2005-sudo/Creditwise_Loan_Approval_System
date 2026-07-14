import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="CreditWise Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern fintech theme
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #102a43;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(#102a43, #334e68);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #486581;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Cards */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0f52ba 0%, #0a3a82 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(15, 82, 186, 0.2);
    }
    
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(15, 82, 186, 0.4);
        background: linear-gradient(135deg, #0a3a82 0%, #072a5e 100%);
        color: white;
    }
    
    /* Result Cards */
    .success-card {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(5, 150, 105, 0.4), 0 10px 10px -5px rgba(5, 150, 105, 0.1);
        margin-top: 2rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(220, 38, 38, 0.4), 0 10px 10px -5px rgba(220, 38, 38, 0.1);
        margin-top: 2rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .result-text {
        font-size: 1.25rem;
        opacity: 0.9;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load("creditwise_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading models. Please ensure 'creditwise_model.pkl' and 'scaler.pkl' are in the same directory. Details: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=80)
    st.markdown("### CreditWise")
    st.markdown("---")
    st.markdown("#### Project Description")
    st.markdown("An advanced Machine Learning system that predicts loan approval chances based on applicant profiles, financial history, and loan details.")
    
    st.markdown("#### Model Details")
    st.info("**Algorithm:** Random Forest Classifier\n\n**Accuracy:** 90%\n\n**Weighted F1-score:** 0.90")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #718096; font-size: 0.9rem;'>Developed by<br><b>Soham Banerjee</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #718096; font-size: 0.8rem; margin-top: 1rem;'>Technology Stack:<br>Python • Streamlit • Scikit-learn • Pandas • NumPy</div>", unsafe_allow_html=True)

# Main Hero Section
st.markdown("<div class='hero-title'>CreditWise Loan Approval Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Predict whether a loan application is likely to be approved using a trained Machine Learning model.</div>", unsafe_allow_html=True)

# Form for User Input
with st.form("loan_application_form"):
    
    st.markdown("### 👤 Applicant Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, help="Age of the applicant in years")
        dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0, step=1, help="Number of people dependent on the applicant")
    with col2:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
    with col3:
        marital_status = st.selectbox("Marital Status", options=["Single", "Married"])
        
    st.markdown("---")
    
    st.markdown("### 💼 Employment Information")
    col4, col5 = st.columns(2)
    with col4:
        emp_status = st.selectbox("Employment Status", options=["Salaried", "Self-employed", "Contract", "Unemployed"])
    with col5:
        emp_category = st.selectbox("Employer Category", options=["Private", "Government", "MNC", "Business", "Unemployed"])
        
    st.markdown("---")
    
    st.markdown("### 💰 Financial Information")
    col6, col7, col8 = st.columns(3)
    with col6:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=50000.0, step=1000.0)
        existing_loans = st.number_input("Existing Loans Count", min_value=0, max_value=20, value=0, step=1)
    with col7:
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0.0, value=0.0, step=1000.0)
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0, step=10.0)
    with col8:
        savings = st.number_input("Savings ($)", min_value=0.0, value=10000.0, step=1000.0)
        
    st.markdown("---")
    
    st.markdown("### 📄 Loan Details")
    col9, col10, col11 = st.columns(3)
    with col9:
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000.0, value=200000.0, step=5000.0)
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
    with col10:
        loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=360, value=360, step=12)
        collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=250000.0, step=5000.0)
    with col11:
        loan_purpose = st.selectbox("Loan Purpose", options=["Home", "Car", "Personal", "Education", "Business"])
        dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.05, help="Monthly debt payments divided by gross monthly income")

    # Submit Button
    submit_button = st.form_submit_button("Predict Loan Approval", use_container_width=True)

if submit_button:
    # Feature Engineering
    dti_ratio_sq = dti_ratio ** 2
    credit_score_sq = credit_score ** 2
    applicant_income_log = np.log1p(applicant_income)
    
    # Label Encoding for Education
    education_encoded = 0 if education == "Graduate" else 1
    
    # One-Hot Encoding values exactly matching the trained model's drop="first" behavior
    emp_status_salaried = 1 if emp_status == "Salaried" else 0
    emp_status_self_employed = 1 if emp_status == "Self-employed" else 0
    emp_status_unemployed = 1 if emp_status == "Unemployed" else 0
    
    marital_status_single = 1 if marital_status == "Single" else 0
    
    loan_purpose_car = 1 if loan_purpose == "Car" else 0
    loan_purpose_education = 1 if loan_purpose == "Education" else 0
    loan_purpose_home = 1 if loan_purpose == "Home" else 0
    loan_purpose_personal = 1 if loan_purpose == "Personal" else 0
    
    property_area_semiurban = 1 if property_area == "Semiurban" else 0
    property_area_urban = 1 if property_area == "Urban" else 0
    
    gender_male = 1 if gender == "Male" else 0
    
    emp_cat_gov = 1 if emp_category == "Government" else 0
    emp_cat_mnc = 1 if emp_category == "MNC" else 0
    emp_cat_private = 1 if emp_category == "Private" else 0
    emp_cat_unemployed = 1 if emp_category == "Unemployed" else 0
    
    # Constructing the input array in EXACT order of the scaler
    features = [
        coapplicant_income,
        age,
        dependents,
        existing_loans,
        savings,
        collateral_value,
        loan_amount,
        loan_term,
        education_encoded,
        emp_status_salaried,
        emp_status_self_employed,
        emp_status_unemployed,
        marital_status_single,
        loan_purpose_car,
        loan_purpose_education,
        loan_purpose_home,
        loan_purpose_personal,
        property_area_semiurban,
        property_area_urban,
        gender_male,
        emp_cat_gov,
        emp_cat_mnc,
        emp_cat_private,
        emp_cat_unemployed,
        dti_ratio_sq,
        credit_score_sq,
        applicant_income_log
    ]
    
    # Scale the features
    input_data = np.array(features).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    confidence = np.max(probabilities) * 100
    
    # Display Result
    if prediction == 1:
        st.markdown(f"""
        <div class="success-card" role="alert" aria-live="assertive">
            <div class="result-title">🎉 Approved!</div>
            <div class="result-text">Congratulations! Based on the provided details, this loan application is highly likely to be approved.</div>
            <br>
            <div style="font-size: 1.1rem; font-weight: 600; background: rgba(255,255,255,0.2); display: inline-block; padding: 0.5rem 1.5rem; border-radius: 20px;">
                Confidence: {confidence:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-card" role="alert" aria-live="assertive">
            <div class="result-title">⚠️ Rejected</div>
            <div class="result-text">Unfortunately, based on the provided details, this loan application is likely to be rejected.</div>
            <br>
            <div style="font-size: 1.1rem; font-weight: 600; background: rgba(255,255,255,0.2); display: inline-block; padding: 0.5rem 1.5rem; border-radius: 20px;">
                Confidence: {confidence:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
