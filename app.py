import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('linear_model.pkl')

st.set_page_config(
    page_title="💼 Employee Salary Prediction",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            color: #000000;
        }
        .main-title {
            font-size: 32px;
            font-weight: 700;
            color: #2b2b2b;
            text-align: center;
            padding-bottom: 0.2em;
        }
        .sub-text {
            text-align: center;
            color: #555;
            font-size: 16px;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #0078ff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #005fcc;
        }
        .result-box {
            background-color: #e8f4ff;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            color: #004080;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💼 Employee Salary Prediction</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Predict employee salary (in thousands) based on experience, skills, and performance metrics.</p>', unsafe_allow_html=True)

st.subheader("📋 Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=0)
    certifications = st.number_input("Certifications Count", min_value=0, max_value=20, value=0)
    leadership_score = st.number_input("Leadership Score", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col2:
    education_level = st.number_input("Education Level (Numeric Code)", min_value=0, max_value=20, value=0)
    projects_handled = st.number_input("Projects Handled", min_value=0, max_value=100, value=0)
    communication_score = st.number_input("Communication Score", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col3:
    skills_score = st.number_input("Skills Score", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    location_index = st.number_input("Location Index", min_value=0, max_value=10, value=0)
    department_index = st.number_input("Department Index", min_value=0, max_value=10, value=0)

input_data = pd.DataFrame({
    'experience': [experience],
    'education_level': [education_level],
    'certifications': [certifications],
    'skills_score': [skills_score],
    'projects_handled': [projects_handled],
    'leadership_score': [leadership_score],
    'communication_score': [communication_score],
    'location_index': [location_index],
    'department_index': [department_index]
})

st.write("")
predict_btn = st.button("🔍 Predict Salary")

if predict_btn:
    prediction = model.predict(input_data)[0]
    st.markdown(f'<div class="result-box">💰 Predicted Salary: <b>{prediction:.2f} K</b></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("This model uses **Lasso Regression with Cross-Validation**, which automatically selects the most relevant features and removes unnecessary ones.")

st.markdown("---")
st.caption("👨‍💻 Developed by **Venmugil Rajan**")
