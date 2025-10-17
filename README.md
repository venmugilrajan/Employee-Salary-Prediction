# 💼 Employee Salary Prediction App

A simple **Streamlit web app** that predicts employee salaries (in thousands) using **Lasso Regression**.  
The model identifies important factors like experience, skills, and leadership to estimate salary.
## Visit this
1.Visit this in hugging face: https://huggingface.co/spaces/venmugilrajan/Employee_Salary_Prediction_AI

2.Visit this in streamlit: https://salarypredictionai.streamlit.app/

---

## 🚀 Features
- Predict salary in real-time  
- Automatic feature selection (LassoCV)  
- Clean and modern UI  
- Lightweight and easy to deploy  

---

## 🧠 Model Info
- Algorithm: **Lasso Regression (5-fold CV)**
- Target: `salary_k`
- Features: experience, education_level, certifications, skills_score, projects_handled, leadership_score, communication_score, location_index, department_index

---

## ⚙️ Run Locally
```bash
pip install -r requirements.txt
python app.py

