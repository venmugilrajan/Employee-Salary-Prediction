import pandas as pd
import joblib
import gradio as gr

# -------------------- Load Model --------------------
model = joblib.load("linear_model.pkl")

# -------------------- Prediction Function --------------------
def predict_salary(experience, education_level, certifications, skills_score,
                   projects_handled, leadership_score, communication_score,
                   location_index, department_index):
    
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
    
    prediction = model.predict(input_data)[0]
    return f"💰 Predicted Salary: {prediction:.2f} K"

# -------------------- Gradio App --------------------
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center; color:#2b2b2b;'>💼 Employee Salary Prediction</h1>")
    gr.Markdown("<p style='text-align:center; color:#555;'>Predict employee salary (in thousands) based on experience, skills, and performance metrics.</p>")

    with gr.Row():
        with gr.Column():
            experience = gr.Number(label="Experience (Years)", value=0)
            certifications = gr.Number(label="Certifications Count", value=0)
            leadership_score = gr.Number(label="Leadership Score", value=0.0)
        
        with gr.Column():
            education_level = gr.Number(label="Education Level (Numeric Code)", value=0)
            projects_handled = gr.Number(label="Projects Handled", value=0)
            communication_score = gr.Number(label="Communication Score", value=0.0)
        
        with gr.Column():
            skills_score = gr.Number(label="Skills Score", value=0.0)
            location_index = gr.Number(label="Location Index", value=0)
            department_index = gr.Number(label="Department Index", value=0)

    predict_btn = gr.Button("🔍 Predict Salary", variant="primary")
    output_text = gr.Textbox(label="Prediction", interactive=False)

    predict_btn.click(
        fn=predict_salary,
        inputs=[experience, education_level, certifications, skills_score,
                projects_handled, leadership_score, communication_score,
                location_index, department_index],
        outputs=output_text
    )

    gr.Markdown("---")
    gr.Markdown("This model uses **Lasso Regression with Cross-Validation**, which automatically selects the most relevant features and removes unnecessary ones.")
    gr.Markdown("👨‍💻 Developed by **Venmugil Rajan**")

# Launch Gradio app
demo.launch()
