from pathlib import Path
import json
import pandas as pd
import streamlit as st
import joblib


st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Performance Predictor")
st.caption(
    "Educational analytics demo using synthetic employee data. "
    "Not for real employment decisions."
)

DATA_PATH = Path("data/employee_performance.csv")
MODEL_PATH = Path("models/employee_performance_model.joblib")

if not DATA_PATH.exists():
    st.error("Dataset not found. Run: python main.py")
    st.stop()

df = pd.read_csv(DATA_PATH)

tab1, tab2, tab3 = st.tabs([
    "📈 Analytics Dashboard",
    "🔮 Try a Prediction",
    "ℹ️ About This Project"
])

with tab1:
    st.subheader("Synthetic Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Departments", df["department"].nunique())
    col3.metric("Performance Bands", df["perf_band_next"].nunique())

    st.subheader("Performance Band Distribution")
    counts = df["perf_band_next"].value_counts()
    st.bar_chart(counts)

    st.subheader("Average Delivery Rate by Performance Band")
    delivery = df.groupby("perf_band_next")[
        "on_time_delivery_rate"
    ].mean()
    st.bar_chart(delivery)

    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)

        st.subheader("Model Test Results")
        a, b = st.columns(2)
        a.metric(
            "Test Accuracy",
            f"{metrics['test_accuracy']:.3f}"
        )
        b.metric(
            "Test Macro-F1",
            f"{metrics['test_macro_f1']:.3f}"
        )
        st.caption(
            f"Selected model: {metrics['selected_model']}. "
            "These results apply only to the synthetic test dataset."
        )

    for image_path in [
        "outputs/target_distribution.png",
        "outputs/confusion_matrix.png",
        "outputs/feature_importance.png"
    ]:
        if Path(image_path).exists():
            st.image(image_path, use_container_width=True)

with tab2:
    st.subheader("Try a Synthetic Profile")

    if not MODEL_PATH.exists():
        st.warning("Model not trained yet. Run: python main.py")
    else:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                age = st.number_input("Age", 18, 65, 29)
                experience = st.number_input(
                    "Experience (years)", 0, 40, 5
                )
                department = st.selectbox(
                    "Department",
                    ["IT", "HR", "Finance", "Sales", "Operations"]
                )
                job_level = st.selectbox(
                    "Job Level", ["Junior", "Mid", "Senior"]
                )
                education = st.selectbox(
                    "Education", ["Bachelor", "Master", "PhD"]
                )
                manager_tenure = st.number_input(
                    "Manager tenure (years)", 0, 30, 3
                )

            with c2:
                projects = st.number_input(
                    "Project count", 0, 30, 4
                )
                delays = st.number_input(
                    "Task delays", 0, 30, 2
                )
                delivery = st.slider(
                    "On-time delivery rate", 0.0, 100.0, 88.0
                )
                quality = st.slider(
                    "Quality score", 0.0, 100.0, 82.0
                )
                customer = st.slider(
                    "Customer score", 0.0, 100.0, 80.0
                )
                peer = st.slider(
                    "Peer score", 0.0, 100.0, 85.0
                )
                previous = st.slider(
                    "Previous performance score",
                    0.0, 100.0, 80.0
                )

            with c3:
                training = st.number_input(
                    "Training hours", 0, 200, 30
                )
                certifications = st.number_input(
                    "Certifications", 0, 20, 2
                )
                sick_days = st.number_input(
                    "Sick days", 0, 100, 3
                )
                absences = st.number_input(
                    "Unplanned absences", 0, 50, 1
                )
                workload = st.number_input(
                    "Workload hours/week", 1, 100, 40
                )
                salary_band = st.selectbox(
                    "Salary band", ["Low", "Medium", "High"]
                )

            submitted = st.form_submit_button("Predict Demo Band")

        if submitted:
            profile = {
                "age": age,
                "experience_years": experience,
                "department": department,
                "job_level": job_level,
                "education": education,
                "manager_tenure_years": manager_tenure,
                "project_count": projects,
                "task_delays": delays,
                "on_time_delivery_rate": delivery,
                "quality_score": quality,
                "customer_score": customer,
                "peer_score": peer,
                "training_hours": training,
                "certifications": certifications,
                "sick_days": sick_days,
                "unplanned_absences": absences,
                "workload_hours": workload,
                "previous_performance_score": previous,
                "salary_band": salary_band
            }

            model = joblib.load(MODEL_PATH)
            input_df = pd.DataFrame([profile])
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            st.success(f"Demo prediction: {prediction}")

            st.write("Model class probabilities")
            probability_df = pd.DataFrame({
                "Performance Band": model.classes_,
                "Probability": probabilities
            }).set_index("Performance Band")

            st.bar_chart(probability_df)

            st.info(
                "This is an educational prediction from synthetic data. "
                "It is not a measurement of a real person's performance "
                "and must not be used for employment decisions."
            )

with tab3:
    st.subheader("Project Information")
    st.write("""
    This project demonstrates a basic machine-learning workflow:

    - Synthetic dataset generation
    - Exploratory data analysis
    - Logistic Regression and Random Forest comparison
    - Model evaluation using macro-F1 and accuracy
    - Streamlit dashboard and demonstration predictions

    **Responsible use:** This project is for learning only. It does not
    establish actual employee performance, and it must not be used for
    hiring, promotion, salary, discipline, or termination decisions.
    """)