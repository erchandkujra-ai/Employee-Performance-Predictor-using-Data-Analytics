import joblib
import pandas as pd


MODEL_PATH = "models/employee_performance_model.joblib"


def predict_employee(profile):
    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([profile])
    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    probability_dict = {
        str(label): round(float(probability), 3)
        for label, probability in zip(classes, probabilities)
    }

    return prediction, probability_dict


if __name__ == "__main__":
    sample_employee = {
        "age": 29,
        "experience_years": 5,
        "department": "IT",
        "job_level": "Mid",
        "education": "Bachelor",
        "manager_tenure_years": 3,
        "project_count": 4,
        "task_delays": 2,
        "on_time_delivery_rate": 88.0,
        "quality_score": 82.0,
        "customer_score": 80.0,
        "peer_score": 85.0,
        "training_hours": 30,
        "certifications": 2,
        "sick_days": 3,
        "unplanned_absences": 1,
        "workload_hours": 40,
        "previous_performance_score": 80.0,
        "salary_band": "Medium"
    }

    label, probabilities = predict_employee(sample_employee)
    print("Predicted synthetic performance band:", label)
    print("Class probabilities:", probabilities)