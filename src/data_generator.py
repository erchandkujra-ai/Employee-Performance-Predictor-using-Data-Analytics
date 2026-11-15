from pathlib import Path
import numpy as np
import pandas as pd


def generate_data(rows=1200, random_state=42):
    rng = np.random.default_rng(random_state)

    departments = ["IT", "HR", "Finance", "Sales", "Operations"]
    job_levels = ["Junior", "Mid", "Senior"]
    education_levels = ["Bachelor", "Master", "PhD"]
    salary_bands = ["Low", "Medium", "High"]

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, rows + 1)],
        "age": rng.integers(21, 60, rows),
        "experience_years": rng.integers(0, 25, rows),
        "department": rng.choice(departments, rows),
        "job_level": rng.choice(job_levels, rows),
        "education": rng.choice(education_levels, rows),
        "manager_tenure_years": rng.integers(0, 15, rows),
        "project_count": rng.integers(1, 10, rows),
        "task_delays": rng.integers(0, 12, rows),
        "on_time_delivery_rate": rng.uniform(45, 100, rows).round(1),
        "quality_score": rng.uniform(40, 100, rows).round(1),
        "customer_score": rng.uniform(40, 100, rows).round(1),
        "peer_score": rng.uniform(40, 100, rows).round(1),
        "training_hours": rng.integers(0, 80, rows),
        "certifications": rng.integers(0, 6, rows),
        "sick_days": rng.integers(0, 15, rows),
        "unplanned_absences": rng.integers(0, 8, rows),
        "workload_hours": rng.integers(25, 60, rows),
        "previous_performance_score": rng.uniform(40, 100, rows).round(1),
        "salary_band": rng.choice(salary_bands, rows),
        "gender_audit_only": rng.choice(["Group_A", "Group_B"], rows)
    })

    # Transparent synthetic scoring formula for educational use only.
    score = (
        0.25 * df["on_time_delivery_rate"]
        + 0.20 * df["quality_score"]
        + 0.15 * df["customer_score"]
        + 0.10 * df["peer_score"]
        + 0.15 * df["previous_performance_score"]
        + 0.10 * np.minimum(df["training_hours"], 60)
        - 1.2 * df["task_delays"]
        - 1.5 * df["unplanned_absences"]
        + rng.normal(0, 10, rows)
    )

    df["perf_band_next"] = np.select(
        [score >= 65, score >= 45],
        ["High", "Medium"],
        default="Low"
    )

    return df


if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    data = generate_data()
    data.to_csv(output_dir / "employee_performance.csv", index=False)

    print("Synthetic dataset created successfully!")
    print(f"Rows: {len(data)}")
    print(data.head())