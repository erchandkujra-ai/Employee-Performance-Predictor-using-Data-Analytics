from src.data_generator import generate_data
from src.eda import run_eda
from src.train_model import train
from pathlib import Path


def main():
    Path("data").mkdir(exist_ok=True)

    print("\nSTEP 1: Generating synthetic dataset...")
    df = generate_data(rows=1200)
    df.to_csv("data/employee_performance.csv", index=False)
    print("Dataset generated successfully.")

    print("\nSTEP 2: Running exploratory data analysis...")
    run_eda()

    print("\nSTEP 3: Training and evaluating models...")
    train()

    print("\nPROJECT PIPELINE COMPLETED!")
    print("Now run: streamlit run app.py")


if __name__ == "__main__":
    main()