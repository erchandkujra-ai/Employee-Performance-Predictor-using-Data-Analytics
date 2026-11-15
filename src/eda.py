from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(data_path="data/employee_performance.csv"):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    # Target distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="perf_band_next")
    plt.title("Synthetic Performance Band Distribution")
    plt.xlabel("Performance Band")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png")
    plt.close()

    # Delivery rate by performance band
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="perf_band_next", y="on_time_delivery_rate")
    plt.title("On-Time Delivery Rate by Synthetic Performance Band")
    plt.tight_layout()
    plt.savefig(output_dir / "delivery_by_band.png")
    plt.close()

    # Numeric correlation heatmap
    numeric_df = df.select_dtypes(include="number").drop(
        columns=["employee_id"], errors="ignore"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    print("EDA completed. Charts saved in outputs/.")


if __name__ == "__main__":
    run_eda()