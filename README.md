# 📊 Employee Performance Predictor using Data Analytics

**Developed by: ER.CHAND KUJRA**

An AI-powered Employee Performance Analytics project built using Python, Machine Learning, Pandas, Scikit-learn and Streamlit.

## 🚀 Project Overview

The Employee Performance Predictor is an educational data analytics and machine learning project designed to explore synthetic employee productivity data and demonstrate performance classification.

The project generates synthetic employee records, performs exploratory data analysis, trains machine learning models, evaluates their performance and displays analytical insights through an interactive Streamlit dashboard.

## 🎯 Project Objectives

- Analyze employee productivity indicators.
- Explore synthetic employee performance data.
- Build machine learning classification models.
- Compare Logistic Regression and Random Forest.
- Visualize performance trends and analytics.
- Demonstrate model predictions through a dashboard.

## ✨ Key Features

- Synthetic employee dataset generation.
- Exploratory Data Analysis (EDA).
- Employee performance classification.
- Logistic Regression model.
- Random Forest model.
- Model comparison using Macro F1-score.
- Confusion matrix visualization.
- Feature importance analysis.
- Interactive Streamlit dashboard.
- Performance prediction demonstration.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data Analysis |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Streamlit | Interactive Dashboard |
| Joblib | Model Saving |

## 🎨 Dashboard Design

**Theme:** Dark Navy Blue

- Primary Background: `#14233F`
- Card Background: `#1B2D4A`
- Orange Accent: `#D9822B`
- Golden Yellow: `#FFB900`
- Text Color: `#FFFFFF`

## 📁 Project Structure

```text
Employee-Performance-Predictor/
│
├── data/
│   └── employee_performance.csv
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── eda.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── employee_performance_model.joblib
│
├── outputs/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── target_distribution.png
│   └── delivery_by_band.png
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/Employee-Performance-Predictor.git
```

### Step 2: Open Project Folder

```bash
cd Employee-Performance-Predictor
```

### Step 3: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate Environment (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run Complete Project

```bash
python main.py
```

### Step 7: Launch Dashboard

```bash
streamlit run app.py
```

Open in your browser:

```text
http://localhost:8501
```

## 🤖 Machine Learning Workflow

1. Generate synthetic employee data.
2. Perform data preprocessing.
3. Split data into training, validation and testing sets.
4. Train Logistic Regression and Random Forest models.
5. Select the model using validation Macro F1-score.
6. Evaluate the selected model on the held-out test set.
7. Save the trained model.
8. Display results through Streamlit.

## 📊 Evaluation Metrics

- Accuracy
- Macro F1-score
- Precision
- Recall
- Classification Report
- Confusion Matrix
- Permutation Feature Importance

Actual model results are generated when the project is executed. No performance score is claimed in advance.

## 📸 Project Screenshots

Add your dashboard screenshots in the `images/` folder and include them here:

```markdown
![Dashboard](images/dashboard.png)
![Model Results](images/model_results.png)
![Prediction Demo](images/prediction_demo.png)
```

## 🎓 Learning Outcomes

Through this project, I explored:

- Python programming.
- Data analytics and visualization.
- Machine learning classification.
- Model evaluation and comparison.
- Interactive dashboard development.
- Practical project organization.
- Git and GitHub workflow.

## ⚠️ Responsible AI & Disclaimer

This project uses synthetic data generated for educational and demonstration purposes only.

Predictions are not evidence of actual employee performance. This system must not be used for real-world hiring, promotion, salary, disciplinary or termination decisions.

## 👨‍💻 Developer

