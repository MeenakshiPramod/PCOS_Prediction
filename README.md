# PCOS Prediction using Machine Learning

This project aims to develop a machine learning model that predicts the likelihood of Polycystic Ovary Syndrome (PCOS) in individuals based on various clinical and diagnostic parameters. The goal is to support early diagnosis and medical intervention.

## 📚 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Future Work](#future-work)
- [License](#license)

---

## 🧠 About the Project

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder that affects many women of reproductive age. Early detection is key to managing symptoms and preventing complications. This project uses a dataset of clinical and ultrasound data to train several machine learning models and identify the best-performing one for PCOS prediction.

---

## 📂 Dataset

- **Source**: Kaggle (or any dataset link used)
- **Size**: 541 instances with 44 features (clinical, hormonal, and diagnostic parameters)
- **Target Variable**: `PCOS (Y/N)`

Key features include:
- Age (yrs), Weight (Kg), Height(Cm), BMI, Pulse rate (bpm), RR (breaths/min)
- Blood pressure, FSH, LH, PRL, TSH, AMH, Vit D3, PRG
- Cycle(R/I), Cycle length (days), Hair growth(Y/N), Skin darkening (Y/N)
- Fast food (Y/N), Pimples(Y/N), Weight gain(Y/N), Hair loss(Y/N)
- ...and more.

---

## 🛠 Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

---

## ⚙️ Installation

To get this project up and running locally, follow these steps:

```bash
# Clone the repo
git clone https://github.com/your-username/pcos-prediction.git
cd pcos-prediction

# Create a virtual environment (optional)
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the file `PCOS_Prediction.ipynb` and run all cells sequentially.

---

## 🔄 Project Workflow

1. **Data Preprocessing**:
    - Handling missing values
    - Encoding categorical variables
    - Feature selection

2. **Exploratory Data Analysis (EDA)**:
    - Distribution plots, heatmaps, correlation analysis

3. **Model Training**:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Decision Tree

4. **Model Evaluation**:
    - Accuracy
    - Confusion Matrix
    - Precision, Recall, F1-score
    - ROC Curve and AUC

---


## 🌱 Future Work

- Hyperparameter tuning using GridSearchCV
- Deployment using Streamlit or Flask
- Integration with medical dashboards
- Real-time prediction using web interface

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 💬 Acknowledgements

- The dataset was obtained from [Kaggle](https://kaggle.com/)
- Thanks to open-source tools and libraries used in the development

---

> 💡 **Note**: This model is not a replacement for professional medical diagnosis. Always consult a healthcare provider for medical advice.
