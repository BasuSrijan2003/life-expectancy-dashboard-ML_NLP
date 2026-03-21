# 🌍 AI-Powered Life Expectancy Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)](https://pandas.pydata.org/)

An end-to-end Machine Learning pipeline and interactive web application that predicts the average life expectancy of a country based on demographic, economic, and health-related factors.

**🔴 Live Demo:** https://life-expectancy-prediction-2026.streamlit.app/



**🌐 Contact Owner:** https://srijanbasu.netlify.app/

---

## 📌 Project Overview

This project goes beyond simply training a model—it demonstrates a complete **data science lifecycle**.

Using data from the **Global Health Observatory (GHO)** under the **World Health Organization (WHO)**, the model predicts life expectancy while also providing **Explainable AI (XAI)** insights to understand *why* predictions are made.

### 🔍 Key Highlights

* 📊 **Feature Importance:** Identifies top factors affecting life expectancy
  *(e.g., HIV/AIDS prevalence, Years of Schooling)*
* 🎯 **High Accuracy:** Achieved **R² Score of 96.89%**
* ⚡ **Real-Time Predictions:** Interactive Streamlit web app
* 🧠 **Explainable AI:** Model transparency using feature ranking

---

## 🏗️ Model Selection & Benchmarking

Models were evaluated using **Grid Search** and **K-Fold Cross Validation**.

| Model                       | R² Score   | Status      |
| --------------------------- | ---------- | ----------- |
| Decision Tree Regressor     | 90.12%     | Baseline    |
| Gradient Boosting Regressor | 96.71%     | Challenger  |
| Random Forest (Tuned)       | **96.89%** | 🏆 Champion |

---

## 📂 Repository Structure

```
📁 Life-Expectancy-Project
│
├── 📁 data/
│   └── Life Expectancy Data.csv
│
├── 📁 notebooks/
│   ├── LifeExpectancyIrfan.ipynb
│   └── model_selection.ipynb
│
├── 📁 model/
│   ├── rf_model.pkl
│   └── feature_names.json
│
├── app.py
├── train_model.py
└── requirements.txt
```

---

## 🚀 Installation & Local Setup

Follow these steps to run the project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/life-expectancy-predictor.git
cd life-expectancy-predictor
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧠 Tech Stack

* Python 3.12
* Scikit-learn
* Pandas
* Streamlit
* NumPy

---

## 👨‍💻 Author

**Srijan Basu**
Master of Computer Applications (MCA)

📧 [2003srijanbasu@gmail.com](mailto:2003srijanbasu@gmail.com)

🔗 https://srijanbasu.netlify.app/

---

## 💡 Future Improvements

* Add more advanced XAI (SHAP values)
* Deploy on AWS
* Improve UI/UX
* Add country-wise visualization dashboards

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
