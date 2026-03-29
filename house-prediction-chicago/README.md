
# 🏠 House Price Prediction (End-to-End ML Project)

## 📌 Overview

This project predicts house prices using machine learning.
It includes data preprocessing, model comparison, hyperparameter tuning, and deployment using a Streamlit web application.

---

## ⚙️ Features

* Built an end-to-end ML pipeline using `Pipeline` and `ColumnTransformer`
* Compared multiple models:

  * Linear Regression
  * Ridge & Lasso
  * Random Forest
  * HistGradientBoosting
* Performed hyperparameter tuning using GridSearchCV
* Evaluated models using RMSE, MAE, and R²
* Conducted residual and error analysis
* Deployed an interactive web app using Streamlit

---

## 📊 Model Performance

* Best model: **HistGradientBoostingRegressor**
* Improved performance after hyperparameter tuning
* Evaluated using cross-validation

---

## 📉 Error Analysis

* Residual plots show predictions are mostly unbiased
* Slightly higher errors for high-value houses
* Model captures general trends but struggles with extreme values

---

## 🚀 Web App

The project includes a Streamlit app where users can:

* Input house features (location, rooms, income, etc.)
* Get real-time price predictions

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run housepredapp.py
```

---

## 📷 App Preview

(Add screenshot here after uploading image)

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## 📁 Project Structure

```
house-price-prediction/
│
├── house_prediction.ipynb
├── housepredapp.py
├── model.pkl
├── requirements.txt
├── README.md
```

---

## 🎯 Key Learnings

* Built and evaluated multiple ML models
* Used pipelines for clean preprocessing
* Applied hyperparameter tuning to improve performance
* Performed detailed error analysis
* Deployed a machine learning model as a web app

---
<img width="781" height="964" alt="image" src="https://github.com/user-attachments/assets/d50965c1-ea82-4130-b70c-e648029ef678" />

