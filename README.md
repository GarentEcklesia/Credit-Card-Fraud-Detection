# 💳 Credit Card Fraud Detection

An end-to-end machine learning project to detect fraudulent credit card transactions. This repository includes data preprocessing, advanced feature engineering, model training, and a real-time detection app deployed with Streamlit.

## 🚀 Live Demo

Experience the fraud detection model in action with our interactive web app:

[**➡️ Click here to launch the Streamlit App**](YOUR_STREAMLIT_APP_URL_HERE)

## 💡 Features

* **Real-Time Prediction:** A Streamlit web app to simulate and predict fraud on new transaction data.
* **Advanced Feature Engineering:** Creates new predictive features like `transaction_velocity`, `purchase_anomaly`, and `home_distance_risk`.
* **Imbalance Handling:** Uses **SMOTE** (Synthetic Minority Over-sampling TEchnique) to effectively train the model on the rare fraud class.
* **Robust Preprocessing:** Includes custom scikit-learn transformers for outlier clipping (IQR) and feature creation.
* **Optimized Model:** A fine-tuned Random Forest model provides the best balance of precision and recall.

## ⚙️ Tech Stack

* **Data Handling & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, Imbalanced-learn (for SMOTE)
* **Model Experimentation:** XGBoost, LightGBM
* **Web Framework:** Streamlit
* **Notebook:** Jupyter
* **Deployment Platform:** Streamlit Cloud

## 🧠 Model Details

* **Model:** **Random Forest Classifier** (Selected after comparing with Logistic Regression, LightGBM, and XGBoost).
* **Dataset:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
* **Key Metrics (on Test Set):**
    * **Overall Accuracy:** 98.35%
    * **Fraud Recall: 95.96%** (This means the model successfully identifies ~96% of all actual fraud).
    * **Fraud Precision: 86.63%** (When the model predicts fraud, it is correct 86.6% of the time).

## 🛠️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/GarentEcklesia/Credit-Card-Fraud-Detection.git](https://github.com/GarentEcklesia/Credit-Card-Fraud-Detection.git)
    cd Credit-Card-Fraud-Detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 📬 Contact

Garent Ecklesia - [garentecklesia45678@gmail.com](mailto:garentecklesia45678@gmail.com)

## 📝 License
This project is open-source and free to use for educational and research purposes.
