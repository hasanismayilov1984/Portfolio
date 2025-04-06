# 🧠 Hasan Ismayilov – Data Science Portfolio

Welcome to my data science portfolio!  
Here you'll find hands-on projects demonstrating my skills in machine learning, data preprocessing, time series forecasting, and model evaluation using Python.

---

## 📂 Projects

---

### 📌 Multi-label Text Classification (Scikit-learn)

This project classifies scientific paper abstracts and titles into multiple relevant research categories, such as Computer Science, Mathematics, and Physics.

🧠 **Key Features:**
- Combined text features from `TITLE` and `ABSTRACT`
- Applied **TF-IDF** vectorization with max 10,000 features
- Trained a **OneVsRest Logistic Regression** model for multi-label prediction
- Used **custom probability thresholds** to fine-tune label sensitivity
- Output predictions on a separate test dataset

🔧 **Tools & Libraries:**  
`Pandas`, `Scikit-learn`, `NumPy`, `Matplotlib`, `Seaborn`

📁 **Project Files:**  
[`Test.py`](./Test.py), [`test_predictions.csv`](./test_predictions.csv)
### 🧠 Multi-label Text Classification  
Classifies scientific paper abstracts into multiple topics using TF-IDF and Logistic Regression.  
📘 [View Full Project on Kaggle](https://www.kaggle.com/code/gnkanalytics/multi-output-classification-text-precessing-sci)

---

### 📊 Gold Price Prediction (Scikit-learn)

This project predicts gold closing prices using historical market data. It covers full-cycle modeling: data cleaning, cyclical feature engineering, model evaluation, hyperparameter tuning, and saving the final model pipeline.

🧠 **Key Features:**
- Reads and cleans raw gold price data (from Google Drive)
- Extracts & encodes time features using sine/cosine cycles
- Preprocessing pipeline: Imputation, scaling, feature encoding
- Trains & compares 14+ models (Linear Regression, Ridge, SVR, LightGBM, etc.)
- Evaluates with MSE, MAE, R², and cross-validation
- Hyperparameter tuning using `GridSearchCV`
- Saves final model + pipeline with `joblib`

🔧 **Tools & Libraries:**  
`Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `LightGBM`, `Joblib`

📁 **Project Files:**  
[`Gold Prediction Scikit-Learn.py`](./Gold%20Prediction%20Scikit-Learn.py), [`pipeline.pkl`](./pipeline.pkl)

Predicts gold closing prices using time-based features and 14+ regression models.  
📘 [View Full Project on Kaggle](https://www.kaggle.com/code/gnkanalytics/gold-stock-closing-prediction)

---

## 🔜 More Projects Coming Soon...
Stay tuned as I continue exploring topics like NLP, computer vision, and deep learning.
