# 📦 Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 📂 Load Data
train = pd.read_csv(r"C:\Users\HasanIsmayilov\OneDrive - 360Incentives.com Canada ULC\Desktop\Multilabel classification\train.csv")
test = pd.read_csv(r"C:\Users\HasanIsmayilov\OneDrive - 360Incentives.com Canada ULC\Desktop\Multilabel classification\test.csv")

# 📝 Combine TITLE and ABSTRACT
train['text'] = train['TITLE'].fillna('') + ' ' + train['ABSTRACT'].fillna('')
test['text'] = test['TITLE'].fillna('') + ' ' + test['ABSTRACT'].fillna('')

# 🔡 TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train = vectorizer.fit_transform(train['text'])
X_test = vectorizer.transform(test['text'])

# 🎯 Labels
label_cols = ['Computer Science', 'Physics', 'Mathematics',
              'Statistics', 'Quantitative Biology', 'Quantitative Finance']
y_train = train[label_cols]

# 🧪 Train/Validation Split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# 🤖 Train OneVsRestClassifier
model = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000))
model.fit(X_train_split, y_train_split)

# 📈 Predict on Validation Set with Probabilities
y_val_proba = model.predict_proba(X_val)

# 🎚️ Custom Thresholds
thresholds = {
    'Computer Science': 0.5,
    'Physics': 0.5,
    'Mathematics': 0.5,
    'Statistics': 0.5,
    'Quantitative Biology': 0.3,
    'Quantitative Finance': 0.25
}
label_indices = {label: idx for idx, label in enumerate(label_cols)}

# 🧮 Apply Thresholds on Validation Set
y_val_pred_thresh = np.zeros_like(y_val, dtype=int)
for label, threshold in thresholds.items():
    idx = label_indices[label]
    y_val_pred_thresh[:, idx] = (y_val_proba[:, idx] >= threshold).astype(int)

# 📊 Evaluate on Validation Set
print(classification_report(y_val, y_val_pred_thresh, target_names=label_cols))

# 🚀 Predict on Test Set
y_test_proba = model.predict_proba(X_test)
proba_matrix = np.column_stack(y_test_proba)
y_test_pred = np.zeros_like(proba_matrix, dtype=int)
for label, threshold in thresholds.items():
    idx = label_indices[label]
    y_test_pred[:, idx] = (proba_matrix[:, idx] >= threshold).astype(int)

# 🔧 Fix Shape (if needed)
if y_test_pred.shape[0] == len(label_cols):
    y_test_pred = y_test_pred.T

# 🧾 Add Predictions to Test DataFrame
for idx, label in enumerate(label_cols):
    test[label] = y_test_pred[:, idx]

# 💾 Save Predictions
test[['TITLE', 'ABSTRACT'] + label_cols].to_csv('test_predictions.csv', index=False)