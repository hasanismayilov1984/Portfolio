# 🧠 Student Grade Classification with Scikit-Learn Pipelines & Hyperparameter Tuning

# 📦 1. Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, r2_score, confusion_matrix
import joblib

df = pd.read_csv(r"C:\Users\HasanIsmayilov\OneDrive - 360Incentives.com Canada ULC\Desktop\Students_Grading_Dataset.csv")

# 📊 2. Feature Correlation Analysis
df_analytic = df.copy()
corr_matrix = df_analytic.select_dtypes(include = ['number']).corr()
plt.figure(figsize = (8,6))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 📈 3. Feature Distributions
plt.rc('font', size=12)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
df_analytic.hist(figsize=(16,10))
plt.suptitle('Column Distributions')
plt.tight_layout()
plt.show()

# 🧮 4. Grade Distribution
df_analytic['Grade'].value_counts().sort_index().plot.bar(rot = 0, grid = True)
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.title("Distribution of Target Classes (Grade)")
plt.show()

# 🧼 5. Data Cleaning & Preprocessing Pipeline
df.drop(columns=['Student_ID', 'First_Name', 'Last_Name', 'Email'], inplace=True)

X = df.drop(columns=['Grade'])
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(df['Grade'])

num_features = X.select_dtypes(include=['number']).columns.tolist()
cat_features = X.select_dtypes(exclude=['number']).columns.tolist()

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# 📂 6. Train-Test Split & Final Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train)
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

X_train_processed = pd.DataFrame(X_train_transformed, columns=pipeline.get_feature_names_out())
X_test_processed = pd.DataFrame(X_test_transformed, columns=pipeline.get_feature_names_out())

print("Preprocessing complete. Transformed data is ready for model training.")

# 🤖 7. Model Training & Evaluation
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = []

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    
    accuracy = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, accuracy, r2))
    
    print(f"\n{name} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 📊 8. Visualizing Model Accuracy Comparison
result_df = pd.DataFrame(results, columns = ['name', 'accuracy', 'r2'])
plt.figure(figsize = (12,6))
sns.barplot(x = 'name', y = 'accuracy', data = result_df, palette = 'Reds')
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()

# 🧪 9. Hyperparameter Tuning & Final Model Evaluation
param_grid = {
    'n_estimators': [100, 300, 500], 
    'max_depth': [10, 30, None],  
    'min_samples_split': [2, 5], 
    'min_samples_leaf': [1, 3],  
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv = 5,
    scoring='accuracy', 
    verbose = 2,
    n_jobs = 1
)

grid_search.fit(X_train_processed, y_train)
best_params = grid_search.best_params_
print('Best Hyperparameter:', best_params)

best_model = RandomForestClassifier(
    max_depth=10, 
    max_features='log2', 
    min_samples_leaf=3, 
    min_samples_split=2, 
    n_estimators=500, 
    random_state=42
)

best_model.fit(X_train_processed, y_train)
y_pred = best_model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nFinal Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"R2 Score: {r2:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 💾 10. Saving the Final Model & Preprocessor
joblib.dump(best_model, 'student_grading_model.pkl')
joblib.dump(pipeline, 'preprocessor.pkl')
print('Model and preprocessor saved successfully.')