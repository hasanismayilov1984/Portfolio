import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import joblib
import os

# Read and clean the raw data
file_id = '1okDrRBvTh9FA6ypZzWhHFJ_50nkkErEf'
download_url = f'https://drive.google.com/uc?id={file_id}'
df_raw = pd.read_csv(download_url, header=None)
df = df_raw[0].str.split('\t', expand=True)
df = df.iloc[1:].reset_index(drop=True)
df.columns = ['date', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d', errors='coerce')
for col in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Extract time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Sort by date
df = df.sort_values(by='date').reset_index(drop=True)

# Cyclical encoding
def cyclical_encode(data):
    data = pd.DataFrame(data).copy()
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    return data.drop(columns=['day', 'month', 'date'], errors='ignore')

cyclical_encoder = FunctionTransformer(cyclical_encode, validate=False)

# Time-based split
total_len = len(df)
train_end = int(0.7 * total_len)
val_end = int(0.85 * total_len)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

X_train, y_train = train_df.drop(columns=['close']), train_df['close']
X_val, y_val = val_df.drop(columns=['close']), val_df['close']
X_test, y_test = test_df.drop(columns=['close']), test_df['close']

# Preprocessing pipeline
pipeline = Pipeline([
    ('cyclical', cyclical_encoder),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
joblib.dump(pipeline, 'pipeline.pkl')

X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)
X_test = pipeline.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(max_iter=10000),
    'ElasticNet': ElasticNet(max_iter=10000),
    'Bayesian Ridge': BayesianRidge(),
    'Huber Regressor': HuberRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'LightGBM': LGBMRegressor(),
    'Support Vector Machine': SVR(),
    'KNN': KNeighborsRegressor(),
    'MLP Regressor': MLPRegressor(max_iter=1000)
}

# Model evaluation
trained_models = {}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{name}: MSE = {mse:.2f}, MAE = {mae:.2f}, R² = {r2:.4f}")
    trained_models[name] = model
    results[name] = {'mse': mse, 'mae': mae, 'r2': r2}

# Visualize evaluation results
results_df = pd.DataFrame(results).T.sort_values(by='r2', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df['r2'])
plt.xticks(rotation=45, ha='right')
plt.title('Model R² on Validation Set')
plt.ylabel('R² Score')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df['mae'])
plt.xticks(rotation=45, ha='right')
plt.title('Model MAE on Validation Set')
plt.ylabel('Mean Absolute Error')
plt.tight_layout()
plt.show()

# TimeSeriesSplit CV
scorer = make_scorer(mean_squared_error, greater_is_better=False)
tscv = TimeSeriesSplit(n_splits=5)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, scoring=scorer, cv=tscv, n_jobs=-1)
    scores = -scores
    print(f"{name} CV: Mean CV MSE = {scores.mean():.2f}, Std = {scores.std():.2f}")

# Hyperparameter tuning (optional)
tuning_param_grids = {
    'Ridge Regression': {'alpha': [0.01, 0.1, 1, 10, 100]},
    'Lasso Regression': {'alpha': [0.01, 0.1, 1, 10]},
    'ElasticNet': {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
    'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]},
    'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [31, 50]},
    'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1], 'kernel': ['linear', 'rbf']},
    'MLP Regressor': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
}

models_to_tune = ['Ridge Regression', 'Lasso Regression', 'LightGBM']

print("\n\n🔧 Hyperparameter Tuning Results")
for name in models_to_tune:
    if name in tuning_param_grids:
        model = models[name]
        param_grid = tuning_param_grids[name]
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        print(f"{name} Best Params: {grid_search.best_params_}")
        print(f"  Validation MAE: {mean_absolute_error(y_val, y_pred):.2f}, R²: {r2_score(y_val, y_pred):.4f}\n")
    else:
        print(f"No param grid defined for {name}")
# 🏆 Save the best model (Linear Regression) trained on full train + validation data
X_full = pd.concat([train_df.drop(columns=['close']), val_df.drop(columns=['close'])])
y_full = pd.concat([train_df['close'], val_df['close']])

X_full_transformed = pipeline.fit_transform(X_full)  # 👈 this both fits and transforms

# Re-train the best model
final_model = LinearRegression()
final_model.fit(X_full_transformed, y_full)

# Save the model and the fitted pipeline
home_path = os.path.expanduser("~")
model_path = os.path.join(home_path, 'linear_regression_gold_model.pkl')
pipeline_path = os.path.join(home_path, 'pipeline.pkl')

joblib.dump(final_model, model_path)
joblib.dump(pipeline, pipeline_path)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Pipeline saved to: {pipeline_path}")

