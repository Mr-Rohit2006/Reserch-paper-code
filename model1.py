# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# STEP 2: Create DataFrame manually (based on your image)
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October'],
    '2019': [11.11, 10.91, 9.78, 7.75, 6.15, 7.26, 8.18, 8.01, 7.52, 9.45],
    '2022': [2.16, 2.55, 3.58, 4.11, 4.46, 5.47, 6.69, 5.25, 5.52, 6.77],
    '2023': [8.91, 8.93, 8.25, 6.26, 6.18, 6.68, 7.86, 6.64, 6.67, 8.32],
    '2024': [9.59, 10.03, 8.60, 6.51, 6.00, 7.06, 7.76, 6.36, 6.69, 8.20]
}

df = pd.DataFrame(data)

# STEP 3: Prepare input features (X) and target (y)
X = df[['2019', '2022', '2023']]
y = df['2024']

# STEP 4: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Initialize models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor()
}

# STEP 6: Fit, predict, evaluate
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results[name] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(predictions, label='Predicted', marker='x')
    plt.title(f'{name} - Actual vs Predicted')
    plt.xlabel('Test Sample Index')
    plt.ylabel('2024 FTAs (in Lakhs)')
    plt.legend()
    plt.grid(True)
    plt.show()

# STEP 7: Show Evaluation Results
results_df = pd.DataFrame(results).T
print("Model Evaluation Metrics:\n")
print(results_df)
