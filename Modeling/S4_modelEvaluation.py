from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

df= pd.read_csv('../Dataset/acs_data2_cleaned.csv')
scaler = joblib.load('../Dataset/scaler.pkl')

X = df.drop(['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min','ad_volume', 'pr_volume'], axis=1)
y = df[['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

best_model_gridsearch=joblib.load('./Models/random_forest_model_gridsearch_optimized.pkl')
best_model_bayesian=joblib.load('./Models/random_forest_model_bayesian_optimized.pkl')

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    for i, col in enumerate(y_test.columns):
        mse_column = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        print(f'MSE for {col}: {mse_column}')

    for i, col in enumerate(y_test.columns):
        r2_column = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f'R2 for {col}: {r2_column}')

def plot_feature_importance(model, X_train, ax, title):
    estimators = [estimator for estimator in model.estimators_]
    importances = [e.feature_importances_ for e in estimators]
    average_importances = np.mean(importances, axis=0)
    feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': average_importances})
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    sns.barplot(x='importance', y='feature', data=feature_importances, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')

print("Evaluating GridSearch Optimized Model:")
evaluate_model(best_model_gridsearch, X_test, y_test)
print("Evaluating Bayesian Optimized Model:")
evaluate_model(best_model_bayesian, X_test, y_test)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

plot_feature_importance(best_model_gridsearch, X_train, axes[0], "Feature Importance for GridSearch Optimized Model")
plot_feature_importance(best_model_bayesian, X_train, axes[1], "Feature Importance for Bayesian Optimized Model")

plt.tight_layout()
plt.savefig("./Charts/feature_importance.png")
plt.show()
