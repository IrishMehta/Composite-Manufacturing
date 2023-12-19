import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay


df= pd.read_csv('../Dataset/acs_data2_cleaned.csv')
scaler = joblib.load('../Dataset/scaler.pkl')

X = df.drop(['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min','ad_volume', 'pr_volume'], axis=1)
y = df[['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

best_model=joblib.load('./Models/random_forest_model_bayesian_optimized.pkl')

# Feature importance
for i in range(len(y.columns)):
    print(f"Feature importances for output variable {y.columns[i]}:")
    importance = best_model.estimators_[i].feature_importances_
    print(f"Feature importance of {y_train.columns[i]} = {importance}")
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importance, y=X.columns, palette="viridis")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importances for {y.columns[i]}')
    plt.savefig(f'./Charts/feature_importances_{y.columns[i]}.png')

# Partial Dependence Plots
features = [i for i in range(X.shape[1])]
n_cols = 3  # You can adjust this value to change the number of columns in the grid
n_rows = int(np.ceil(len(features) / n_cols))

for i in range(len(y.columns)):
    print(f"Partial dependence plots for output variable {y.columns[i]}:")

    # Adjust the size & Flatten the array of axes to make it easier to iterate over
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*5))
    axs = axs.flatten()

    for j, feature in enumerate(features):
        display = PartialDependenceDisplay.from_estimator(best_model, X_train, [feature], target=i, ax=axs[j])
        axs[j].set_title(f'Feature {X.columns[feature]}')  # Add a title to each subplot

    # Remove unused subplots
    for j in range(len(features), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f'Partial Dependence Plots for {y.columns[i]}', fontsize=16)  # Add a title to the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make room for the figure title
    plt.savefig(f'./Charts/partial_dependence_plots_{y.columns[i]}.png')
