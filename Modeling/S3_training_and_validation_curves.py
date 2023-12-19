import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

df= pd.read_csv('../Dataset/acs_data2_cleaned.csv')
scaler = joblib.load('../Dataset/scaler.pkl')

X = df.drop(['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min','ad_volume', 'pr_volume'], axis=1)
y = df[['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


def plot_learning_curves(X_train, y_train):

    best_estimator = joblib.load('./Models/random_forest_model_gridsearch_optimized.pkl')


    # Define the folder path to save the figures
    folder_path = r".\Charts"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, output_name in enumerate(y_train.columns):

        estimator = best_estimator.estimators_[i]
        X, y = X_train, y_train.iloc[:, i]

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        validation_scores_mean = np.mean(validation_scores, axis=1)
        validation_scores_std = np.std(validation_scores, axis=1)

        plt.figure()
        plt.title(f"Learning Curve - {output_name}")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                         validation_scores_mean + validation_scores_std, alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, validation_scores_mean, 'o-', color="g",
                 label="Validation score")

        plt.legend(loc="best")

        # Save the figure
        figure_path = os.path.join(folder_path, f"learning_curve_{output_name}.png")
        plt.savefig(figure_path)

    return best_estimator


_ = plot_learning_curves(X, y)