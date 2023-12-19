from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

df= pd.read_csv('../Dataset/acs_data2_cleaned.csv')
scaler = joblib.load('../Dataset/scaler.pkl')

X = df.drop(['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min','ad_volume', 'pr_volume'], axis=1)
y = df[['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

def tune_hyperparameters(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    multioutput_model = MultiOutputRegressor(model)

    param_grid = {
        'estimator__n_estimators': [50, 100, 150, 200, 250, 300],
        'estimator__max_depth': [None, 5, 10, 20, 30, 40, 50],
        'estimator__min_samples_split': [2, 5, 10, 15, 20],
        'estimator__min_samples_leaf': [1, 2, 4, 6, 8],
        'estimator__bootstrap': [True, False],
        'estimator__max_features': ['sqrt', 'log2', None],
    }

    grid_search = GridSearchCV(multioutput_model, param_grid, cv=5, verbose=2, n_jobs=3)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    return grid_search.best_estimator_


# tune hyperparameters
best_model = tune_hyperparameters(X_train, y_train)

joblib.dump(best_model, './Models/random_forest_model_gridsearch_optimized.pkl')


