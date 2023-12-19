import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
import joblib

df= pd.read_csv('../Dataset/acs_data2_cleaned.csv')
scaler = joblib.load('../Dataset/scaler.pkl')

X = df.drop(['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min','ad_volume', 'pr_volume'], axis=1)
y = df[['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


def objective(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42
    )
    multioutput_model = MultiOutputRegressor(model)

    return -np.mean(cross_val_score(multioutput_model, X, y, cv=5, scoring='neg_mean_squared_error'))

def tune_hyperparameters(X_train, y_train, n_trials=75):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    trial = study.best_trial

    print(f"Best parameters: {trial.params}")
    print(f"Best score: {-trial.value}")

    best_model = RandomForestRegressor(
        n_estimators=trial.params['n_estimators'],
        max_depth=trial.params['max_depth'],
        min_samples_split=trial.params['min_samples_split'],
        min_samples_leaf=trial.params['min_samples_leaf'],
        max_features=trial.params['max_features'],
        bootstrap=trial.params['bootstrap'],
        random_state=42
    )
    multioutput_best_model = MultiOutputRegressor(best_model)

    return multioutput_best_model


# tune hyperparameters
best_model = tune_hyperparameters(X_train, y_train)
best_model.fit(X_train, y_train)

joblib.dump(best_model, './Models/random_forest_model_bayesian_optimized.pkl')