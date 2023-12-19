from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import numpy as np

df = pd.read_csv('../Dataset/oos_testing_data.csv')

X = df.drop(['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent',
             'cure_cycle_total_time_min', 'ad_volume', 'pr_volume'], axis=1)
y = df[['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent',
        'cure_cycle_total_time_min']]


best_model=joblib.load('./Models/random_forest_model_bayesian_optimized.pkl')

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    # Create a DataFrame for the input parameters
    df_input = pd.DataFrame(X_test, columns=X_test.columns)

    df_original = y_test.reset_index(drop=True)

    df_pred = pd.DataFrame(y_pred, columns=[col for col in y_test.columns])

    # Calculate the percentage difference
    percent_diff = ((df_pred.values - df_original.values) / np.where(df_original.values != 0, df_original.values, 1e-7)) * 100

    percent_diff = pd.DataFrame(percent_diff, columns=[col + '_percent_diff' for col in y_test.columns])

    # Concatenate the input parameters, original values, predicted values and percentage differences into one DataFrame
    df_result = pd.concat([df_input, df_original, df_pred, percent_diff], axis=1)

    return df_result



# Evaluate the model and get the result dataframe
df_result = evaluate_model(best_model, X, y)
df_result.to_csv('../Dataset/outofsample_predictions.csv', index=False)
