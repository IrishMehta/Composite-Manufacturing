from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib

df= pd.read_csv('../Dataset/acs_data2_cleaned.csv')

X = df.drop(['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min','ad_volume', 'pr_volume'], axis=1)
y = df[['ad_porosity_percent', 'pr_porosity_percent', 'eff_porosity_percent', 'max_fiber_volume_fraction_percent', 'cure_cycle_total_time_min']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

def evaluate_model(model, X_train, y_train, name):
    if name not in ['Random Forest', 'XGBoost']:
        scaler = StandardScaler()
        joblib.dump(scaler, '../Dataset/scaler.pkl')
        X_train = scaler.fit_transform(X_train)


    multioutput_model = MultiOutputRegressor(model)
    cv = cross_val_score(multioutput_model, X_train, y_train, cv=5)
    return cv.mean()

# Trying the default versions of all types of regression models that support multi-output regression
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": Pipeline([
        ('poly_features', PolynomialFeatures(degree=2)),
        ('linear_regression', LinearRegression())
    ]),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor()
}

for name, model in models.items():
    print(f'Evaluating {name}...')
    score = evaluate_model(model, X_train, y_train, name)
    print(f'{name} Score: {score}\n')

