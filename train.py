import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
import joblib
from sklearn.tree import DecisionTreeRegressor


def train(model, hyper_params):

    df = pd.read_csv("data/train.csv")

    features = [
        "rainfall",
        "mean_temperature",
        "disease_cases_ma_3",
        "rainfall_ma_3",
        "mean_temperature_ma_3",
        "disease_cases_lag",
    ]
    target = ["disease_cases"]

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mse_scores = []
    r2_scores = []
    splits = 5
    poly = None

    match model:
        case "ridge":
            poly = PolynomialFeatures(degree=hyper_params["poly_degree"])
            X_scaled = poly.fit_transform(X_scaled)
            model = Ridge(alpha=hyper_params["lmb"])
        case "dt":
            model = DecisionTreeRegressor(
                max_depth=hyper_params["max_depth"],
                min_samples_split=hyper_params["min_samples_split"],
            )
        case "rf":
            model = RandomForestRegressor(
                max_depth=hyper_params["max_depth"],
                min_samples_split=hyper_params["min_samples_split"],
                n_estimators=hyper_params["num_trees"],
            )
        case "bagging":
            model = BaggingRegressor(
                n_estimators=hyper_params["num_trees"],
                max_samples=hyper_params["max_samples"],
                max_features=hyper_params["max_samples"],
            )

    # Time series cross validation
    for fold in range(1, splits + 1):
        # Define train/test split sizes for expanding window
        train_size = int(len(X_scaled) * fold / splits)

        test_size = len(X_scaled) - train_size
        if test_size < 1:
            break

        # Train on expanding window
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        print(
            f"Fold {fold}: Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)
    return model, np.mean(mse_scores), np.mean(r2_scores), scaler, poly
