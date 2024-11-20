import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
import joblib
from sklearn.tree import DecisionTreeRegressor
from utils import generate_subtitle, plot_results


def train(model_name, hyper_params, plot_number):

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

    match model_name:
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

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Create plot
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.8, shuffle=False
    )
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plot_subtitle = generate_subtitle(
        hyper_params, np.mean(mse_scores), np.mean(r2_scores)
    )
    plot_results(
        model_name,
        y_test,
        y_pred,
        f"figures/{model_name}/{plot_number}.png",
        plot_subtitle,
    )
    return model, np.mean(mse_scores), np.mean(r2_scores), scaler, poly
