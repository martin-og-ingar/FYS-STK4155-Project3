import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
import joblib


def train(train_data_csv, lambda_value, model_output):
    print("training...")
    df = pd.read_csv(train_data_csv)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    features = ["rainfall", "mean_temperature"]
    target = ["disease_cases"]

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=2)

    X_scaled = scaler.fit_transform(X)
    X_poly = poly.fit_transform(X_scaled)

    mse_scores = []
    r2_scores = []
    splits = 5

    for fold in range(1, splits + 1):
        # Define train/test split sizes for expanding window
        train_size = int(len(X_scaled) * fold / splits)

        test_size = len(X_scaled) - train_size
        if test_size < 1:
            break

        # Train on expanding window
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        print(
            f"Fold {fold}: Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples"
        )

        model = Ridge(alpha=lambda_value)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    print(f"Mean MSE: {np.mean(mse_scores)}")
    print(f"Mean R2: {np.mean(r2_scores)}")

    model = Ridge(alpha=lambda_value)
    model.fit(X_poly, y)

    joblib.dump(model, model_output)
    joblib.dump(scaler, "data/scaler.pkl")
    joblib.dump(poly, "data/poly.pkl")


def split_data(train_data, historic_data, future_data):

    df = pd.read_csv(train_data)
    locations = df["location"].unique()

    features = [
        "time_period",
        "rainfall",
        "mean_temperature",
        "location",
    ]
    target = ["disease_cases"]

    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

    for loc in locations:
        location_data = df[df["location"] == loc]
        X_loc = location_data[features]
        y_loc = location_data[target]
        # split data into train and test sets.
        X_train_loc, X_test_loc, y_train_loc, y_test_loc = train_test_split(
            X_loc, y_loc, test_size=0.2, shuffle=False
        )

        X_train_list.append(X_train_loc)
        X_test_list.append(X_test_loc)
        y_train_list.append(y_train_loc)
        y_test_list.append(y_test_loc)

    X_train = pd.concat(X_train_list)
    X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)

    historic_data = pd.concat([X_train, y_train], axis=1)
    future_data = pd.concat([X_test, y_test], axis=1)

    historic_data.to_csv("data/historic_data.csv")
    future_data.to_csv("data/future_data.csv")


if __name__ == "__main__":
    train_data_csv = "data/train_data.csv"
    historic_data = "data/historic_data.csv"
    future_data = "data/future_data.csv"
    split_data(train_data_csv, historic_data, future_data)
