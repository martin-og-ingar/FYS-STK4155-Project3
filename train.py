import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
import joblib


def train(train_data_csv, lambda_value, model_output):
    print("training...")
    df = pd.read_csv(train_data_csv)

    features = ["rainfall", "mean_temperture"]
    target = ["disease_cases"]

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # train model
    model = Ridge(alpha=lambda_value)
    model.fit(X_train, y)

    joblib.dump(model, model_output)


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
