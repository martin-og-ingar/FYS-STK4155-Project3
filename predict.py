import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys


def predict(model_name):

    df = pd.read_csv("data/test.csv")
    features = [
        "rainfall",
        "mean_temperature",
        "disease_cases_ma_3",
        "rainfall_ma_3",
        "mean_temperature_ma_3",
        "disease_cases_lag",
    ]
    Y = df["disease_cases"]
    X = df[features]
    model = joblib.load(f"models/{model_name}.pkl")

    scaler = joblib.load(f"models/{model_name}_scaler.pkl")
    X_scaled = scaler.transform(X)

    if model_name == "ridge":
        poly = joblib.load("models/ridge_poly.pkl")
        X_scaled = poly.transform(X_scaled)

    pred = model.predict(X_scaled)
    mse = mean_squared_error(Y, pred)
    r2 = r2_score(Y, pred)

    df["pred cases"] = pred
    df.to_csv(f"data/{model_name}_predictions.csv", index=False)

    return mse, r2


if __name__ == "__main__":
    argument = sys.argv[1]

    print(predict(model_name=argument))
