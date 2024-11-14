import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def predict(trained_model, scaler, poly, historic_data, future_data, predictions):
    print("predicting")

    df = pd.read_csv(future_data)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    features = ["rainfall", "mean_temperature"]
    X_new = df[features]
    model = joblib.load(trained_model)
    scale = joblib.load(scaler)
    pol = joblib.load(poly)

    X_new_scaled = scale.transform(X_new)
    X_new_poly = pol.transform(X_new_scaled)

    pred = model.predict(X_new_poly)

    y_true = df["disease_cases"]

    mse = mean_squared_error(y_true, pred)
    r2 = r2_score(y_true, pred)

    df["pred cases"] = pred
    df.to_csv(predictions, index=False)

    print(f"Predictions saved to {predictions}")

    return mse, r2


if __name__ == "__main__":
    predict()
