import joblib
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
import sys
from utils import plot_results, generate_subtitle


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
    rmse = root_mean_squared_error(Y, pred)
    r2 = r2_score(Y, pred)

    df["pred cases"] = pred
    df.to_csv(f"data/{model_name}_predictions.csv", index=False)

    match model_name:
        case "ridge":
            hyper_params = {"lmb": model.alpha, "poly_degree": poly.degree}
        case "dt":
            hyper_params = {
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
            }
        case "rf":
            hyper_params = {
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "n_estimators": model.n_estimators,
            }
        case "bagging":
            hyper_params = {
                "n_estimators": model.n_estimators,
                "max_features": model.max_features,
                "max_samples": model.max_samples,
            }
    subtitle = generate_subtitle(
        hyper_params=hyper_params, rmse=rmse, r2=r2, mean_rmse=None, mean_r2=None
    )
    plot_results(
        model_name, plot_fn=f"figures/test/{model_name}.png", subtitle=subtitle
    )

    return rmse, r2


if __name__ == "__main__":
    argument = sys.argv[1]

    print(predict(model_name=argument))
