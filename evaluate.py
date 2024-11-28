import joblib
from train import train
from predict import predict
import sys
import pandas as pd

from utils import plot_scores

# Hyper params to be evaluated
LMB_VALUES = [0.0001, 0.001, 0.01, 0.1]
POLY_DEGREES = [1, 3, 5, 7, 9]
MAX_DEPTHS = [5, 10, 15]
MIN_SAMPLE_SPLITS = [5, 10, 15]
NUM_TREES = [10, 50, 100, 200, 300]
MAX_SAMPLES = [0.5, 0.75, 1.0]
MAX_FEAUTRES = [0.5, 0.75, 1.0]


def evaluate(model_name):
    best_rmse = float("inf")
    results = []
    match model_name:
        case "ridge":
            for i, lmb in enumerate(LMB_VALUES):
                for j, poly in enumerate(POLY_DEGREES):
                    hyper_params = {"lmb": lmb, "poly_degree": poly}
                    model, rmse, r2, scaler, poly_ = train(
                        model_name=model_name,
                        hyper_params=hyper_params,
                        plot_number=str(i) + str(j),
                    )
                    results.append(
                        {"lmb": lmb, "poly_degree": poly, "rmse": rmse, "r2": r2}
                    )
                    if rmse < best_rmse:
                        best_rmse = rmse
                        current_model = model
                        current_scaler = scaler
                        current_poly = poly_

        case "dt":
            for i, max_depth in enumerate(MAX_DEPTHS):
                for j, min_samples_split in enumerate(MIN_SAMPLE_SPLITS):
                    if model_name == "dt":
                        hyper_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                        }
                        model, rmse, r2, scaler, _ = train(
                            model_name=model_name,
                            hyper_params=hyper_params,
                            plot_number=str(i) + str(j),
                        )
                        results.append(
                            {
                                "max_depth": max_depth,
                                "min_samples_split": min_samples_split,
                                "rmse": rmse,
                                "r2": r2,
                            }
                        )
                        if rmse < best_rmse:
                            best_rmse = rmse
                            current_model = model
                            current_scaler = scaler

        case "rf":
            for i, max_depth in enumerate(MAX_DEPTHS):
                for j, min_samples_split in enumerate(MIN_SAMPLE_SPLITS):
                    for k, num_trees in enumerate(NUM_TREES):
                        hyper_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "num_trees": num_trees,
                        }
                        model, rmse, r2, scaler, _ = train(
                            model_name=model_name,
                            hyper_params=hyper_params,
                            plot_number=str(i) + str(j) + str(k),
                        )
                        results.append(
                            {
                                "max_depth": max_depth,
                                "min_samples_split": min_samples_split,
                                "num_trees": num_trees,
                                "rmse": rmse,
                                "r2": r2,
                            }
                        )
                        if rmse < best_rmse:
                            best_rmse = rmse
                            current_model = model
                            current_scaler = scaler

        case "bagging":
            for i, max_features in enumerate(MAX_FEAUTRES):
                for j, max_samples in enumerate(MAX_SAMPLES):
                    for k, num_trees in enumerate(NUM_TREES):
                        hyper_params = {
                            "num_trees": num_trees,
                            "max_features": max_features,
                            "max_samples": max_samples,
                        }
                        model, rmse, r2, scaler, _ = train(
                            model_name=model_name,
                            hyper_params=hyper_params,
                            plot_number=str(i) + str(j) + str(k),
                        )
                        results.append(
                            {
                                "max_samples": max_samples,
                                "max_features": max_features,
                                "num_trees": num_trees,
                                "rmse": rmse,
                                "r2": r2,
                            }
                        )
                        if rmse < best_rmse:
                            best_rmse = rmse
                            current_model = model
                            current_scaler = scaler

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"scores/{model_name}.csv")

    plot_scores(model_name, results_df)

    joblib.dump(current_model, f"models/{model_name}.pkl")
    joblib.dump(current_scaler, f"models/{model_name}_scaler.pkl")
    if model_name == "ridge":
        joblib.dump(current_poly, f"models/{model_name}_poly.pkl")

    print(
        predict(
            model_name,
        )
    )


if __name__ == "__main__":

    model_name = sys.argv[1]
    evaluate(model_name)
