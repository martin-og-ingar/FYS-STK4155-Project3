import joblib
from train import train
from predict import predict
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utils import plot_results

LMB_VALUES = [0.0001, 0.001, 0.01, 0.1]
POLY_DEGREES = [1, 3, 5, 7, 9]
MAX_DEPTHS = [5, 10, 15]
MIN_SAMPLE_SPLITS = [5, 10, 15]
NUM_TREES = [10, 50, 100, 200, 300]
MAX_SAMPLES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_FEAUTRES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def evaluate(model_name):
    results = []
    best_mse = float("inf")
    match model_name:
        case "ridge":
            for i, lmb in enumerate(LMB_VALUES):
                for j, poly in enumerate(POLY_DEGREES):
                    hyper_params = {"lmb": lmb, "poly_degree": poly}
                    model, mse, r2, scaler, poly = train(
                        model_name=model_name,
                        hyper_params=hyper_params,
                        plot_number=str(i) + str(j),
                    )
                    results.append(
                        {
                            "lmb": lmb,
                            "poly": poly,
                            "mse": mse,
                            "r2": r2,
                        }
                    )
                    if mse < best_mse:
                        best_mse = mse
                        current_model = model
                        current_scaler = scaler
                        current_poly = poly

        case "dt":
            for i, max_depth in enumerate(MAX_DEPTHS):
                for j, min_samples_split in enumerate(MIN_SAMPLE_SPLITS):
                    if model_name == "dt":
                        hyper_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                        }
                        model, mse, r2, scaler, _ = train(
                            model_name=model_name,
                            hyper_params=hyper_params,
                            plot_number=str(i) + str(j),
                        )
                        if mse < best_mse:
                            best_mse = mse
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
                        model, mse, r2, scaler, _ = train(
                            model_name=model_name,
                            hyper_params=hyper_params,
                            plot_number=str(i) + str(j) + str(k),
                        )
                        if mse < best_mse:
                            best_mse = mse
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
                        model, mse, r2, scaler, _ = train(
                            model_name=model_name,
                            hyper_params=hyper_params,
                            plot_number=str(i) + str(j) + str(k),
                        )
                        if mse < best_mse:
                            best_mse = mse
                            current_model = model
                            current_scaler = scaler
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
    plot_results(model_name)
