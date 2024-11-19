import joblib
from train import train
from predict import predict
import matplotlib.pyplot as plt
import pandas as pd
import sys

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
            for lmb in LMB_VALUES:
                for poly in POLY_DEGREES:
                    hyper_params = {"lmb": 0.001, "poly_degree": 2}
                    model, mse, r2, scaler, poly = train(
                        model=model_name, hyper_params=hyper_params
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
            for max_depth in MAX_DEPTHS:
                for min_samples_split in MIN_SAMPLE_SPLITS:
                    if model_name == "dt":
                        hyper_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                        }
                        model, mse, r2, scaler, _ = train(
                            model=model_name, hyper_params=hyper_params
                        )
                        if mse < best_mse:
                            best_mse = mse
                            current_model = model
                            current_scaler = scaler

        case "rf":
            for max_depth in MAX_DEPTHS:
                for min_samples_split in MIN_SAMPLE_SPLITS:
                    for num_trees in NUM_TREES:
                        hyper_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "num_trees": num_trees,
                        }
                        model, mse, r2, scaler, _ = train(
                            model=model_name, hyper_params=hyper_params
                        )
                        if mse < best_mse:
                            best_mse = mse
                            current_model = model
                            current_scaler = scaler

        case "bagging":
            for max_features in MAX_FEAUTRES:
                for max_samples in MAX_SAMPLES:
                    for num_trees in NUM_TREES:
                        hyper_params = {
                            "num_trees": num_trees,
                            "max_features": max_features,
                            "max_samples": max_samples,
                        }
                        model, mse, r2, scaler, _ = train(
                            model=model_name, hyper_params=hyper_params
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


def plot_results(model_name):
    df = pd.read_csv(f"data/{model_name}_predictions.csv")
    y_true = df["disease_cases"]
    y_pred = df["pred cases"]
    plt.figure(figsize=(10, 6))

    # Plotting the true disease cases
    plt.plot(df.index, y_true, label="True Disease Cases", color="blue", linewidth=2)

    # Plotting the predicted disease cases
    plt.plot(
        df.index,
        y_pred,
        label="Predicted Disease Cases",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # Adding labels and title
    plt.xlabel("Index (Time or Instances)")
    plt.ylabel("Disease Cases")
    plt.title("True vs Predicted Disease Cases")
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    argument = sys.argv[1]

    evaluate(argument)
    plot_results(argument)
