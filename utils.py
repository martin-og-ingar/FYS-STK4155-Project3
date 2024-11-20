import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_subtitle(hyper_params, mse, r2):
    subtitle_parts = []
    for key, value in hyper_params.items():
        subtitle_parts.append(f"{key}: {value}")
    subtitle_parts.append(f"mse: {mse} (TSCV)")
    subtitle_parts.append(f"r2: {r2} (TSCV)")
    return "; ".join(subtitle_parts)


def plot_results(model_name, y_true=None, y_pred=None, plot_fn=None, subtitle=""):

    if y_true is None and y_pred is None:
        df = pd.read_csv(f"data/{model_name}_predictions.csv")
        y_true = df["disease_cases"]
        y_pred = df["pred cases"]

    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        x_axis = np.arange(len(y_true))
    else:
        x_axis = y_true.index

    plt.figure(figsize=(10, 6))

    # Plotting the true disease cases
    plt.plot(x_axis, y_true, label="True Disease Cases", color="blue", linewidth=2)

    # Plotting the predicted disease cases
    plt.plot(
        x_axis,
        y_pred,
        label="Predicted Disease Cases",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # Adding labels and title
    plt.xlabel("Index (Time or Instances)")
    plt.ylabel("Disease Cases")
    plt.suptitle("True vs Predicted Disease Cases")
    plt.title(subtitle)
    plt.legend()

    # Display the plot
    plt.grid(True)
    if plot_fn:
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()
