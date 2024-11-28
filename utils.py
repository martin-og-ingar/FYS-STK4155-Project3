import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
import seaborn as sns


def generate_subtitle(hyper_params, rmse, r2, mean_rmse, mean_r2):
    subtitle_parts = []
    if hyper_params:
        for key, value in hyper_params.items():
            subtitle_parts.append(f"{key}: {value}")
    subtitle_parts.append(f"\n rmse: {rmse}")
    subtitle_parts.append(f"r2: {r2}")
    if mean_rmse:
        subtitle_parts.append(f"rmse: {mean_rmse} (TSCV)")
    if mean_r2:
        subtitle_parts.append(f"r2: {mean_r2} (TSCV)")
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

    plt.figure(figsize=(12, 10))

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
    plt.suptitle(f"{model_name}: True vs Predicted Disease Cases")
    plt.title(subtitle)
    plt.legend()

    # Display the plot
    plt.grid(True)
    if plot_fn:
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()


def plot_tree(model, plot_fn):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    tree.plot_tree(model, filled=True, rounded=True, ax=axes)
    plt.savefig(plot_fn)


def plot_scores(model_name, results):
    match model_name:
        case "ridge":
            plot_heatmap(
                results,
                "lmb",
                "poly_degree",
                "rmse",
                model_name,
                f"rmse/{model_name}/{model_name}",
            )
            plot_heatmap(
                results,
                "lmb",
                "poly_degree",
                "r2",
                model_name,
                f"r2/{model_name}/{model_name}",
            )
        case "dt":
            plot_heatmap(
                results,
                "min_samples_split",
                "max_depth",
                "rmse",
                model_name,
                f"rmse/{model_name}/{model_name}",
            )
            plot_heatmap(
                results,
                "min_samples_split",
                "max_depth",
                "r2",
                model_name,
                f"r2/{model_name}/{model_name}",
            )
        case "rf":
            for num_trees in results["num_trees"].unique():
                new_results = results[results["num_trees"] == num_trees]
                plot_heatmap(
                    new_results,
                    "min_samples_split",
                    "max_depth",
                    "rmse",
                    model_name,
                    f"rmse/{model_name}/{model_name}_{num_trees}",
                )
                plot_heatmap(
                    new_results,
                    "min_samples_split",
                    "max_depth",
                    "r2",
                    model_name,
                    f"r2/{model_name}/{model_name}_{num_trees}",
                )
        case "bagging":
            for num_trees in results["num_trees"].unique():
                new_results = results[results["num_trees"] == num_trees]
                plot_heatmap(
                    new_results,
                    "max_samples",
                    "max_features",
                    "rmse",
                    model_name,
                    f"rmse/{model_name}/{model_name}_{num_trees}",
                )
                plot_heatmap(
                    new_results,
                    "max_samples",
                    "max_features",
                    "r2",
                    model_name,
                    f"r2/{model_name}/{model_name}_{num_trees}",
                )


def plot_heatmap(df, x, y, z, model_name, plot_fn):
    heatmap_data = df.pivot_table(index=y, columns=x, values=z)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": z},
    )

    plt.title(f"{model_name} Hyperparameter Heatmap")
    plt.xlabel(x)
    plt.ylabel(y)

    plt.savefig(f"figures/{plot_fn}.png")
    plt.close()
