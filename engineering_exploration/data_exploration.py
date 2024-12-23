import pandas as pd
import matplotlib.pyplot as plt
import joblib


import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor


def plot_scores(model_name, results):
    match model_name:
        case "ridge":
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
                    "r2",
                    model_name,
                    f"r2/{model_name}/{model_name}_{num_trees}",
                )


def plot_heatmap(df, x, y, z, model_name, plot_fn):
    # Pivot the DataFrame for heatmap compatibility
    heatmap_data = df.pivot_table(index=y, columns=x, values=z)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        heatmap_data,
        annot=True,  # Annotate with numerical values
        fmt=".3f",  # Format numbers to 3 decimal places
        cmap="viridis",  # Color scheme
        cbar_kws={"label": z},  # Add color bar label
    )

    # Add titles and labels
    plt.title(f"{model_name} Hyperparameter Heatmap")
    plt.xlabel(x)
    plt.ylabel(y)

    # Save the heatmap as a PNG file
    plt.savefig(f"figures/{plot_fn}.png")
    plt.close()  # Close the plot to free memory


model_name = "bagging_test"
df = pd.read_csv("data/train.csv")

features = [
    "rainfall",
    "mean_temperature",
    "disease_cases_ma_3",
    "rainfall_ma_3",
    "mean_temperature_ma_3",
    "disease_cases_lag",
]
target = ["disease_cases"]

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=0.8, shuffle=False
)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
results = []
NUM_TREES = [10, 50, 100, 200, 300]
MAX_SAMPLES = [0.5, 0.75, 1.0]
MAX_FEAUTRES = [0.5, 0.75, 1.0]
MAX_DEPTHS = [5, 10, 15, 30, 45]
MIN_SAMPLE_SPLITS = [2, 3, 5, 10, 15]

param_grid = {"max"}
dtree = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(
    dtree,
)
