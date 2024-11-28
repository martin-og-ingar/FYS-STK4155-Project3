import pandas as pd
import matplotlib.pyplot as plt
import joblib


import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
MAX_DEPTHS = [5, 10, 15]
MIN_SAMPLE_SPLITS = [5, 10, 15]
best_rmse = float("inf")
for i in NUM_TREES:
    for j in MAX_SAMPLES:
        for k in MAX_FEAUTRES:
            for l in MAX_DEPTHS:
                for p in MIN_SAMPLE_SPLITS:
                    model = BaggingRegressor(
                        n_estimators=i,
                        max_samples=j,
                        max_features=k,
                        # estimator=DecisionTreeRegressor(
                        #     max_depth=l, min_samples_split=p, criterion="absolute_error"
                        # ),
                    )
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    rmse = root_mean_squared_error(y_test, y_pred)
                    if rmse < best_rmse:
                        current_model = model

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

X_scaled = scaler.transform(X)

y_pred = current_model.predict(X)
rmse = root_mean_squared_error(Y, y_pred)
r2 = r2_score(Y, y_pred)
print(rmse)
print(r2)
