import pandas as pd
import matplotlib.pyplot as plt
import joblib


# # df = pd.read_csv("data/alagoas_dataset.csv")
# # plt.figure(figsize=(10, 6))
# # plt.title("Disease cases in Alagoas")
# # plt.ylabel("disease_cases")

# # print(df["disease_cases"].skew())

# # plt.plot(df["disease_cases"].skew())
# # plt.show()

# # print(df.describe())

# # df = df.drop(columns=["time_period", "location"])
# # corr_matrix = df.corr()
# # print(corr_matrix)

# # plt.figure(figsize=(10, 8))
# # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
# # plt.title("Correlation Heatmap")
# # plt.show()


# # rolmean = df["disease_cases"].rolling(window=12).mean()
# # rolstd = df["disease_cases"].rolling(window=12).std()

# # # Plot rolling statistics
# # plt.figure(figsize=(10, 6))
# # plt.plot(df["disease_cases"], label="Original")
# # plt.plot(rolmean, label="Rolling Mean")
# # plt.plot(rolstd, label="Rolling Std")
# # plt.legend(loc="best")
# # plt.title("Rolling Mean & Standard Deviation")
# # plt.show()

# # # Perform Dickey-Fuller test
# # print("Results of Dickey-Fuller Test:")
# # dftest = adfuller(df["disease_cases"], autolag="AIC")
# # dfoutput = pd.Series(
# #     dftest[0:4],
# #     index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
# # )
# # for key, value in dftest[4].items():
# #     dfoutput[f"Critical Value ({key})"] = value
# # print(dfoutput)
# # model_name = "ridge"
# # plot_results(model_name, plot_fn=f"test/{model_name}.png")

# # poly = joblib.load("models/ridge_poly.pkl")
# # print(poly.__dict__)

# # import matplotlib.pyplot as plt
# # import matplotlib.image as mpimg

# # images = [
# #     "figures/ridge/00.png",
# #     "figures/ridge/04.png",
# #     "figures/ridge/30.png",
# #     "figures/ridge/34.png",
# # ]

# # # Create a figure with a 2x2 grid (adjust layout as needed)
# # fig, axs = plt.subplots(2, 2, figsize=(30, 30))

# # # Load and plot each image
# # for ax, image_path in zip(axs.ravel(), images):
# #     img = mpimg.imread(image_path)
# #     ax.imshow(img)
# #     ax.tick_params(axis="both", which="major", labelsize=24)
# #     ax.axis("off")  # Hide axes


# # # Adjust spacing
# # plt.tight_layout()
# # plt.savefig("figures/report_figures/ridge.png")

# from sklearn.discriminant_analysis import StandardScaler
# from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, root_mean_squared_error
# from sklearn.model_selection import train_test_split

# df = pd.read_csv("data/train.csv")

# features = [
#     "rainfall",
#     "mean_temperature",
#     "disease_cases_ma_3",
#     "rainfall_ma_3",
#     "mean_temperature_ma_3",
#     "disease_cases_lag",
# ]
# target = ["disease_cases"]

# X = df[features]
# y = df[target]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# mae_scores = []
# x_train, x_test, y_train, y_test = train_test_split(
#     X_scaled, y, train_size=0.8, shuffle=False
# )
# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()
# num_trees = [
#     10,
#     50,
#     100,
#     200,
#     300,
# ]  # 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 180, 240, 260, 300]
# for i in num_trees:
#     model = BaggingRegressor(
#         max_samples=1.0,
#         max_features=1.0,
#         n_estimators=i,
#     )
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     mae = root_mean_squared_error(y_test, y_pred)
#     mae_scores.append(mae)


# plt.plot(num_trees, mae_scores)
# plt.show()
import seaborn as sns


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


model_name = "bagging"
results = pd.read_csv(f"scores/{model_name}.csv")
plot_scores(model_name, results)
