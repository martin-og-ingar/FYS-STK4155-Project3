from train import train
from predict import predict


def evaluate():
    print("evaluating...")
    train("data/train_data.csv", 0.001, "data/model.pkl")
    print(
        predict(
            "data/model.pkl",
            "data/scaler.pkl",
            "data/poly.pkl",
            "data/historic_data.csv",
            "data/future_data.csv",
            "data/predictions.csv",
        )
    )


if __name__ == "__main__":
    evaluate()
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("data/predictions.csv")
    y_true = df["disease_cases"]
    print(y_true.max())
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
