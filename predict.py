import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def predict(trained_model, test_data_csv, predictions):
    print("predicting")

    df = pd.read_csv(test_data_csv)
    features = ["rainfall", "mean_temperture"]
    X_new = df[features]
    model = joblib.load(trained_model)

    predictions = model.predict(X_new)

    train_data = pd.read_csv("data/train_data.csv")
    y_train = train_data["disease_cases"]
    X_train = train_data[features]

    mse = mean_squared_error(y_train, model.predict(X_train))
    r2 = r2_score(y_train, model.predict(X_train))

    # add gaussian noise?

    df.to_csv(predictions, index=False)

    print(f"Predictions saved to {predictions}")


if __name__ == "__main__":
    predict()
