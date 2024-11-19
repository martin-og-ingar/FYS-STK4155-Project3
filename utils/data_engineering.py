import pandas as pd

df = pd.read_csv("data/dataset.csv")
df = df.drop(columns="Unnamed: 0")
df = df[df["location"] == "Alagoas"]


# Calculate and add moving average
df["disease_cases_ma_3"] = df["disease_cases"].rolling(window=3).mean().round(2)
df["rainfall_ma_3"] = df["rainfall"].rolling(window=3).mean().round(2)
df["mean_temperature_ma_3"] = df["mean_temperature"].rolling(window=3).mean().round(2)

# Add lagged value for disease cases
df["disease_cases_lag"] = df["disease_cases"].shift(1)

df = df.dropna().reset_index(drop=True)


df.to_csv("data/alagoas_dataset.csv", index=False)

split_index = int(len(df) * 0.8)

train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)
