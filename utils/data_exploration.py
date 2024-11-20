import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("data/alagoas_dataset.csv")
plt.figure(figsize=(10, 6))
plt.title("Disease cases in Alagoas")
plt.ylabel("disease_cases")

plt.plot(df["disease_cases"])

print(df.describe())

df = df.drop(columns=["time_period", "location"])
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


rolmean = df["disease_cases"].rolling(window=12).mean()
rolstd = df["disease_cases"].rolling(window=12).std()

# Plot rolling statistics
plt.figure(figsize=(10, 6))
plt.plot(df["disease_cases"], label="Original")
plt.plot(rolmean, label="Rolling Mean")
plt.plot(rolstd, label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean & Standard Deviation")
plt.show()

# Perform Dickey-Fuller test
print("Results of Dickey-Fuller Test:")
dftest = adfuller(df["disease_cases"], autolag="AIC")
dfoutput = pd.Series(
    dftest[0:4],
    index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
)
for key, value in dftest[4].items():
    dfoutput[f"Critical Value ({key})"] = value
print(dfoutput)
