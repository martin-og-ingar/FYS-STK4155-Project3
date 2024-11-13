import pandas as pd


def clean_up_historic_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    df.drop(
        columns=["meantemperature", "rainsum", "ID_spat", "E", "Cases"], inplace=True
    )

    df["disease_cases"] = df.groupby("location")["disease_cases"].transform(
        lambda group: group.fillna(group.median())
    )

    df.to_csv(output_csv, index=False)
