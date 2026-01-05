import pandas as pd

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop identifier / non-informative columns
    drop_cols = ['id', 'Date']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Fix renovation year (row-wise)
    df.loc[df['Renovation Year'] == 0, 'Renovation Year'] = df['Built Year']

    # Feature engineering
    CURRENT_YEAR = 2024
    df['house_age'] = CURRENT_YEAR - df['Built Year']
    df['years_since_renovation'] = CURRENT_YEAR - df['Renovation Year']

    # Drop original year columns
    df.drop(columns=['Built Year', 'Renovation Year'], inplace=True)

    return df
