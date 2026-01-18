import pandas as pd

def preprocess_final_dataset(csv_path):
    """
    Minimal preprocessing since dataset is already cleaned and merged.
    Just shuffle rows and ensure correct column ordering.
    """

    print("Loading final dataset...")
    df = pd.read_csv(csv_path)

    print("Current dataset shape:", df.shape)

    # Ensure no identifier columns remain
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)

    # Final shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Dataset shuffled. Final shape:", df.shape)

    return dfCommit on 2026-01-18T20:47:25
