import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import preprocess_final_dataset


def compute_regression_accuracy(y_true, y_pred, tolerance=0.15):
    """
    Calculates practical accuracy for regression:
    percentage of predictions within ±tolerance% of actual values
    """
    correct = np.sum(
        (y_pred >= y_true * (1 - tolerance)) &
        (y_pred <= y_true * (1 + tolerance))
    )
    return correct / len(y_true) * 100


def train_model():

    # Load and preprocess only the final merged dataset
    df = preprocess_final_dataset("../data/house_price.csv")

    # Separate features and target
    X = df.drop(columns=['Price'])
    y = df['Price']

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest...")

    # Use the best parameters you supplied
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=6,
        min_samples_leaf=1,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nModel Performance:")
    print("MAE :", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2  :", r2_score(y_test, preds))

    # Feature importance display
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(importances.head(10))

    importances.head(10).plot(kind='barh')
    plt.title("Top 10 Important Features")
    plt.show()
    
    accuracy = compute_regression_accuracy(preds, y_test)
    print(f"\nRegression Accuracy: {accuracy:.2f}%")


    # Save final model
    print("\nSaving trained model...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as model.pkl")

    return model


if __name__ == "__main__":
    train_model()
Commit at 2026-01-11T00:49:21.472909
Commit at 2026-01-15T06:44:20.707631
Commit at 2026-01-15T02:01:24.874188
Commit at 2026-01-15T17:21:28.044694
Commit at 2026-01-15T09:34:17.236065
Commit at 2026-01-12T02:22:21.397228
Commit on 2026-01-19T01:53:21
Commit on 2026-01-18T23:35:21
Commit on 2026-01-17T17:51:10
Commit on 2026-01-17T03:21:07
Commit on 2026-01-16T12:39:06
