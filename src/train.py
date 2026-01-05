import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import preprocess_data

def train_model():
    df = preprocess_data("../data/house_price.csv")

    X = df.drop(columns=['Price'])
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features=0.7,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )



    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Model Performance")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R2:", r2_score(y_test, predictions))

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_model()
