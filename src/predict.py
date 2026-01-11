import pickle
import pandas as pd

def predict_price(input_data):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches training
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    sample_house = {
        "number of bedrooms": 4,
        "number of bathrooms": 2,
        "living area": 3000,
        "lot area": 5000,
        "number of floors": 2,
        "waterfront present": 0,
        "number of views": 0,
        "condition of the house": 4,
        "grade of the house": 8,
        "Area of the house(excluding basement)": 2000,
        "Area of the basement": 1000,
        "Postal Code": 122004,
        "Lattitude": 52.88,
        "Longitude": -114.47,
        "living_area_renov": 2800,
        "lot_area_renov": 5000,
        "Number of schools nearby": 2,
        "Distance from the airport": 50,
        "house_age": 20,
        "years_since_renovation": 10
    }

    price = predict_price(sample_house)
    print("Predicted House Price:", round(price, 2))
Commit at 2026-01-11T15:30:12.886151
