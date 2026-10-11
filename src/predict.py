import joblib
import pandas as pd

# Load the saved model
model = joblib.load("models/house_price_model.pkl")

# Example house details
house = pd.DataFrame([{
    "Area": 1500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Parking": 1,
    "Age": 5
}])

# Predict the house price
predicted_price = model.predict(house)[0]

print(f"Predicted house price: {predicted_price:.2f} lakhs")