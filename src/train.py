import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Set up MLflow experiment
mlflow.set_experiment("House Price Prediction")

# Load dataset
data = pd.read_csv("data/house_prices.csv")

# Select features and target
X = data[["Area", "Bedrooms", "Bathrooms", "Parking", "Age"]]
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and track the experiment
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mae", mae)

    # Log the trained model
    mlflow.sklearn.log_model(model, name="house_price_model")

    print(f"Mean Absolute Error: {mae:.2f} lakhs")
    print("Experiment logged to MLflow!")

# Save model locally too
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/house_price_model.pkl")

print("Model trained and saved successfully!")