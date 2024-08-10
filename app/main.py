"""
Author: Musa Misto
Date: 2024-08-10
Description: This FastAPI application is designed for predicting fraudulent transactions using a pre-trained Random Forest model.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import json

# Initialize FastAPI app
app = FastAPI()

# Load the trained model, encoder, scaler, and column names
model = joblib.load("app/random_forest_model.pkl")
encoder = joblib.load("app/one_hot_encoder.pkl")
scaler = joblib.load("app/scaler.pkl")
X_columns = joblib.load("app/random_forest_features.pkl")

# Mount the static directory to serve CSS, JS, and other static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Handles GET requests to the root URL.
    Renders the main page of the Fraud Detection System.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Handles POST requests to /predict/ for fraud detection.
    
    Expects a JSON file containing transaction data, processes the data,
    and returns whether the transaction is fraudulent or legitimate.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        input_data = json.loads(contents.decode("utf-8"))

        # Convert the JSON data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Adjust input data format to match training data
        for col in ['category', 'merchant', 'age', 'gender']:
            input_df[col] = "'" + input_df[col].astype(str) + "'"

        # Encode categorical variables using the same OneHotEncoder as used during training
        encoded_data = encoder.transform(input_df[['category', 'merchant', 'age', 'gender']])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['category', 'merchant', 'age', 'gender']))

        # Prepare the input data for prediction
        numerical_data = input_df[['amount']]
        processed_data = pd.concat([numerical_data, encoded_df], axis=1)

        # Ensure the processed data has the same columns as the training data
        missing_cols = set(X_columns) - set(processed_data.columns)
        for col in missing_cols:
            processed_data[col] = 0
        processed_data = processed_data[X_columns]

        # Scale the 'amount' column using the pre-fitted scaler
        processed_data[['amount']] = scaler.transform(processed_data[['amount']])

        # Make a prediction using the trained model
        prediction = model.predict(processed_data)
        result = "Fraudulent transaction" if prediction[0] == 1 else "Legitimate transaction"

        # Return the prediction result
        return {"result": result}
    except Exception as e:
        # Handle errors and return them as a result
        return {"result": f"Error: {e}"}
