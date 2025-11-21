import joblib
import pandas as pd
import numpy as np

# Load the trained model and column names
model = joblib.load('backend/models/price_risk_model.joblib')
model_columns = joblib.load('backend/models/model_columns.joblib')

# --- Create Sample Data for Prediction ---

# This sample data should have the same structure as the training data,
# including the one-hot encoded features.
sample_data = {
    'temperature_2m_mean': [25.0],
    'precipitation_sum': [10.0],
    'Commodity_Onion': [0],
    'Commodity_Tomato': [0],
    'Commodity_Wheat': [1],
    'Variety_Other': [1],
    'Variety_Red': [0],
    'Variety_Maharashtra 2189':[0]
}

# Create a DataFrame from the sample data
df_sample = pd.DataFrame(sample_data)

# --- Reorder Columns to Match Training Data ---

# Get the list of all columns from the saved model columns
all_cols = model_columns

# Create a new DataFrame with all the columns, initialized to 0
df_pred = pd.DataFrame(columns=all_cols)
new_row = pd.Series(0, index=all_cols)
df_pred = pd.concat([df_pred, new_row.to_frame().T], ignore_index=True)


# Fill in the values from the sample data
for col in df_sample.columns:
    if col in df_pred.columns:
        df_pred[col] = df_sample[col].values[0]

# Ensure all columns are numeric
df_pred = df_pred.apply(pd.to_numeric)


# --- Make Prediction ---

prediction = model.predict(df_pred)
prediction_proba = model.predict_proba(df_pred)

print("--- 'Prediction' ---")
print(f"Sample Data: {sample_data}")
print(f"\nPrediction (0 = No Risk, 1 = Price Risk): {prediction[0]}")
print(f"Prediction Probability: {prediction_proba[0]}")
