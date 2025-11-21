import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Create a directory to save the model
os.makedirs('backend/models', exist_ok=True)

# Load the dataset
df = pd.read_csv('../test.csv')

# --- Feature Engineering and Preprocessing ---

# Select relevant features
features = ['Commodity', 'Variety', 'temperature_2m_mean', 'precipitation_sum', 'Modal_Price']
df_model = df[features].copy()

# Create a target variable for price risk
# A risk of 1 if the price drops more than 10% from the previous day for the same commodity
df_model['price_change'] = df_model.groupby('Commodity')['Modal_Price'].pct_change()
df_model['price_risk'] = (df_model['price_change'] < -0.05).astype(int)

# Drop rows with NaN values created by pct_change
df_model.dropna(inplace=True)

print("Value counts for 'price_risk':")
print(df_model['price_risk'].value_counts())

# One-hot encode categorical features
categorical_features = ['Commodity', 'Variety']
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(df_model[categorical_features])

# Get the column names for the encoded features
encoded_cols = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_cols, index=df_model.index)

# Combine with numerical features
numerical_features = ['temperature_2m_mean', 'precipitation_sum']
X = pd.concat([df_model[numerical_features], encoded_df], axis=1)
y = df_model['price_risk']

# --- Model Training ---

# Given the small dataset, we will train on the entire dataset
# and skip the train-test split and evaluation.
print("\nTraining model on the entire dataset...")
model = LogisticRegression(random_state=42)
model.fit(X, y)
print("Model training complete.")

# --- Save the Model and Encoder ---

# Save the trained model
joblib.dump(model, 'backend/models/price_risk_model.joblib')
print("\nModel saved to backend/models/price_risk_model.joblib")

# Save the encoder columns
joblib.dump(list(X.columns), 'backend/models/model_columns.joblib')
print("Model columns saved to backend/models/model_columns.joblib")
