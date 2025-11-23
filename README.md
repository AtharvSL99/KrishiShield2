**KrishiShield - Predictive Crop Loss Early-Warning Dashboard**

KrishiShield is an intelligent, offline-first decision support system designed to help Indian farmers mitigate financial risks caused by unpredictable weather and market volatility. It leverages historical data and machine learning to provide actionable crop advisories and price risk assessments.

🚀 Features

Crop Risk Calculator: Predicts potential price fluctuations based on weather forecasts.

Multi-Crop Support: Supports Onion, Wheat, and Potato.

Offline-First: Caches weather data for offline usage in rural areas.

Advisory System: Provides actionable farming advice based on risk levels.

Localized Insights: Uses specific market and weather data for precise predictions.

🛠️ Setup & Installation

Clone the Repository:

git clone [https://github.com/atharvsl99/krishishield2.git](https://github.com/atharvsl99/krishishield2.git)
cd krishishield2


Install Dependencies:
Ensure you have Python installed. Then, install the required packages:

pip install streamlit pandas numpy xgboost scikit-learn requests


Verify Data Files:
Ensure the following files are present in the root directory:

semifinal.csv (Primary Data) or test.csv (Fallback)

Model files: Onion.pkl, Wheat.pkl, Potato.pkl

Scaler files: onion_scaler.pkl, wheat_scaler.pkl, potato_scaler.pkl

Processed feature files: *_market_lagged_features.csv

▶️ Usage

To launch the application, run the following command in your terminal:

python -m streamlit run app.py


The application will open in your default web browser at http://localhost:8501.

📂 Project Structure

app.py: Main Streamlit application code.

data_preparation.py: Script for cleaning and preparing data.

model_training.py: Script for training XGBoost models.

*.pkl: Trained models and scalers.

*.csv: Dataset files.
