import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta, date
import os
import xgboost as xgb

# --- Configuration ---
DATA_FILE = 'semifinal.csv'
MODEL_FILE = 'Onion.pkl'
SCALER_FILE = 'Onion_scaler.pkl'
LAG_WINDOW = 4
DEFAULT_COMMODITY = 'Onion'
PRICE_COLUMN = 'Modal_Price'

# --- Utility Functions ---

@st.cache_data
def load_base_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=False)
        # Cleanup redundant index columns
        cols_to_drop = []
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '0':
            cols_to_drop.append(df.columns[0])
        if len(df.columns) > 1 and (df.columns[1].startswith('Unnamed') or df.columns[1] == '0'):
            cols_to_drop.append(df.columns[1])
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def load_artifacts():
    """Loads model/scaler and forces CPU mode for stability."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
            
        # Force CPU to avoid "Mismatched devices" crash
        if isinstance(model, xgb.XGBRegressor):
            model.set_params(device='cpu', tree_method='hist')
            try:
                model.get_booster().set_param({'device': 'cpu', 'tree_method': 'hist'})
            except:
                pass
            
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Artifacts not found. Please re-train model.")
        return None, None

@st.cache_data
def load_lagged_data(commodity):
    file_name = f'{commodity.lower()}_market_lagged_features.csv'
    try:
        df = pd.read_csv(file_name, index_col=[0, 1], parse_dates=True)
        df.index.names = ['Market Name', 'Price Date'] 
        return df
    except Exception:
        return pd.DataFrame()

# --- Helpers ---

def get_historical_features(df_lagged, market, target_date):
    if df_lagged.empty: return None
    try:
        market_data = df_lagged.loc[market]
        target_dt = pd.to_datetime(target_date)
        diffs = np.abs(market_data.index - target_dt)
        closest_idx = np.argmin(diffs)
        if diffs[closest_idx] < timedelta(days=4):
            return market_data.iloc[[closest_idx]]
    except:
        pass
    return None

def prepare_input_features(simulated_data, model_features):
    """
    Prepares input vector. 
    UPDATED: Uses direct weather_code assignment (No OHE).
    """
    X_pred = pd.DataFrame(0.0, index=[0], columns=model_features)
    
    est_price = np.mean([d.get(f'{PRICE_COLUMN}_Lag1', 2500) for d in simulated_data])

    for lag in range(1, LAG_WINDOW + 1):
        data = simulated_data[lag-1]
        
        # Assign all numerical lags including weather_code
        cols = ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 
                'wind_speed_10m_max', 'precipitation_sum', 'precipitation_hours',
                'weather_code'] # <-- Now treated as simple number
        
        for col in cols:
            col_name = f'{col}_Lag{lag}'
            if col_name in X_pred.columns:
                X_pred.loc[0, col_name] = data[col]
            
        # Assign Price
        X_pred.loc[0, f'{PRICE_COLUMN}_Lag{lag}'] = data.get(f'{PRICE_COLUMN}_Lag1', est_price)
            
    return X_pred

# --- UI ---

def manual_input_ui():
    st.warning("No history found. Enter estimates:")
    data = []
    cols = st.columns(LAG_WINDOW)
    for i, col in enumerate(cols):
        lag = i + 1
        with col:
            st.markdown(f"**Week -{lag}**")
            price = st.number_input(f"Price (W-{lag})", 500.0, 10000.0, 2500.0, key=f'p_{lag}')
            temp = st.number_input(f"Max Temp", 10.0, 50.0, 30.0, key=f't_{lag}')
            rain = st.number_input(f"Rain (mm)", 0.0, 500.0, 0.0, key=f'r_{lag}')
            code = st.selectbox(f"Weather Code", [0.0, 1.0, 3.0, 51.0, 63.0, 95.0], key=f'c_{lag}')
            
            data.append({
                f'{PRICE_COLUMN}_Lag1': price,
                'temperature_2m_max': temp,
                'precipitation_sum': rain,
                'weather_code': float(code),
                'temperature_2m_mean': temp - 5,
                'temperature_2m_min': temp - 10,
                'wind_speed_10m_max': 10.0,
                'precipitation_hours': 0.0
            })
    return data

def main():
    st.set_page_config(page_title="Crop Predictor", layout="wide")
    st.title(f"🌾 {DEFAULT_COMMODITY} Price Predictor")
    
    df_base = load_base_data(DATA_FILE)
    model, scaler = load_artifacts()
    
    if df_base.empty or model is None:
        st.stop()
        
    st.sidebar.header("Settings")
    market = st.sidebar.selectbox("Market", df_base['Market Name'].unique())
    min_date = date(2024, 6, 6)
    max_date = date(2025, 6, 6)
    pred_date = st.sidebar.date_input("Prediction Date", max_date, min_value=min_date)
    
    st.divider()
    
    df_lagged = load_lagged_data(DEFAULT_COMMODITY)
    hist_row = get_historical_features(df_lagged, market, pred_date)
    
    simulated_data = []
    if hist_row is not None:
        st.success(f"✅ Loaded historical data for {market}")
        st.dataframe(hist_row.iloc[:, :8], use_container_width=True) # Preview
        
        for lag in range(1, LAG_WINDOW + 1):
            # Extract raw features directly from the row
            simulated_data.append({
                f'{PRICE_COLUMN}_Lag1': hist_row[f'{PRICE_COLUMN}_Lag{lag}'].values[0],
                'temperature_2m_max': hist_row[f'temperature_2m_max_Lag{lag}'].values[0],
                'precipitation_sum': hist_row[f'precipitation_sum_Lag{lag}'].values[0],
                'weather_code': hist_row[f'weather_code_Lag{lag}'].values[0], # Direct value
                'temperature_2m_mean': hist_row[f'temperature_2m_mean_Lag{lag}'].values[0],
                'temperature_2m_min': hist_row[f'temperature_2m_min_Lag{lag}'].values[0],
                'wind_speed_10m_max': hist_row[f'wind_speed_10m_max_Lag{lag}'].values[0],
                'precipitation_hours': hist_row[f'precipitation_hours_Lag{lag}'].values[0],
            })
    else:
        simulated_data = manual_input_ui()

    if st.button("Predict Price", type="primary"):
        try:
            feat_names = model.get_booster().feature_names
            X_input = prepare_input_features(simulated_data, feat_names)
            X_scaled = scaler.transform(X_input)
            price = model.predict(X_scaled)[0]
            st.metric(f"Predicted {DEFAULT_COMMODITY} Price", f"₹ {price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()