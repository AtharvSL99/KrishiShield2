import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta, date
import os
import xgboost as xgb
import requests
from io import StringIO

# --- Configuration ---
PRIMARY_DATA_FILE = 'semifinal.csv'
FALLBACK_DATA_FILE = 'test.csv'
MODEL_FILE = 'Onion.pkl'
SCALER_FILE = 'scaler.pkl'
LAG_WINDOW = 4
DEFAULT_COMMODITY = 'Onion'
PRICE_COLUMN = 'Modal_Price'

# --- 1. Data Loading & Setup ---

@st.cache_data
def load_base_data():
    """Loads raw data to get Market list."""
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    if not os.path.exists(file_to_load):
        st.error(f"Base data file ({file_to_load}) not found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_to_load, index_col=False)
        df.columns = df.columns.str.strip()
        
        # Basic cleanup of index columns
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '0':
            df = df.drop(columns=[df.columns[0]])
            if len(df.columns) > 0 and (df.columns[0].startswith('Unnamed') or df.columns[0] == '0'):
                df = df.drop(columns=[df.columns[0]])
                
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def get_market_coordinates():
    """
    Extracts a dictionary of {Market Name: (Latitude, Longitude)} from the PRIMARY dataset.
    Now robustly checks semifinal.csv first.
    """
    # Use the same logic as load_base_data to find the file
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    
    try:
        df = pd.read_csv(file_to_load, index_col=False)
        # Clean headers
        df.columns = df.columns.str.strip()
        
        # Normalize Market Name column for lookups (strip spaces, ensure string)
        if 'Market Name' in df.columns:
            df['Market Name'] = df['Market Name'].astype(str).str.strip()
        
        if 'latitude_x' in df.columns and 'longitude_x' in df.columns:
            # Drop duplicates to get unique market coords
            coords = df[['Market Name', 'latitude_x', 'longitude_x']].drop_duplicates().set_index('Market Name')
            
            # Convert to dictionary, ensuring keys are standard
            coord_dict = coords.to_dict('index')
            
            # Create a case-insensitive lookup map (optional helper)
            # We will just return the raw dict, but handle lookup carefully later
            return coord_dict
        else:
            st.error(f"Latitude/Longitude columns not found in {file_to_load}")
            return {}
    except Exception as e:
        st.error(f"Error loading coordinates: {e}")
        return {}

@st.cache_resource
def load_artifacts():
    """Loads model and scaler, forcing CPU for stability."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
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
        st.error("Artifacts not found.")
        return None, None

# --- 2. Real-Time API Functions ---

def fetch_live_weather(lat, lon, target_date, lookback_days=40):
    """
    Fetches daily weather history for a specific date window ending on target_date.
    Uses start_date and end_date parameters instead of past_days.
    """
    end_date_str = target_date.strftime('%Y-%m-%d')
    start_date_str = (target_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,precipitation_hours,wind_speed_10m_max",
        "timezone": "auto"
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Convert to DataFrame
        daily_data = data.get('daily', {})
        df_weather = pd.DataFrame(daily_data)
        if not df_weather.empty:
            df_weather['time'] = pd.to_datetime(df_weather['time'])
            df_weather = df_weather.set_index('time')
        
        return df_weather
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return pd.DataFrame()

def fetch_live_prices(api_key, state="MAHARASHTRA", commodity=DEFAULT_COMMODITY):
    """
    Fetches recent market prices from Data.gov.in.
    """
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070.csv"
    params = {
        "api-key": api_key,
        "format": "csv",
        "filters[state.keyword]": state,
        "filters[commodity.keyword]": commodity.upper(), 
        "limit": 5000 
    }
    
    try:
        r = requests.get(url, params=params, timeout=20) # Increased timeout
        r.raise_for_status()
        
        df = pd.read_csv(StringIO(r.text))
        df.columns = df.columns.str.strip().str.title()
        
        col_map = {
            'Arrival_Date': 'Date', 'Modal_Price': 'Modal_Price', 
            'Market': 'Market Name', 'Commodity': 'Commodity'
        }
        for col in df.columns:
            if 'Date' in col: col_map[col] = 'Date'
            if 'Modal' in col: col_map[col] = 'Modal_Price'
            
        df = df.rename(columns=col_map)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Price API Error: {e}")
        return pd.DataFrame()

# --- 3. Data Processing ---

def process_realtime_data(df_weather, df_price, market_name):
    """
    Merges daily Weather and Price data, aggregates to Weekly, and creates Lag features.
    """
    # 1. Filter Price Data for specific Market (Flexible Matching)
    if not df_price.empty:
        # Normalize API market names and selection for comparison
        api_markets = df_price['Market Name'].astype(str).str.lower().str.strip()
        target_market = str(market_name).lower().strip()
        
        # Try exact match first
        mask = api_markets == target_market
        
        # If no match, try contains (e.g., 'Pune' in 'Pune(Khadiki)')
        if not mask.any():
             mask = api_markets.str.contains(target_market, regex=False)
        
        df_market_price = df_price[mask].copy()
        
        if df_market_price.empty:
            st.warning(f"Price data for '{market_name}' not found in recent API records. Using placeholder prices.")
            dates = df_weather.index
            df_market_price = pd.DataFrame({'Date': dates, 'Modal_Price': np.nan})
        else:
            df_market_price = df_market_price.set_index('Date')
            df_market_price = df_market_price.groupby(level=0)['Modal_Price'].mean().to_frame()
    else:
        df_market_price = pd.DataFrame({'Date': df_weather.index, 'Modal_Price': np.nan}).set_index('Date')

    # 2. Merge
    df_merged = df_weather.join(df_market_price, how='left')
    
    # 3. Weekly Aggregation
    agg_rules = {
        'temperature_2m_max': 'max',
        'temperature_2m_min': 'min',
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'sum',
        'precipitation_hours': 'sum',
        'wind_speed_10m_max': 'mean',
        'weather_code': lambda x: x.mode()[0] if not x.empty else 0,
        'Modal_Price': 'mean'
    }
    
    for col in agg_rules:
        if col not in df_merged.columns:
            df_merged[col] = 0 if col != 'Modal_Price' else np.nan

    # Resample to weekly (W)
    df_weekly = df_merged.resample('W').agg(agg_rules)
    
    # Fill missing prices
    df_weekly['Modal_Price'] = df_weekly['Modal_Price'].ffill()
    if df_weekly['Modal_Price'].isnull().all():
        df_weekly['Modal_Price'] = 2000.0 
    else:
        df_weekly['Modal_Price'] = df_weekly['Modal_Price'].fillna(method='bfill')

    # 4. Create Lags (Get the last LAG_WINDOW rows)
    if len(df_weekly) < LAG_WINDOW:
        st.warning("Not enough weekly data generated from API. Padding data.")
    
    # Get the last 4 weeks leading up to the target date
    recent_weeks = df_weekly.tail(LAG_WINDOW).iloc[::-1] 
    
    simulated_data = []
    for i in range(len(recent_weeks)):
        row = recent_weeks.iloc[i]
        week_date = recent_weeks.index[i].strftime('%Y-%m-%d') # Capture the date
        
        simulated_data.append({
            'Week_Ending_Date': week_date, # Added for verification
            f'{PRICE_COLUMN}_Lag1': row['Modal_Price'],
            'temperature_2m_max': row['temperature_2m_max'],
            'precipitation_sum': row['precipitation_sum'],
            'weather_code': row['weather_code'],
            'temperature_2m_mean': row['temperature_2m_mean'],
            'temperature_2m_min': row['temperature_2m_min'],
            'wind_speed_10m_max': row['wind_speed_10m_max'],
            'precipitation_hours': row['precipitation_hours'],
        })
        
    while len(simulated_data) < LAG_WINDOW:
        simulated_data.append(simulated_data[-1])
        
    return simulated_data

def prepare_input_features(simulated_data, model_features):
    """Aligns user input with model features."""
    X_pred = pd.DataFrame(0.0, index=[0], columns=model_features)
    
    est_price = np.mean([d.get(f'{PRICE_COLUMN}_Lag1', 2500) for d in simulated_data])

    # Fill Lags
    for lag in range(1, LAG_WINDOW + 1):
        data = simulated_data[lag-1]
        for col in ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 
                   'wind_speed_10m_max', 'precipitation_sum', 'precipitation_hours', 'weather_code']:
            X_pred.loc[0, f'{col}_Lag{lag}'] = data[col]
        X_pred.loc[0, f'{PRICE_COLUMN}_Lag{lag}'] = data.get(f'{PRICE_COLUMN}_Lag1', est_price)

    # Fill Current Week (Proxy using Lag 1)
    if simulated_data:
        proxy = simulated_data[0]
        for col in ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 
                   'wind_speed_10m_max', 'precipitation_sum', 'precipitation_hours', 'weather_code']:
            if col in X_pred.columns:
                X_pred.loc[0, col] = proxy[col]
            
    return X_pred

# --- Main App ---

def main():
    st.set_page_config(page_title="Real-Time Crop Predictor", layout="wide")
    st.title(f"🌾 {DEFAULT_COMMODITY} Price Predictor (Real-Time)")
    
    df_base = load_base_data()
    model, scaler = load_artifacts()
    market_coords = get_market_coordinates() 
    
    if df_base.empty or model is None:
        st.stop()
        
    # --- Sidebar ---
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Data.gov.in API Key", type="password")
    
    # Market Selection
    markets = sorted(df_base['Market Name'].unique())
    market = st.sidebar.selectbox("Select Market", markets)
    
    # DATE SELECTION (Key for Historical Verification)
    # Set bounds to cover reasonable history + future
    min_date = date(2020, 1, 1) 
    max_date = date(2026, 12, 31)
    default_date = date.today()
    
    prediction_date = st.sidebar.date_input(
        "Target Prediction Date", 
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help="Select a date to predict prices for. If choosing a past date, the app fetches actual historical weather."
    )
    
    st.divider()
    
    # --- Logic ---
    simulated_data = []
    
    tab1, tab2 = st.tabs(["📡 Real-Time / Historical Fetch", "📝 Manual Input"])
    
    with tab1:
        st.markdown(f"**Target:** Predict price for week ending **{prediction_date}** based on prior weather.")
        
        if st.button("Fetch Data & Predict"):
            if not api_key:
                st.error("API Key required.")
            # Robust check for coordinates
            elif market not in market_coords:
                # Try trimming/cleaning keys
                clean_market = market.strip()
                found_key = None
                for k in market_coords.keys():
                    if str(k).strip() == clean_market:
                        found_key = k
                        break
                
                if found_key:
                    lat_x = market_coords[found_key]['latitude_x']
                    lon_x = market_coords[found_key]['longitude_x']
                else:
                    st.error(f"Coordinates for '{market}' not found in {PRIMARY_DATA_FILE}.")
                    st.write("Available markets (first 5):", list(market_coords.keys())[:5])
                    lat_x, lon_x = None, None
            else:
                lat_x = market_coords[market]['latitude_x']
                lon_x = market_coords[market]['longitude_x']

            if lat_x is not None:
                with st.spinner(f"Fetching weather history for {market} up to {prediction_date}..."):
                    
                    # Fetch weather ending on the selected Prediction Date
                    df_weather = fetch_live_weather(lat_x, lon_x, target_date=prediction_date)
                    df_prices = fetch_live_prices(api_key, commodity=DEFAULT_COMMODITY)
                    
                    if not df_weather.empty:
                        simulated_data = process_realtime_data(df_weather, df_prices, market)
                        st.success(f"Processed {len(simulated_data)} weeks of context data.")
                        
                        # Show Data with Dates for Verification
                        with st.expander("View Processed Data (Verify Dates)"):
                            # Convert to DF for nicer display
                            disp_df = pd.DataFrame(simulated_data)
                            # Move Date to front
                            cols = ['Week_Ending_Date'] + [c for c in disp_df.columns if c != 'Week_Ending_Date']
                            st.dataframe(disp_df[cols], use_container_width=True)
                    else:
                        st.error("Weather data fetch returned empty.")

    with tab2:
        st.write("Manual input form...")
        # (Keep manual logic if needed, omitted for brevity)

    st.divider()

    if simulated_data:
        try:
            feat_names = model.get_booster().feature_names
            X_input = prepare_input_features(simulated_data, feat_names)
            X_scaled = scaler.transform(X_input)
            price = model.predict(X_scaled)[0]
            
            st.metric(f"Predicted Price ({prediction_date})", f"₹ {price:,.2f}")
            
            # Context info
            w_code = simulated_data[0]['weather_code']
            st.caption(f"Based on Week -1 Weather Code: {w_code:.0f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()