import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet

def parse_quarterly_dates(df, date_col):
    """Converts formats like 'Q1 2015' or '2015 Q1' to standard datetime."""
    df = df.copy()
    df[date_col] = df[date_col].astype(str).str.replace('Q1', '03-31')
    df[date_col] = df[date_col].str.replace('Q2', '06-30')
    df[date_col] = df[date_col].str.replace('Q3', '09-30')
    df[date_col] = df[date_col].str.replace('Q4', '12-31')
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)

def create_time_series_features(df, target_col):
    """Engineers lags and rolling windows to feed time-awareness into XGBoost."""
    df_feat = df.copy()
    df_feat['lag_1'] = df_feat[target_col].shift(1)
    df_feat['lag_4'] = df_feat[target_col].shift(4)
    df_feat['rolling_mean_2'] = df_feat[target_col].rolling(window=2).mean()
    df_feat['rolling_std_4'] = df_feat[target_col].rolling(window=4).std()
    
    df_feat['quarter_idx'] = df_feat['Quarter'].dt.quarter
    df_feat['year'] = df_feat['Quarter'].dt.year
    return df_feat.dropna().reset_index(drop=True)

def train_and_tune_model(df, target_col, ignore_cols):
    """Trains an XGBoost model using TimeSeriesSplit to prevent data leakage."""
    X = df.drop(columns=[target_col] + ignore_cols)
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Chronological Split (last 20% for testing)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=tscv, 
        scoring='neg_mean_absolute_error', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"XGBoost Test RMSE: {rmse:.2f} | Test MAE: {mae:.2f}")
    return best_model

def forecast_with_prophet(df, date_col, target_col, periods=8):
    """Uses Facebook Prophet for long-term trend and seasonality forecasting."""
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=periods, freq='Q')
    forecast = model.predict(future)
    
    fig = model.plot(forecast)
    plt.title('Prophet Forecast for Deliveries')
    plt.savefig('prophet_forecast.png')
    print("Prophet forecast visualization saved to 'prophet_forecast.png'")
    return forecast

def main():
    data_path = "tesla_deliveries.csv"
    
    # Generate dummy data if the Kaggle CSV isn't in the same directory
    if not os.path.exists(data_path):
        print(f"Dataset '{data_path}' not found. Generating a dummy dataset for demonstration...")
        dates = [f"Q{i%4+1} {2015 + i//4}" for i in range(40)]
        deliveries = [10000 + i*5000 + (i%4)*2000 for i in range(40)]
        production = [d + 1000 for d in deliveries]
        pd.DataFrame({'Quarter': dates, 'Total_Deliveries': deliveries, 'Total_Production': production}).to_csv(data_path, index=False)
        
    print("1. Loading and parsing dates...")
    df = pd.read_csv(data_path)
    df = parse_quarterly_dates(df, 'Quarter')
    
    print("2. Engineering time-series features (Lags & Rolling Windows)...")
    df_feat = create_time_series_features(df, 'Total_Deliveries')
    
    print("3. Training and Tuning XGBoost Model...")
    best_xgb = train_and_tune_model(df_feat, 'Total_Deliveries', ignore_cols=['Quarter', 'Total_Production'])
    
    print("4. Running Prophet Forecasting...")
    forecast = forecast_with_prophet(df, 'Quarter', 'Total_Deliveries', periods=8)
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
