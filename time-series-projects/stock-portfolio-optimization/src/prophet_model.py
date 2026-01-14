from prophet import Prophet
import pandas as pd

def forecast_stock(df, periods=30):
    """
    df: Prophet-ready DataFrame
    periods: forecast horizon (days)
    """
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast

def forecast_next_day(df):
    forecast = forecast_stock(df, periods=1)
    return forecast["yhat"].iloc[-1]