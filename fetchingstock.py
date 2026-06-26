import datetime
import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Set up the Streamlit page
st.set_page_config(page_title="Indian Stock ARIMA Forecaster", layout="wide")
st.title("📈 Indian Stock Price Forecasting (ARIMA)")
st.write("Fetches 5 years of historical data from Yahoo Finance and forecasts up to June 2027.")

# 1. User Input for Indian Stocks
ticker_input = st.text_input("Enter Indian Stock Ticker (e.g., RELIANCE, TCS, INFYS):", value="RELIANCE").strip().upper()

# Ensure it works with Yahoo Finance format (.NS for NSE)
if ticker_input and not (ticker_input.endswith(".NS") or ticker_input.endswith(".BO")):
    ticker = f"{ticker_input}.NS"
else:
    ticker = ticker_input

if ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        # 2. Get Last 5 Years of Data
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=5*365)
        
        data = yf.download(ticker, start=start_date, end=end_date)
        
    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please ensure it is a valid NSE/BSE symbol.")
    else:
        # Process the closing price
        df = data[['Close']].copy()
        # Handle multi-level column indexing if returned by modern yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.resample('B').ffill() # Resample to business days & forward fill gaps
        
        st.subheader(f"Historical Data Preview for {ticker}")
        st.dataframe(df.tail())

        # 3. Calculate Target Steps for June 2027
        target_date = datetime.date(2027, 6, 30)
        last_historical_date = df.index[-1].date()
        
        # Calculate how many business days to forecast
        forecast_days = len(pd.date_range(start=last_historical_date, end=target_date, freq='B'))

        with st.spinner("Training ARIMA model and generating forecast..."):
            try:
                # Basic ARIMA configuration (p=1, d=1, q=1) as a robust baseline
                model = ARIMA(df['Close'], order=(1, 1, 1))
                model_fitted = model.fit()
                
                # Forecast steps
                forecast_res = model_fitted.get_forecast(steps=forecast_days)
                forecast_index = pd.date_range(start=last_historical_date + datetime.timedelta(days=1), periods=forecast_days, freq='B')
                forecast_series = pd.Series(forecast_res.predicted_mean.values, index=forecast_index)
                
                # Confidence intervals
                conf_int = forecast_res.conf_int()
            except Exception as e:
                st.error(f"ARIMA model optimization failed to converge: {e}")
                st.stop()

        # Filter forecast specifically up to June 2027
        forecast_series = forecast_series[forecast_series.index <= '2027-06-30']
        
        # 4. Interactive Graph
        st.subheader("Forecast Visualizer")
        fig = go.Figure()
        
        # Historical Trace
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical Price", line=dict(color="#1f77b4")))
        # Forecasted Trace
        fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, name="ARIMA Forecast", line=dict(color="#ff7f0e", dash="dash")))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Stock Price (INR)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 5. Display Projected Numbers for June 2027
        st.subheader("📋 Projected June 2027 Price Numbers")
        june_2027_forecast = forecast_series[forecast_series.index.strftime('%Y-%m') == '2027-06']
        
        if not june_2027_forecast.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Expected End of June 2027 Price", 
                    value=f"₹{june_2027_forecast.values[-1]:.2f}"
                )
            with col2:
                # Show full month's matrix table
                display_df = pd.DataFrame(june_2027_forecast).rename(columns={0: "Projected Close Price (₹)"})
                display_df.index = display_df.index.date
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("Could not calculate points for June 2027. Ensure your timeframe boundaries are correctly indexed.")
