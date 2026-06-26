import datetime
import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Set up the Streamlit page
st.set_page_config(page_title="Indian Stock ARIMA Forecaster", layout="wide")
st.title("📈 Indian Stock Price Forecasting (ARIMA)")

# 1. Moving Controls to the Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_input = st.text_input("Enter Indian Stock Ticker:", value="RELIANCE").strip().upper()
    st.info("Tip: You don't need to type '.NS'. The app appends it automatically for NSE stocks.")

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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.resample('B').ffill() # Resample to business days & forward fill gaps
        
        # 3. Calculate Target Steps for June 2027
        target_date = datetime.date(2027, 6, 30)
        last_historical_date = df.index[-1].date()
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
            except Exception as e:
                st.error(f"ARIMA model optimization failed to converge: {e}")
                st.stop()

        # Filter forecast specifically up to June 2027
        forecast_series = forecast_series[forecast_series.index <= '2027-06-30']
        june_2027_forecast = forecast_series[forecast_series.index.strftime('%Y-%m') == '2027-06']
        
        # 4. KPI Performance Metrics Columns
        if not june_2027_forecast.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Current Price", value=f"₹{df['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric(label="Expected End of June 2027 Price", value=f"₹{june_2027_forecast.values[-1]:.2f}")
            with col3:
                st.metric(label="June 2027 Average Projected", value=f"₹{june_2027_forecast.values.mean():.2f}")
        
        # 5. Interactive Chart Setup (With Auto-Zoom Fix)
        fig = go.Figure()
        
        # Historical Trace
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical Price", line=dict(color="#1f77b4", width=2)))
        
        # Forecasted Trace
        fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, name="ARIMA Forecast", line=dict(color="#ff7f0e", width=2.5, dash="dash")))
        
        # Dynamic date boundaries for auto-zoom (shows last 6 months of history + full forecast)
        zoom_start_date = last_historical_date - datetime.timedelta(days=180)
        zoom_end_date = target_date + datetime.timedelta(days=15)

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Stock Price (INR)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(range=[zoom_start_date, zoom_end_date]) # Focuses graph viewport on forecast region
        )

        # 6. Custom Layout Tabs Implementation
        tab1, tab2, tab3 = st.tabs(["📊 Forecast Visualizer", "📋 June 2027 Projections", "⏳ Historical Data"])

        with tab1:
            st.subheader("ARIMA Trend Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            # Conclusion and Thoughts Section
            st.markdown("---")
            st.subheader("🔍 Model Summary & Analyst Thoughts")
            
            # Extract basic direction metrics
            last_price = df['Close'].iloc[-1]
            predicted_june_end = june_2027_forecast.values[-1]
            price_change = predicted_june_end - last_price
            pct_change = (price_change / last_price) * 100
            direction = "🔺 BULLISH" if price_change > 0 else "🔻 BEARISH"
            
            col_insight1, col_insight2 = st.columns([1, 2])
            
            with col_insight1:
                with st.container(border=True):
                    st.markdown(f"**Mathematical Bias:** `{direction}`")
                    st.markdown(f"**Est. Net Change:** `₹{price_change:,.2f}`")
                    st.markdown(f"**Est. Growth Rate:** `{pct_change:.2f}%`")
            
            with col_insight2:
                st.markdown(
                    f"""
                    **Technical Breakdown:**
                    The ARIMA(1,1,1) model evaluates the historical momentum of **{ticker}** across the last 5 years to establish a baseline trajectory. 
                    Based entirely on past moving averages, the mathematical projection estimates an expected terminal value of **₹{predicted_june_end:,.2f}** by late June 2027.
                    
                    **Why does the prediction line look relatively straight?**
                    * **Mean Reversion & Convergence:** ARIMA models are linear equations. When tasked with forecasting out more than a few weeks or months into the future, the moving average components exhaust their lag memories and naturally smooth out into a long-term linear mean drift.
                    * **The Volatility Blind Spot:** This model assumes that the variance of historical errors remains constant. It cannot see or guess macro economic indicators, company earnings transcripts, or corporate actions (like stock splits or dividends) that cause non-linear price spikes in Indian equity markets.
                    
                    **Conclusion:** Treat this projection strictly as a structural trend baseline rather than a literal destination. It reveals the underlying *momentum inertia* of the stock, but should always be paired with volume analysis, fundamental indicators, or an institutional ML network (like LSTM or Transformers) for strategic decision making.
                    """
                )

        with tab2:
            st.subheader("Projected June 2027 Price Numbers")
            if not june_2027_forecast.empty:
                display_df = pd.DataFrame(june_2027_forecast).rename(columns={0: "Projected Close Price (₹)"})
                display_df.index = display_df.index.date
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No data found explicitly for the timeframe of June 2027.")

        with tab3:
            st.subheader("Last 5 Years Historical Cleaned Data")
            st.dataframe(df.tail(100), use_container_width=True)

        # 7. Additional Info Container Box at bottom
        with st.container(border=True):
            st.caption("⚠️ **Disclaimer:** This projection is purely mathematical based on a standalone ARIMA configuration. It cannot account for sudden market gaps, regulatory changes, or macroeconomic events.")
