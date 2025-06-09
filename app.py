import streamlit as st
import yfinance as yf
from model import predict_with_lr, predict_with_arima
from indicators import add_indicators
import plotly.graph_objects as go
from datetime import date

# ------------------- Page Config -------------------
st.set_page_config("📈 Stock Price Predictor Dashboard", layout="wide", page_icon="📊")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Stock Price Prediction Dashboard")
st.markdown("""
    Use this dashboard to explore stock prices, technical indicators, and forecast future stock movements 
    using **Linear Regression** or **ARIMA** models. 📈📉
""")

# ------------------- Inputs -------------------
st.sidebar.header("🛠️ Configuration")
ticker = st.sidebar.selectbox("Choose a Stock Ticker", ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "RELIANCE.NS"])
start = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end = st.sidebar.date_input("End Date", date.today())
model_choice = st.sidebar.radio("Choose Prediction Model", ["Linear Regression", "ARIMA"])

# ------------------- On Click -------------------
if st.sidebar.button("🚀 Fetch & Predict"):
    data = yf.download(ticker, start=start, end=end)

    # 🔧 Flatten multi-index columns if needed
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    if data.empty:
        st.error("⚠️ Ticker not found or no data available for the selected range.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Historical Stock Prices")
            st.line_chart(data['Close'], use_container_width=True)

            st.subheader("📐 Technical Indicators (MA20 & MA50)")
            df_ind = add_indicators(data.copy())
            df_ind.columns = [col[0] if isinstance(col, tuple) else col for col in df_ind.columns]
            st.line_chart(df_ind[['Close', 'MA_20', 'MA_50']], use_container_width=True)

        # 💱 Convert USD to INR (approx conversion rate, can be replaced with real-time API)
        USD_TO_INR = 83.0
        low_price_inr = data['Close'].min() * USD_TO_INR
        high_price_inr = data['Close'].max() * USD_TO_INR

        with col2:
            st.metric("📊 Total Trading Days", len(data))
            st.metric("📉 Lowest Price (₹)", f"₹{low_price_inr:.2f}")
            st.metric("📈 Highest Price (₹)", f"₹{high_price_inr:.2f}")

        # 🔮 Prediction Results
        st.subheader(f"🔮 Price Prediction using {model_choice}")
        if model_choice == "Linear Regression":
            pred_df = predict_with_lr(data)
        else:
            pred_df = predict_with_arima(data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Actual"], name="Actual", line=dict(color="#10b981")))
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Predicted"], name="Predicted", line=dict(color="#f97316")))
        fig.update_layout(template="plotly_dark", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Prediction Completed Successfully!")
