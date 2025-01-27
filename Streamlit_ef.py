import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Streamlit interface
st.title("Portfolio Optimization with Efficient Frontier")

# Sidebar inputs
st.sidebar.header("Portfolio Configuration")
stock_input = st.sidebar.text_area(
    "Enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOGL):",
    value="AAPL, MSFT, GOOGL, AMZN, TSLA")
stocks = [s.strip().upper() for s in stock_input.split(",")]

# Date range input
start_date = st.sidebar.date_input("Start date:", value=pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End date:", value=pd.Timestamp("2024-12-31"))

# Frequency input
frequency = st.sidebar.selectbox(
    "Select data frequency:",
    options=["daily", "weekly", "monthly"],
    index=2  # Default to "daily"
)

# Risk-free rate input
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (default = 0.02):", min_value=0.0, max_value=1.0, value=0.02
)

# Maximum weight constraint input
max_weight = st.sidebar.number_input(
    "Maximum weight per stock (0 = no constraint):", min_value=0.0, max_value=1.0, value=0.0
)
if max_weight == 0:
    max_weight = None  # No constraint

# Minimum weight constraint input
min_weight = st.sidebar.number_input(
    "Minimum weight per stock (0 = no constraint):", min_value=0.0, max_value=1.0, value=0.0
)
if min_weight == 0:
    min_weight = None  # No constraint

# Map frequency to yfinance intervals
freq_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
interval = freq_map[frequency]

# Fetch stock data
st.write("Fetching data for selected stocks...")
try:
    data = yf.download(
        tickers=stocks + ['^GSPC'],  # Add S&P 500 ticker (^GSPC)
        start=start_date,
        end=end_date,
        interval=interval,
        group_by="ticker"
    )
    close_prices = pd.DataFrame({
        ticker: data[ticker]["Close"] for ticker in stocks + ['^GSPC'] if ticker in data
    })
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Check for valid tickers
if close_prices.empty:
    st.error("No valid stock data could be fetched. Please check your tickers and try again.")
    st.stop()

# Calculate returns for the entire period
returns = close_prices.pct_change().dropna()

# Add slider for dynamic date filtering (Only for weights and efficient frontier)
date_range = st.slider(
    "Select date range:",
    min_value=returns.index.min().to_pydatetime(),
    max_value=returns.index.max().to_pydatetime(),
    value=(returns.index.min().to_pydatetime(), returns.index.max().to_pydatetime()),
    format="YYYY-MM-DD"
)

# Filter returns based on slider (This will only affect the efficient frontier and weights)
date_filtered_returns = returns.loc[date_range[0]:date_range[1]]

# Separate stock returns
stock_returns = date_filtered_returns[stocks]
sp500_returns = date_filtered_returns['^GSPC']

# Fetch Target Median Price Return and adjust mean returns
adjusted_returns = {}
for stock in stocks:
    ticker = yf.Ticker(stock)
    target_median_price = ticker.info.get('targetMedianPrice', None)
    current_price = close_prices[stock].iloc[-1]  # Τρέχουσα τιμή μετοχής (τελευταία διαθέσιμη τιμή)

    if target_median_price and current_price:
        target_return = (target_median_price / current_price) - 1
        historical_mean_return = stock_returns[stock].mean()
        if historical_mean_return != 0:  # Αποφυγή διαίρεσης με το μηδέν
            adjusted_returns[stock] = target_return / historical_mean_return
        else:
            adjusted_returns[stock] = 0
    else:
        adjusted_returns[stock] = 0

# Μετατροπή σε Series για χρήση
adjusted_mean_returns = pd.Series(adjusted_returns)

# Αντικατάσταση του mean_returns
mean_returns = adjusted_mean_returns

# Υπολογισμός της συνδιακύμανσης
cov_matrix = stock_returns.cov()

# Portfolio simulation
num_portfolios = 1000
results = np.zeros((4, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    while True:
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        if ((max_weight is None or np.all(weights <= max_weight)) and
            (min_weight is None or np.all(weights >= min_weight))):
            break
    weights_record.append(weights)
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = sharpe_ratio
    results[3, i] = i

# Identify optimal and minimum risk portfolios
max_sharpe_idx = results[2].argmax()
min_risk_idx = results[1].argmin()

optimal_weights = weights_record[int(results[3, max_sharpe_idx])]
min_risk_weights = weights_record[int(results[3, min_risk_idx])]

# Υπολογισμός του standard deviation για κάθε μετοχή
stock_volatility = np.sqrt(np.diag(cov_matrix))

# Δημιουργία DataFrames με τα πρόσθετα δεδομένα
optimal_df = pd.DataFrame({
    "Stock": stocks,
    "Optimal Weights": optimal_weights,
    "Adjusted Mean Return": adjusted_mean_returns,
    "Standard Deviation": stock_volatility,
}).sort_values(by="Optimal Weights", ascending=False)

min_risk_df = pd.DataFrame({
    "Stock": stocks,
    "Min Risk Weights": min_risk_weights,
    "Adjusted Mean Return": adjusted_mean_returns,
    "Standard Deviation": stock_volatility,
}).sort_values(by="Min Risk Weights", ascending=False)

# Εμφάνιση των πινάκων με τα πρόσθετα δεδομένα
st.subheader("Optimal Portfolio Weights")
st.table(optimal_df)

st.subheader("Minimum Risk Portfolio Weights")
st.table(min_risk_df)

# Extract results for plotting
portfolio_returns = results[0]
portfolio_risks = results[1]
sharpe_ratios = results[2]

# Plot efficient frontier
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=portfolio_risks,
    y=portfolio_returns,
    mode="markers",
    marker=dict(color=sharpe_ratios, colorscale="Viridis", size=5, showscale=True),
    text=[
        f"Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe Ratio: {s:.2f}"
        for r, v, s in zip(portfolio_returns, portfolio_risks, sharpe_ratios)
    ],
    name="Portfolios"
))

fig.add_trace(go.Scatter(
    x=[portfolio_risks[max_sharpe_idx]],
    y=[portfolio_returns[max_sharpe_idx]],
    mode="markers",
    marker=dict(color="red", size=12, symbol="star"),
    name="Optimal Portfolio",
))

fig.add_trace(go.Scatter(
    x=[portfolio_risks[min_risk_idx]],
    y=[portfolio_returns[min_risk_idx]],
    mode="markers",
    marker=dict(color="blue", size=12, symbol="star"),
    name="Minimum Risk Portfolio",
))

fig.update_layout(
    title="Efficient Frontier (Interactive)",
    xaxis_title="Volatility (Standard Deviation)",
    yaxis_title="Return",
    legend=dict(
        title="Portfolio Types",
        x=1.20,
        y=1,
        xanchor="left",
        yanchor="top",
    ),
    template="plotly_dark"
)

st.plotly_chart(fig)


# Compare optimal and minimum risk portfolios with S&P 500 (For all available period)
st.write("Comparing Optimal Portfolio and Minimum Risk Portfolio with S&P 500")

# Calculate the cumulative returns for the optimal portfolio, min risk portfolio, and S&P 500 for the entire period
optimal_cum_returns = (1 + returns[stocks].dot(optimal_weights)).cumprod()
min_risk_cum_returns = (1 + returns[stocks].dot(min_risk_weights)).cumprod()
sp500_cum_returns = (1 + returns['^GSPC']).cumprod()

# Plot the comparison
comparison_fig = go.Figure()

comparison_fig.add_trace(go.Scatter(
    x=optimal_cum_returns.index,
    y=optimal_cum_returns,
    mode="lines",
    name="Optimal Portfolio",
    line=dict(color="green"),
))

comparison_fig.add_trace(go.Scatter(
    x=min_risk_cum_returns.index,
    y=min_risk_cum_returns,
    mode="lines",
    name="Minimum Risk Portfolio",
    line=dict(color="orange"),
))

comparison_fig.add_trace(go.Scatter(
    x=sp500_cum_returns.index,
    y=sp500_cum_returns,
    mode="lines",
    name="S&P 500",
    line=dict(color="blue", dash="dot"),
))

comparison_fig.update_layout(
    title="Cumulative Returns: Optimal Portfolio vs Minimum Risk Portfolio vs S&P 500",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    template="plotly_dark"
)

st.plotly_chart(comparison_fig)
