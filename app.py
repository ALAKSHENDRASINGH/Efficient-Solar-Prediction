import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    solar_url = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
    df = pd.read_csv(solar_url)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Solar_Lag1'] = df['Solar'].shift(1)
    df['Solar_Lag2'] = df['Solar'].shift(2)
    df['Solar_Rolling7'] = df['Solar'].rolling(window=7).mean()
    return df.dropna()

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_year = st.sidebar.selectbox("Select Year", df["Year"].unique())

# Filtered Data
filtered_df = df[df["Year"] == selected_year]

# Display Data
st.title("Solar Power Generation Analysis")
st.write(f"Showing data for {selected_year}")
st.write(filtered_df.head())

# Plot Solar Generation Trends
st.subheader("Solar Generation Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_df["Date"], filtered_df["Solar"], label="Daily Solar Generation")
ax.plot(filtered_df["Date"], filtered_df["Solar_Rolling7"], label="7-Day Average", linestyle="--")
ax.set_title("Solar Power Generation Trend")
ax.set_ylabel("Solar Generation (GWh)")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig)

# Model Training
st.subheader("Train Solar Prediction Model")
features = ["Month", "DayOfYear", "Solar_Lag1", "Solar_Lag2", "Solar_Rolling7", "Consumption", "Wind"]
target = "Solar"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=1000)
model.fit(
    X_train, 
    y_train, 
    eval_set=[(X_test, y_test)], 
    eval_metric="mae",
    callbacks=[lgb.early_stopping(50)]
)

# Model Evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

st.write(f"Training MAE: {train_mae:.2f} GWh")
st.write(f"Testing MAE: {test_mae:.2f} GWh")
st.write(f"Testing R2: {test_r2:.4f}")

# Predict Next Day
st.subheader("Next Day Solar Prediction")
latest_data = df.iloc[-1][features].values.reshape(1, -1)
prediction = model.predict(latest_data)[0]
actual = df.iloc[-1]["Solar"]

st.write(f"Predicted Solar Generation: {prediction:.2f} GWh")
st.write(f"Previous Day Actual: {actual:.2f} GWh")
st.write(f"Difference: {abs(prediction - actual):.2f} GWh")

st.sidebar.info("Use filters to explore yearly trends!")
