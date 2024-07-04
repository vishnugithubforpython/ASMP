import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
import joblib

# Load the data
data = pd.read_csv("AAPL.csv",parse_dates=['Date'], dayfirst=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Sidebar options
st.sidebar.title("Apple Stock Data Analysis")
option = st.sidebar.selectbox("Choose Analysis",
                              ["Data Overview", "Correlation Heatmap", "Price Distributions",
                               "Time Series Plot", "Rolling Statistics", "Moving Averages",
                               "Model Training and Evaluation", "ARIMA Model", "LSTM Model"])

# Data Overview
if option == "Data Overview":
    st.title("Data Overview")
    st.write(data.head())
    st.write(data.describe().T)
    st.write(data.info())

# Correlation Heatmap
if option == "Correlation Heatmap":
    st.title("Correlation Heatmap")
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot()

# Price Distributions
if option == "Price Distributions":
    st.title("Price Distributions")
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in price_columns:
        sns.distplot(data[col])
        st.pyplot()

# Time Series Plot
if option == "Time Series Plot":
    st.title("Time Series Plot")
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Open'], label='Open')
    plt.plot(data.index, data['High'], label='High')
    plt.plot(data.index, data['Low'], label='Low')
    plt.plot(data.index, data['Close'], label='Close')
    plt.title('APPLE Stock Prices (2012-2019)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Rolling Statistics
if option == "Rolling Statistics":
    st.title("Rolling Statistics")
    rolling_mean = data['Close'].rolling(window=30).mean()
    rolling_std = data['Close'].rolling(window=30).std()
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(data.index, rolling_mean, label='30-Day Rolling Mean', color='red')
    plt.plot(data.index, rolling_std, label='30-Day Rolling Std', color='black')
    plt.title('Rolling Mean and Standard Deviation of AAPL Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Moving Averages
if option == "Moving Averages":
    st.title("Moving Averages")
    plt.figure(figsize=(14, 8))
    data['Close'].plot(label='Close Price')
    data['Close'].rolling(window=20).mean().plot(label='20-Day Moving Average')
    data['Close'].rolling(window=50).mean().plot(label='50-Day Moving Average')
    plt.title('APPLE Stock Prices with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Model Training and Evaluation
if option == "Model Training and Evaluation":
    st.title("Model Training and Evaluation")

    # Interactive input for test size
    test_size = st.slider('Test Size (fraction of data)', 0.1, 0.5, 0.2)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[['Open', 'High', 'Low', 'Volume']], data['Close'], test_size=test_size, random_state=42)

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
    r2_lr = r2_score(y_test, y_pred_lr)

    st.write('Linear Regression Model:')
    st.write(f'MSE: {mse_lr:.2f}')
    st.write(f'RMSE: {rmse_lr:.2f}')
    st.write(f'MAE: {mae_lr:.2f}')
    st.write(f'MAPE: {mape_lr:.2f}%')
    st.write(f'R2 Score: {r2_lr:.2f}\n')

    # Support Vector Regression Model
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr_model = SVR(kernel='linear')
    svr_model.fit(X_train_scaled, y_train)
    y_pred_svr = svr_model.predict(X_test_scaled)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mse_svr)
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    mape_svr = np.mean(np.abs((y_test - y_pred_svr) / y_test)) * 100
    r2_svr = r2_score(y_test, y_pred_svr)

    st.write('Support Vector Regression Model:')
    st.write(f'MSE: {mse_svr:.2f}')
    st.write(f'RMSE: {rmse_svr:.2f}')
    st.write(f'MAE: {mae_svr:.2f}')
    st.write(f'MAPE: {mape_svr:.2f}%')
    st.write(f'R2 Score: {r2_svr:.2f}\n')

    # Random Forest Regression Model
    n_estimators = st.slider('Number of Estimators (Random Forest)', 50, 200, 100)
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
    r2_rf = r2_score(y_test, y_pred_rf)

    st.write('Random Forest Regression Model:')
    st.write(f'MSE: {mse_rf:.2f}')
    st.write(f'RMSE: {rmse_rf:.2f}')
    st.write(f'MAE: {mae_rf:.2f}')
    st.write(f'MAPE: {mape_rf:.2f}%')
    st.write(f'R2 Score: {r2_rf:.2f}\n')

    # Display model comparison
    models = ['Linear Regression', 'Support Vector Regression', 'Random Forest Regression']
    mse_scores = [mse_lr, mse_svr, mse_rf]
    rmse_scores = [rmse_lr, rmse_svr, rmse_rf]
    mae_scores = [mae_lr, mae_svr, mae_rf]
    mape_scores = [mape_lr, mape_svr, mape_rf]
    r2_scores = [r2_lr, r2_svr, r2_rf]

    evaluation_data = pd.DataFrame({'Model': models, 'MSE': mse_scores, 'RMSE': rmse_scores, 'MAE': mae_scores, 'MAPE': mape_scores, 'R2 Score': r2_scores})
    evaluation_data.set_index('Model', inplace=True)

    st.write(evaluation_data)

# ARIMA Model
if option == "ARIMA Model":
    st.title("ARIMA Model")

    # Interactive inputs for ARIMA model parameters
    p = st.number_input('ARIMA Order p', min_value=0, value=5)
    d = st.number_input('ARIMA Order d', min_value=0, value=1)
    q = st.number_input('ARIMA Order q', min_value=0, value=0)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Split the dataset into training and testing sets
    train_data = data_scaled[:int(0.85 * len(data_scaled))]
    test_data = data_scaled[int(0.85 * len(data_scaled)):]

    # Fit the ARIMA model
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()

    # Forecast the test data
    forecast = model_fit.forecast(steps=len(test_data))
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1))

    # Evaluate the model
    test_data_orig = scaler.inverse_transform(test_data)
    mse_arima = mean_squared_error(test_data_orig, forecast)
    st.write(f'Mean Squared Error: {mse_arima}')

    # Plot the predictions vs actual values
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[int(0.85 * len(data)):], test_data_orig, color='blue', label='Actual')
    plt.plot(data.index[int(0.85 * len(data)):], forecast, color='red', linestyle='--', label='Forecasted')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    st.pyplot()

    # Forecast the next 30 days
    future_forecast = model_fit.forecast(steps=30)
    future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1))

    # Plot the forecasted next 30 days
    future_dates = pd.date_range(start=data.index[-1], periods=31, inclusive='right')

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], color='blue', label='Historical')
    plt.plot(future_dates, future_forecast, color='red', linestyle='--', label='30 Day Forecast')
    plt.title('Apple Stock Price 30 Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    st.pyplot()

# LSTM Model
if option == "LSTM Model":
    st.title("LSTM Model")

    # Interactive input for number of epochs
    epochs = st.slider('Number of Epochs', 10, 100, 14)
    batch_size = st.slider('Batch Size', 16, 128, 32)

    # Create a new dataframe with only the 'Close' column
    data_lstm = data.filter(['Close'])
    dataset = data_lstm.values

    # Get the number of rows to train the model on
    len_train_data = int(np.ceil(len(dataset) * .95))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    trained_scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = trained_scaled_data[0:int(len_train_data), :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Create the testing data set
    test_data = trained_scaled_data[len_train_data - 60:, :]
    x_test, y_test = [], dataset[len_train_data:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluate the model
    mse_lstm = mean_squared_error(y_test, predictions)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = mean_absolute_error(y_test, predictions)
    mape_lstm = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2_lstm = r2_score(y_test, predictions)

    st.write('LSTM Model:')
    st.write(f'MSE: {mse_lstm:.2f}')
    st.write(f'RMSE: {rmse_lstm:.2f}')
    st.write(f'MAE: {mae_lstm:.2f}')
    st.write(f'MAPE: {mape_lstm:.2f}%')
    st.write(f'R2 Score: {r2_lstm:.2f}\n')

    # Plot the predictions vs actual values
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(predictions, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot()
