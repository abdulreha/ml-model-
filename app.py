

# thread = threading.Thread(target=run_app)
# thread.start()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import math
import json

def forecast_electricity_demand(
    data_file,
    target_state,
    seq_length=12,
    split_ratio=0.8,
    lstm_units1=100,
    lstm_units2=50,
    dropout_rate=0.2,
    epochs=100,
    batch_size=32
):
    # Set seed
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and prepare data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    if target_state not in df.columns:
        raise ValueError(f"Selected state '{target_state}' not found in dataset.")

    data = df[[target_state]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, seq_length)
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build LSTM model
    model = Sequential([
        LSTM(lstm_units1, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units2),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_lstm_model.h5', save_best_only=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, model_checkpoint],
        verbose=0
    )

    # Predictions
    train_pred = model.predict(X_train).flatten()
    test_pred = model.predict(X_test).flatten()

    train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Evaluation
    train_rmse = math.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
    test_rmse = math.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
    train_mape = mean_absolute_percentage_error(y_train_inv, train_pred_inv) * 100
    test_mape = mean_absolute_percentage_error(y_test_inv, test_pred_inv) * 100

    # Forecast next 12 months
    last_seq = scaled_data[-seq_length:]
    future_preds = []
    for _ in range(12):
        pred_input = last_seq.reshape(1, seq_length, 1)
        pred = model.predict(pred_input)[0][0]
        future_preds.append(pred)
        last_seq = np.vstack([last_seq[1:], [[pred]]])

    future_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Demand_GWh': future_inv
    })

    # Extract year and month before converting to string
    forecast_df['Year'] = forecast_df['Date'].dt.year
    forecast_df['Month'] = forecast_df['Date'].dt.month_name()

    # Convert Date to string for JSON serialization
    forecast_df['Date'] = forecast_df['Date'].astype(str)

    monthly_avg = forecast_df.groupby('Month')['Forecasted_Demand_GWh'].mean().to_dict()
    yearly_avg = forecast_df.groupby('Year')['Forecasted_Demand_GWh'].mean().to_dict()

    # Build final JSON result
    result = {
        "state": target_state,
        "train_rmse": round(train_rmse, 2),
        "test_rmse": round(test_rmse, 2),
        "train_mape": round(train_mape, 2),
        "test_mape": round(test_mape, 2),
        "forecast": forecast_df.to_dict(orient='records'),
        "monthly_avg_forecast": monthly_avg,
        "yearly_avg_forecast": yearly_avg
    }

    return result



from flask import Flask, request, jsonify
from flask_cors import CORS
##from pyngrok import ngrok
#import threading

app = Flask(__name__)
CORS(app)

@app.route('/',methods=['POST','GET'])
def hello():
    return 'Hello World'

# @app.route('/predicts', methods=['POST', 'GET'])
# def predicts():
#     #data = request.get_json()

#     # For demo purposes, replace model prediction with a mock
#     #input_data = data.get('input_data', [])

#     # Dummy output (you should replace with real model prediction)
#     prediction = "sdfdf"

#     return jsonify({'prediction': prediction})

@app.route('/predict',methods=['POST','GET'])
def predict():
    # data = request.get_json()
    # input_data = data['input_data']
    # prediction = model.predict(input_data)
    # #m=main()
    result_json = forecast_electricity_demand(
                      data_file="powerdata.csv",
                      target_state="UP",
                      seq_length=12,
                      split_ratio=0.8,
                      epochs=100,
                      batch_size=32
                  )
    return jsonify(result_json)

# Expose the app to the web
# public_url = ngrok.connect(5000)
#print(" * ngrok tunnel:", public_url)

# Run the Flask app in a thread
if __name__ == '__main__':
    app.run(port=5000)
