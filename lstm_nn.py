import os
import argparse
import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# Define constants
TRAIN_FILE = 'pollution.csv'
MODEL_NAME = 'pollution_model.h5'
SAVE_MODEL_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent, 'trained_models')
TARGET = 'pollution'
FEATURES = ['pollution', 'dew', 'temp', 'pressure', 'w_speed', 'snow', 'rain']

# Define model building function
def build_model(input_shape):
    model = Sequential()
    
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=100))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Load data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess data
def preprocess_data(df, features, target):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM (samples, timesteps, features)
    y = df[target].values
    return X, y

# Train model and log to MLflow
def train_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    experiment_name = "Air_Pollution_Forecasting"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="Air_Pollution_Forecasting") as run:
        df = load_data(TRAIN_FILE)
        if df is None:
            return
        
        features = FEATURES
        target = TARGET
        
        X, y = preprocess_data(df, features, target)
        
        mlflow.set_tag("version", "1.0.0")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 1024)

        model = build_model((X.shape[1], X.shape[2]))  # Input shape (timesteps, features)
        mlflow.keras.autolog()
        
        model.fit(X, y, epochs=50, batch_size=1024, verbose=1)
        
        mlflow.log_artifact(TRAIN_FILE)
        
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        
        model_path = os.path.join(SAVE_MODEL_PATH, MODEL_NAME)
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        # Evaluate and plot predictions
        plot_predictions_and_evaluate(y, predictions)

        mlflow.end_run()

# Make predictions
def make_prediction(input_data, keras_model):
    data = pd.DataFrame(input_data)
    
    if not set(FEATURES).issubset(data.columns):
        raise ValueError("Input data must contain the following features: " + ", ".join(FEATURES))
    
    data = data[FEATURES].values
    data = data.reshape((data.shape[0], 1, data.shape[1]))  # Reshape to (samples, timesteps, features)
    
    prediction = keras_model.predict(data)
    
    return prediction

# Plot predictions and evaluate
def plot_predictions_and_evaluate(actual_values, predicted_values):
    plt.figure(figsize=(10,6))
    plt.plot(predicted_values[:100], color='green', label='Predicted Pollution Level')
    plt.plot(actual_values[:100], color='red', label='Actual Pollution Level')
    plt.title("Air Pollution Prediction (Multivariate)")
    plt.xlabel("Sample Index")
    plt.ylabel("Pollution Level")
    plt.legend()
    # plt.savefig('graph.png')
    # plt.show()
    plt.savefig('AirPollution_Graph.png')  # Save plot to file
    plt.close()  # Close the plot to avoid display on local system
        
    mlflow.log_artifact('AirPollution_Graph.png') 

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-8  # Small constant to avoid division by zero
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    mape = mean_absolute_percentage_error(actual_values, predicted_values)
    print('MAPE:', mape)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    print('RMSE:', rmse)
    print("Mean of Actual Values:", np.mean(actual_values))

def main():
    train_model()  # Always train when running the script

if __name__ == "__main__":
    main()
