import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
from keras.models import load_model
import os


model = "btc_model.h5"
data = pd.read_csv("btc.csv")

def main():
    print("Welcome to the Bitcoin Predictor!")
    print("1. Predict Bitcoin Price")
    print("2. Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        predict_price()
    elif choice == "2":
        exit()
    else:
        print("Invalid choice. Please try again.")
        main()

def load_model(model_path):
    if model_path is None:
        return None
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def predict_price():
    #Set ticker + 3 for the next 3 days
    #temp = dt.datetime(2026,7,14) + dt.timedelta(days=3)
    ticker = len(data) + 3
    mod = load_model(model)
    if mod is None:
        print("Model not loaded. Please check the model path.")
        return
    y_pred = mod.predict(ticker)
    print(y_pred)

main()


