import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
from keras.models import load_model
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


model = "btc_model.h5"
data = pd.read_csv("btc.csv")
x_test = []
y_test = []

def main():
    print("Welcome to the Bitcoin Predictor!")
    print("1. Predict Bitcoin Price")
    print("2. Exit")
    print("3. Predict by Date")
    choice = input("Enter your choice: ")
    if choice == "1":
        predict_price()
    elif choice == "2":
        exit()
    elif choice == "3":
        predict_by_date()
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
    mod = load_model(model)
    if mod is None:
        print("Model not loaded. Please check the model path.")
        return
    target = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(target)
    last200_days = target.tail(200)
    scaled = scaler.transform(last200_days)
    x_test = np.array(scaled)
    x_test = x_test.reshape(1, 200, 1)
    y_pred = mod.predict(x_test)
    result = scaler.inverse_transform(y_pred)
    print(result)
    #scaler.fit_transform(target) #Pakai seluruh data

def predict_by_date():
    date = input("Date (YYYY-MM-DD):")
    date = dt.datetime.strptime(date, "%Y-%m-%d")
    lastdate = dt.datetime(2026,7,14)
    delta = (date - lastdate).days
    scaler = MinMaxScaler(feature_range=(0,1))
    mod = load_model(model)
    target = data[['Close']]
    scaler.fit(target)
    last200_days = target.tail(200)
    transform = scaler.transform(last200_days)
    x_test = np.array(transform)
    for i in range(delta):
        y_test = x_test.reshape(1,200,1)
        y_pred = mod.predict(y_test)
        x_test = np.vstack((x_test[1:], y_pred))  #geser 1/nambah hasil prediksi ke x_test
    result = scaler.inverse_transform(y_pred)
    print(result)
    x = input("again? y/n")
    if x == "y":
        main()
    else:
        exit()

    


main()


#model.py untuk bs dipanggil utk update model, drop row pertama, error handling input tanggal dibawah today