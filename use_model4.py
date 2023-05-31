import tkinter as tk
import pandas as pd
import tensorflow as tf
import joblib
import os

def load_model_and_scaler():
    
    if os.name == 'nt':
        model = tf.keras.models.load_model('saved_models\\model4.h5')
        scaler = joblib.load('saved_models\\scaler4.pkl')
    if os.name == 'posix':
        model = tf.keras.models.load_model('saved_models//model4.h5')
        scaler = joblib.load('saved_models//scaler4.pkl')

    return model, scaler

def predict_churn_probability(model, scaler, age, income, usage, satisfaction, postcode):
    if os.name == 'posix': # For Unix systems
        postcode_csv = 'zipcode_data/db_postcodes.csv'
    elif os.name == 'nt':   # For Windows
        postcode_csv = 'zipcode_data\\db_postcodes.csv'
    postcode_query = pd.read_csv(postcode_csv, sep=';', decimal=',').query(f'Postcode == {postcode}')
    postcode_competitors = postcode_query.iloc[0]['Competitors']

    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'usage': [usage],
        'satisfaction': [satisfaction],
        'competitors' : [postcode_competitors]
    })
    print(input_data)

    
    input_data = scaler.transform(input_data)

    
    churn_probability = model.predict(input_data)[0][0]

    return churn_probability

def run_model():
    model, scaler = load_model_and_scaler()
    age = age_var.get()
    income = income_var.get()
    usage = usage_var.get()
    satisfaction = satisfaction_var.get()
    postcode = postcode_var.get()

    churn_probability = predict_churn_probability(model, scaler, age, income, usage, satisfaction, postcode)
    churn_probability_var.set(f'{churn_probability:.2f}')


window = tk.Tk()
window.title("Churn Prediction App")


window.geometry("450x320")


inputs_frame = tk.Frame(window)
inputs_frame.pack(pady=20)


age_label = tk.Label(inputs_frame, text="Age:")
age_label.grid(row=0, column=0, padx=5, pady=5)

age_var = tk.StringVar()
age_entry = tk.Entry(inputs_frame, textvariable=age_var)
age_entry.grid(row=0, column=1, padx=5, pady=5)

income_label = tk.Label(inputs_frame, text="Monthly Income in €:")
income_label.grid(row=1, column=0, padx=5, pady=5)

income_var = tk.StringVar()
income_entry = tk.Entry(inputs_frame, textvariable=income_var)
income_entry.grid(row=1, column=1, padx=5, pady=5)

usage_label = tk.Label(inputs_frame, text="Usage (1-100):")
usage_label.grid(row=2, column=0, padx=5, pady=5)

usage_var = tk.StringVar()
usage_entry = tk.Entry(inputs_frame, textvariable=usage_var)
usage_entry.grid(row=2, column=1, padx=5, pady=5)

satisfaction_label = tk.Label(inputs_frame, text="Satisfaction (1-10):")
satisfaction_label.grid(row=3, column=0, padx=5, pady=5)

satisfaction_var = tk.StringVar()
satisfaction_entry = tk.Entry(inputs_frame, textvariable=satisfaction_var)
satisfaction_entry.grid(row=3, column=1, padx=5, pady=5)

postcode_label = tk.Label(inputs_frame, text="Postcode:")
postcode_label.grid(row=4, column=0, padx=5, pady=5)

postcode_var = tk.StringVar()
postcode_entry = tk.Entry(inputs_frame, textvariable=postcode_var)
postcode_entry.grid(row=4, column=1, padx=5, pady=5)


run_button = tk.Button(window, text="Predict Churn Probability", command=run_model)
run_button.pack(pady=10)


churn_probability_var = tk.StringVar()
churn_probability_label = tk.Label(window, textvariable=churn_probability_var, font=("Arial", 24))
churn_probability_label.pack(pady=20)

window.mainloop()
