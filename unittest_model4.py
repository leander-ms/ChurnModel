import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
import unittest
from model4 import generate_customer_data 
from sklearn.model_selection import train_test_split
import os

if os.name == 'nt':
    model = load_model('saved_models\\model4.h5')
    scaler = joblib.load('saved_models\\scaler4.pkl')
elif os.name == 'posix':
    model = load_model('saved_models//model4.h5')
    scaler = joblib.load('saved_models//scaler4.pkl')

class TestCustomerChurnModel(unittest.TestCase):

    def test_generate_customer_data(self):
        sample_size = 1000
        noise_factor = 0.1
        data = generate_customer_data(sample_size, noise_factor)

        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (sample_size, 7))

    def test_data_preprocessing(self):
        data = generate_customer_data(1000, 0.1)
        X = data[['age', 'income', 'usage', 'satisfaction', 'competitors']]
        y = data['Churn']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.assertEqual(X_scaled.shape, X.shape)
        self.assertAlmostEqual(np.mean(X_scaled[:, 0]), 0, delta=0.1)
        self.assertAlmostEqual(np.std(X_scaled[:, 0]), 1, delta=0.1)

    def test_model_training(self):
        data = generate_customer_data(1000, 0.1)
        X = data[['age', 'income', 'usage', 'satisfaction', 'competitors']]
        y = data['Churn']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)

    def test_model_evaluation(self):
        data = generate_customer_data(1000, 0.1)
        X = data[['age', 'income', 'usage', 'satisfaction', 'competitors']]
        y = data['Churn']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        val_loss, val_accuracy = model.evaluate(X_val, y_val)

        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_accuracy, float)

    def test_model_load(self):
        if os.name == 'nt':
            loaded_model = load_model('saved_models\\model4.h5')
            loaded_scaler = joblib.load('saved_models\\scaler4.pkl')
        elif os.name == 'posix':
            loaded_model = load_model('saved_models//model4.h5')
            loaded_scaler = joblib.load('saved_models//scaler4.pkl')

        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(loaded_scaler)
    

if __name__ == '__main__':
    unittest.main()