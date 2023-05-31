from flask import Flask, render_template, request, session
import pandas as pd
import tensorflow as tf
import joblib
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top-secret-key'

def load_model_and_scaler():
    if os.name == 'nt':
        model = tf.keras.models.load_model('saved_models\\model4.h5')
        scaler = joblib.load('saved_models\\scaler4.pkl')
    elif os.name == 'posix':
        model = tf.keras.models.load_model('saved_models//model4.h5')
        scaler = joblib.load('saved_models//scaler4.pkl')
        
    return model, scaler

def predict_churn_probability(model, scaler, age, income, usage, satisfaction, postcode):
    if os.name == 'posix':
        postcode_csv = f'{dir_path}//zipcode_data//db_postcodes.csv'
    elif os.name == 'nt':
        postcode_csv = f'{dir_path}\\zipcode_data\\db_postcodes.csv'
    postcode_query = pd.read_csv(postcode_csv, sep=';', decimal=',', dtype={'Postcode': str, 'Competitors': int}).query(f'Postcode == "{postcode}"')
    assert(len(postcode_query == 1))
    postcode_competitors = postcode_query.iloc[0]['Competitors']

    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'usage': [usage],
        'satisfaction': [satisfaction],
        'competitors' : [postcode_competitors]
    })

    input_data = scaler.transform(input_data)
    churn_probability = model.predict(input_data)[0][0]

    return churn_probability

@app.route('/', methods=['GET', 'POST'])
def index():
    churn_probability = None

    if request.method == 'POST':
        model, scaler = load_model_and_scaler()
        age = int(request.form['age'])
        income = float(request.form['income'])
        usage = int(request.form['usage'])
        satisfaction = int(request.form['satisfaction'])
        postcode = str(request.form['postcode'])

        churn_probability = predict_churn_probability(model, scaler, age, income, usage, satisfaction, postcode)
        churn_probability = f'{churn_probability:.2f}'

        session['form_data'] = {
            'age': age,
            'income': income,
            'usage': usage,
            'satisfaction': satisfaction,
            'postcode': postcode
        }

    form_data = session.get('form_data', {})

    return render_template('index.html', churn_probability=churn_probability, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
