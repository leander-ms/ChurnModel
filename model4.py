import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import os
import joblib


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TP_CPP_MIN_LOG_LEVEL'] = '3'

# check if GPU is available
if not tf.config.experimental.list_physical_devices('GPU'):
    print('No GPU was detected. Model will run on CPU.')
else:
    print(f'Model will run on CUDA')

def generate_customer_data(sample_size: int, noise_factor: float):
    np.random.seed(42)  

    # check if running Windows or WSL (Windows subsystem for Linux)
    if os.name == 'nt':
        csv_file = r'zipcode_data\zipcodes.csv'
    elif os.name == 'posix': 
        csv_file = 'zipcode_data/zipcodes.csv'

    zipcodes_df = pd.read_csv(csv_file, sep=';', decimal=',', 
                              dtype={'Ort': str, 'Zusatz': str, 'Plz': str, 'Vorwahl': str, 'Bundesland': str})
    distinct_postcodes = zipcodes_df['Plz'].unique().astype(str)
    
    postcode_competitors = \
        {postcode: np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.4, 0.2, 0.1, 0.1]) for postcode in distinct_postcodes}
    
    postcodes = np.random.choice(distinct_postcodes, size=sample_size)  
    competitors = np.array([postcode_competitors[postcode] for postcode in postcodes])

    postcode_db = pd.DataFrame({'Postcode': distinct_postcodes, 'Competitors': postcode_competitors.values()})
    if os.name == 'nt':
        postcode_db.to_csv('zipcode_data\\db_postcodes.csv', sep=';', decimal=',', index=False)
    elif os.name == 'posix':
        postcode_db.to_csv('zipcode_data//db_postcodes.csv', sep=';', decimal=',', index=False)

    age = np.random.randint(18, 101, size=sample_size)

    median_income = 4000  # median monthly income in Germany
    income = np.random.lognormal(mean=np.log(median_income), sigma=0.5, size=sample_size)
    income = np.clip(income, 1000, 50000)  

    usage = np.random.randint(0, 101, size=sample_size)
    satisfaction = np.random.randint(1, 11, size=sample_size)

    
    age_factor = (age - 18) / (100 - 18)
    income_factor = (income - 1000) / (50000 - 1000)
    usage_factor = usage / 100
    satisfaction_factor = satisfaction / 10
    competitors_factor = competitors / 4
    churn_probability = 0.05 * (1 - age_factor) + 0.1 * (1 - income_factor) + \
        0.2 * usage_factor + 0.35 * (1 - satisfaction_factor) + 0.3 * competitors_factor

    # add random noise
    noise = np.random.uniform(-noise_factor, noise_factor, size=sample_size)
    churn_probability += noise

    
    churn_threshold = 0.65
    churn = (churn_probability > churn_threshold).astype(int)

    data = pd.DataFrame({
        'age': age,
        'income': income,
        'usage': usage,
        'satisfaction': satisfaction,
        'postcode': postcodes,
        'competitors': competitors,
        'Churn': churn
    })

    assert(len(data) == sample_size)

    if os.name == 'nt':
        data.to_excel('Input_Data\\input_data.xlsx', sheet_name='Data', index=None)
    elif os.name == 'posix':
        data.to_excel('Input_Data//input_data.xlsx', sheet_name='Data', index=None)

    return data


def create_model(learning_rate: float, dropout_rate: float, neurons_layer1: int, neurons_layer2: int, neurons_layer3: int):
    print(f'Learning Rate: {learning_rate}, Dropout Rate: {dropout_rate}, Neurons in layer 1: {neurons_layer1}, \
          Neurons in layer 2: {neurons_layer2}, Neurons in layer 3: {neurons_layer3}')
    model = Sequential([
    # input
    Dense(neurons_layer1, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(dropout_rate),
    # second hidden
    Dense(neurons_layer2, activation='relu'),
    Dropout(dropout_rate),
    # third hidden
    Dense(neurons_layer3, activation='relu'),
    Dropout(dropout_rate),
    # out
    Dense(1, activation='sigmoid')
    ])

    # compile step
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    
    data = generate_customer_data(sample_size=10**6, noise_factor=0.1)  
    data = data.interpolate()  
    
    X = data[['age', 'income', 'usage', 'satisfaction', 'competitors']]
    y = data['Churn']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # hyperparameters for grid search
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.0001],
        'dropout_rate': [0.1, 0.2, 0.4],
        'neurons_layer1': [64, 128, 256],
        'neurons_layer2': [32, 64, 128],
        'neurons_layer3': [16, 32, 64]
    }

    model = KerasClassifier(build_fn=create_model, epochs=40, batch_size=64, verbose=1)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, y_train)

    print(f'Best score: {grid_result.best_score_}')
    print(f'Best parameters: {grid_result.best_params_}')

    # kwargs from best model of grid search
    best_model = create_model(**grid_result.best_params_)

    # prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # train model
    model4 = best_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # eval
    val_loss, val_accuracy = best_model.evaluate(X_val, y_val)
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(model4.history['loss'], label='Training Loss')
    plt.plot(model4.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if os.name == 'nt':
        plt.savefig('Plots\\tf_Training_And_Validationloss.png')
    elif os.name == 'posix': 
        plt.savefig('Plots//tf_Training_And_Validationloss.png')    
    plt.show()

    training_loss = model4.history['loss']
    validation_loss = model4.history['val_loss']

    loss_data = pd.DataFrame({
    'Training Loss': training_loss,
    'Validation Loss': validation_loss
    })

    if os.name == 'nt':
        loss_data.to_csv('stats_csv\\loss_data_m4.csv', index_label='Epoch')
    elif os.name == 'posix':
        loss_data.to_csv('stats_csv//loss_data_m4.csv', index_label='Epoch')

    
    plt.figure(figsize=(12, 6))
    plt.plot(model4.history['accuracy'], label='Training Accuracy')
    plt.plot(model4.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    training_accuracy = model4.history['accuracy']
    validation_accuracy = model4.history['val_accuracy']

    accuracy_data = pd.DataFrame({
    'Training Accuracy': training_accuracy,
    'Validation Accuracy': validation_accuracy
    })

    if os.name == 'nt':
        accuracy_data.to_csv('stats_csv\\accuracy_data_m4.csv', index_label='Epoch')
    elif os.name == 'posix':
        accuracy_data.to_csv('stats_csv//accuracy_data_m4.csv', index_label='Epoch')

    
    y_pred_val = best_model.predict(X_val)

    fpr, tpr, thresholds = roc_curve(y_val, y_pred_val)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    if os.name == 'nt':
        plt.savefig('Plots\\tf_ReceiverOperatingCharacteristic.png')
    elif os.name == 'posix': 
        plt.savefig('Plots//tf_ReceiverOperatingCharacteristic.png')    
    plt.show()

    # save the model, so it can be used in the use_model4.py file
    if os.name == 'nt':
        best_model.save('saved_models\\model4.h5')
        joblib.dump(scaler, 'saved_models\\scaler4.pkl')
    elif os.name == 'posix': 
        best_model.save('saved_models//model4.h5')
        joblib.dump(scaler, 'saved_models//scaler4.pkl')
    