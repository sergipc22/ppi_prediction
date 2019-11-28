import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import argparse
import keras

def normalize(data_x, data_y):
    #For features a Min-Max Scaler is used
    # Output array is normalized dividing by the maximum in the dataset
    scaler = MinMaxScaler()
    scaler.fit(data_x)
    X_norm = scaler.transform(data_x)

    max_y = max(data_y)
    y_norm = data_y / max_y


    return X_norm, y_norm, max_y


def make_train_val_data(csv_path):
    #First 6 columns are the features and 7th column is the output (ppi)
    df = pd.read_csv(csv_path)

    X = df.values[:,:6]
    y = df.values[:,6]

    return X, y

def model_nn_1(data, verbose=True):
    """NN with 1 hidden layer"""
    inp = keras.layers.Input(shape=(data.shape[1],), name='input_layer')

    x = keras.layers.Dense(3, activation='relu', name='hidden_layer_1')(inp)

    model_output = keras.layers.Dense(1, activation='linear', name='output_layer')(x)

    model = keras.models.Model(inputs=inp, outputs=model_output)
    model.compile(loss='mean_squared_error', optimizer='adam')

    if verbose is True:

        print(model.summary())

    return model

def model_nn_2(data, verbose=True):
    """NN with 2 hidden layer"""
    inp = keras.layers.Input(shape=(data.shape[1],), name='input_layer')

    x = keras.layers.Dense(4, activation='relu', name='hidden_layer_1')(inp)

    x = keras.layers.Dense(2, activation='relu', name='hidden_layer_2')(x)

    model_output = keras.layers.Dense(1, activation='linear', name='output_layer')(x)

    model = keras.models.Model(inputs=inp, outputs=model_output)
    model.compile(loss='mean_squared_error', optimizer='adam')

    if verbose is True:

        print(model.summary())

    return model

def lr_onplateau(monitor, factor, patience, min_lr):
    """Learning Rate decreasing function"""
    lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor,
                                           patience=patience, min_lr=min_lr)
    return lr 
  
def early_stopping(monitor, min_delta, mode, patience):
    """Early stopping function"""
    es = keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, min_delta = min_delta,
                                       patience=patience)
    return es

def parse_arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Required positional argument
    parser.add_argument('-m', '--model', type=str, default='svr',
                    help='Choose one of possible machine learning models')

    parser.add_argument('-tp', '--train_predict', type=str, default='train',
                    help='Choose one of possible machine learning models')
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    base = os.getcwd()
    #Read csv and separate data into X and
    csv_path = os.path.join(base, 'dataset.csv')
    X, y = make_train_val_data(csv_path)
    #Normalize data
    X_norm, y_norm, max_y = normalize(X, y)
    #Separate data into train and validation(used in Neural Networks)
    X_train, X_val, y_train, y_val = train_test_split(X_norm, y_norm, test_size=0.1, random_state=42)

    args = parse_arguments()

    if args.train_predict == 'train':
        if args.model == 'svr':

            print('SVR model\n')

            #Grid search for SVR
            parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]}
            svr = SVR()
            clf = GridSearchCV(svr, parameters, cv=10, scoring='neg_mean_squared_error')
            clf.fit(X_norm, y_norm)

            #Save model
            storing_path = os.path.join(base, 'models')
            if not os.path.exists(storing_path):
                os.makedirs(storing_path)
            filepath = os.path.join(storing_path, 'svr_model.sav')
            joblib.dump(clf, filepath)

        elif args.model == 'nn_1' or args.model == 'nn_2':
            if args.model == 'nn_1':
                print('NN 1 hidden layers model\n')
                ppi_model = model_nn_1(X_norm) #NN declaration
            elif args.model == 'nn_2':
                print('NN 2 hidden layers model\n')
                ppi_model = model_nn_2(X_norm) #NN declaration

            storing_path = os.path.join(base, 'models')
            if not os.path.exists(storing_path):
                os.makedirs(storing_path)
            if args.model == 'nn_1':
                #Callbacks for training
                save_best = keras.callbacks.ModelCheckpoint(os.path.join(storing_path,'ppi_model_1_layers.h5'),save_best_only=True, monitor = 'val_loss')
                csv_logger = keras.callbacks.CSVLogger(os.path.join(storing_path,'ppi_model_1_layers.log'))
            elif args.model == 'nn_2':
                #Callbacks for training
                save_best = keras.callbacks.ModelCheckpoint(os.path.join(storing_path,'ppi_model_2_layers.h5'),save_best_only=True, monitor = 'val_loss')
                csv_logger = keras.callbacks.CSVLogger(os.path.join(storing_path,'ppi_model_2_layers.log'))

            #Training conditions
            get_lrplateau = lr_onplateau('val_loss', 0.75, 20, 0.000001)
            es = early_stopping('val_loss', 0.00001, 'auto', 50)
            epochs = 5000
            batch_size = 32
            #Training
            history = ppi_model.fit(X_train, y_train, callbacks=[get_lrplateau, save_best, csv_logger, es],
                validation_data=(X_val, y_val), shuffle=True, batch_size=batch_size, epochs=epochs)

            

    elif args.train_predict == 'predict':
        if args.model == 'svr':
            #Load model
            filepath = os.path.join(base, 'models/svr_model.sav')
            svr_model = joblib.load(filepath)
            #Prediction
            print(svr_model.predict(X_val) * max_y)
        elif args.model == 'nn_1':
            #Load model
            filepath = os.path.join(base, 'models/ppi_model_1_layers.h5')
            nn_1_model = keras.models.load_model(filepath)
            #Prediction
            print(nn_1_model.predict(X_val) * max_y)
        elif args.model == 'nn_2':
            #Load model
            filepath = os.path.join(base, 'models/ppi_model_2_layers.h5')
            nn_2_model = keras.models.load_model(filepath)
            #Prediction
            print(nn_2_model.predict(X_val) * max_y)
    
    