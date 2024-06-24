import sys
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from getModel import getTrainedModel
from saveModel import saveRNNModel

# Function to clean data
def clean_data(data):
    redundant_entries = [
        'Screen off (locked)', 'Screen on (unlocked)', 'Screen off (unlocked)', 
        'Screen on (locked)', 'Screen on', 'Screen off', 'Device shutdown', 'Device boot'
    ]
    for entry in redundant_entries:
        data = data[~(data == entry).any(axis=1)]
    data = data.dropna()
    data.index = range(len(data))
    return data

# Function to encode data
def encode_data(data):
    label_encoder_app = LabelEncoder()
    encoded_data = label_encoder_app.fit_transform(data.iloc[:, 0:1])
    encoded_data = pd.DataFrame(data=encoded_data)
    return encoded_data, label_encoder_app

# Function to create and return the RNN model
def getRNNModel(input_shape):
    RNNModel = Sequential()

    RNNModel.add(LSTM(units=200, return_sequences=True, input_shape=input_shape))
    RNNModel.add(Dropout(rate=0.3))
    RNNModel.add(BatchNormalization())

    for _ in range(2):  # Adding 2 more LSTM layers with Dropout and BatchNormalization
        RNNModel.add(LSTM(units=200, return_sequences=True))
        RNNModel.add(Dropout(rate=0.3))
        RNNModel.add(BatchNormalization())

    RNNModel.add(LSTM(units=200, return_sequences=False))  # Last LSTM layer
    RNNModel.add(Dropout(rate=0.3))
    RNNModel.add(BatchNormalization())

    RNNModel.add(Dense(units=36, activation='softmax'))
    RNNModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return RNNModel

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * 0.99

# Main script
data = pd.read_csv('Dataset\dataset.csv')
data = clean_data(data)
encoded_data, label_encoder_app = encode_data(data)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(encoded_data)

X, y = [], []
for i in range(10, len(scaled_data)):
    X.append(scaled_data[i-10:i, 0])
    y.append(encoded_data.iloc[i, 0])

X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
y = to_categorical(y, num_classes=36)

kf = KFold(n_splits=5, shuffle=True)
fold = 1
accuracies = []

for train_index, test_index in kf.split(X):
    print(f"Training fold {fold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    RNNModel = getRNNModel((X_train.shape[1], 1))
    lr_scheduler = LearningRateScheduler(scheduler)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    RNNModel.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.1, callbacks=[lr_scheduler, early_stopping])
    
    y_pred = RNNModel.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred_classes)
    accuracies.append(accuracy)
    
    print(f"Fold {fold} accuracy: {accuracy}")
    
    fold += 1

average_accuracy = np.mean(accuracies)
print(f"Average cross-validation accuracy: {average_accuracy}")

# Save the model
saveRNNModel(RNNModel)
