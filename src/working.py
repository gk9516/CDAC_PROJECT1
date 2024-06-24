import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from getModel import getTrainedModel
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
def encode_data(data):
    label_encoder_app = LabelEncoder()
    encoded_data = label_encoder_app.fit_transform(data.iloc[:, 0:1])
    encoded_data = pd.DataFrame(data=encoded_data)
    return encoded_data, label_encoder_app

# Load the data and model
data = pd.read_csv('Dataset/dataset.csv')
data = clean_data(data)
encoded_data, label_encoder_app = encode_data(data)
total_dataset = encoded_data.iloc[:, 0:1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(total_dataset)

X_test = []
for i in range(10, len(scaled_data)):
    X_test.append(scaled_data[i-10:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

test_set = encoded_data.iloc[10:, 0:1].values.flatten()

RNNModel = getTrainedModel()
predicted_app = RNNModel.predict(X_test)
idx = (-predicted_app).argsort(axis=1)

# Constructing prediction DataFrame
prediction = pd.DataFrame(label_encoder_app.inverse_transform(idx[:, 0]))
for i in range(1, 4):
    prediction = pd.concat([prediction, pd.DataFrame(label_encoder_app.inverse_transform(idx[:, i]))], axis=1)

# Actual apps used
actual_app_used = pd.DataFrame(label_encoder_app.inverse_transform(test_set))

# Combine predictions and actual values
final_outcome = pd.concat([prediction, actual_app_used], axis=1)
final_outcome.columns = ['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4', 'Actual App Used']
print('***********************************FINAL PREDICTION*********************************')
print(final_outcome)
