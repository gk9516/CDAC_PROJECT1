import os
from datetime import datetime

def saveRNNModel(RNNModel):
    current_file_path = os.path.dirname(__file__)
    currentTime = datetime.now()
    formattedTimeinString = currentTime.strftime('%d%m%Y%H%M%S')
    model_json = RNNModel.to_json()
    model_path = os.path.join(current_file_path, '../Models', formattedTimeinString)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, 'trainedModel.json'), "w") as json_file:
        json_file.write(model_json)
    RNNModel.save_weights(os.path.join(model_path, 'weights.h5'))
