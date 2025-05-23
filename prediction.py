import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# Load Trained model

model = load_model("model.h5")

# load the encoders and sclar
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

input_geo_endoded = onehot_encoder_geo.transform([[input_data["Geography"]]])
# print(input_geo_endoded.toarray())

# From Krish's code
geo_encoded = onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# convert input data to data frame
input_df = pd.DataFrame([input_data])
# print(input_df)

# Encode categorical variables
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# concatenation with onehot encoded
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
# print(input_df)

input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

prediction_probab = prediction[0][0]
print(prediction_probab)

if prediction_probab > 0.5:
    print("Customer is likely to churn")
else:
    print("Customer is not likely to churn")