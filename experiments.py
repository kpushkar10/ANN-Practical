import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Preprocess the Data

# Drop Irrelavent Columns
path = 'Churn_Modelling.csv'
data = pd.read_csv(path)

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode catagorical Variable
label_encoder_gender = LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])

# OneHotEncode 'Geography' column
from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])

onehot_encoder_geo.get_feature_names_out(["Geography"])

geo_encoded_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
# print(data.head())

# Save encoders and scaler


with open('label_encoder_gender.pkl', 'wb') as file:
  pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
  pickle.dump(onehot_encoder_geo, file)


# DIVIDE DATA INTO INDEPENDENT AND DEPENDENT FEATURES
x = data.drop('Exited',axis=1)
y = data['Exited']

# SPLIT DATA INTO TRAIN AND TEST DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# SCALE THESE FEATURES
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

with open('scaler.pkl', 'wb') as file:
  pickle.dump(scaler, file)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)), # Hidden Layer 1 - connected with I/p layer
    Dense(32, activation='relu'), # Hidden layer 2
    Dense(1, activation='sigmoid') # output layer
])

print(model.summary())

# Input layer   ---> HL1 (64 Neurons)    ---> HL2 (32 Neurons)   ---> Output Layer

# Compile The model

import tensorflow
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss = tensorflow.keras.losses.BinaryCrossentropy()

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Setup Tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Setup Early Stopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100,
                    callbacks=[tensorflow_callback, early_stopping_callback])

# model.save('model.h5')
model.save("model.keras", save_format="keras")