import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, InputLayer
from sklearn.preprocessing import LabelEncoder
from keras.metrics import RootMeanSquaredError as RMSE
from tcn import TCN
import torch
from sklearn.model_selection import train_test_split

TIME_STEP = 7

label_encoder = LabelEncoder()

def tensor_generator(df):

   windows = []
   Y_list = []

   for window in df.rolling(window=7):

      if len(window)  != TIME_STEP:
         continue
    
      if len(window.index.unique(level='id')) == 1:
         df_x = window.drop('mood',axis=1).to_numpy()

         Y = float(window.mood.iloc[0])
         windows.append(df_x)
         Y_list.append(Y)
    
   X_t = np.array(windows)
   Y = np.array(Y_list)
   return X_t, Y


df = pd.read_csv('cleaned\dataset_cleaned.csv', index_col=['id','time'])

input_dim = len([col for col in df.columns if col != 'mood']) 

X, y = tensor_generator(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model = Sequential()
# RNN layer
model.add(TCN(input_shape=(TIME_STEP, input_dim)))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dropout(0.5)) 
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dropout(0.5)) 
model.add(Dense(32, activation = 'sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(16, activation = 'sigmoid'))
model.add(Dropout(0.5)) 
model.add(Dense(1, activation = 'sigmoid'))
# Compile model

opt = tf.keras.optimizers.Adam(learning_rate = 0.0005)

model.compile(loss='mse' , optimizer=opt)

model.fit(x=X_train, y=y_train, epochs=2500)

results = model.evaluate(X_test, y_test)

print(results)