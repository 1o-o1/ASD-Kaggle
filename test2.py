
from tensorflow import keras
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import cv2
from glob import glob

asd_data = pd.read_csv('Autism_Data.arff',na_values='?')
asd_data.head(n=5)


asd_data.loc[(asd_data['age'].isnull()) |(asd_data['gender'].isnull()) |(asd_data['ethnicity'].isnull()) 
|(asd_data['jundice'].isnull())|(asd_data['austim'].isnull()) |(asd_data['contry_of_res'].isnull())
            |(asd_data['used_app_before'].isnull())|(asd_data['result'].isnull())|(asd_data['age_desc'].isnull())
            |(asd_data['relation'].isnull())]
asd_data.dropna(inplace=True)
asd_data.describe()
# Reminder of the features:
print(asd_data.dtypes)


# Total number of records in clean dataset
n_records = len(asd_data.index)

# TODO: Number of records where individual's with ASD in the clean dataset
n_asd_yes = len(asd_data[asd_data['Class/ASD'] == 'YES'])

# TODO: Number of records where individual's with no ASD in the clean dataset
n_asd_no = len(asd_data[asd_data['Class/ASD'] == 'NO'])

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals diagonised with ASD: {}".format(n_asd_yes))
print("Individuals not diagonised with ASD: {}".format(n_asd_no))
asd_raw = asd_data['Class/ASD']
features_raw = asd_data[['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'result',
                      'relation','A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score',
                      'A9_Score','A10_Score']]

scaler = MinMaxScaler()
numerical = ['age', 'result']

features_minmax_transform = pd.DataFrame(data = features_raw)
features_minmax_transform[numerical] = scaler.fit_transform(features_raw[numerical])
features_minmax_transform
# Show an example of a record with scaling applied
print(features_minmax_transform.head(n = 5))
#One-hot encode the 'features_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_minmax_transform)
X=features_final
print(features_final.head(5))

asd_classes = asd_raw.apply(lambda x: 1 if x == 'YES' else 0)
y=asd_classes

encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
print(encoded)
print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

inputs = keras.Input(shape=94, name="ANN")
out0 = Dense(256, activation="relu")(inputs)
out0 = BatchNormalization()(out0)
out0 = Dropout(0.5)(out0)
x = Dense(512, activation="relu" )(out0)
x = Dense(512, activation="relu" )(x)
x = Dropout(0.25)(x)
y = Dense(512, activation="relu" )(out0)
y = Dense(512, activation="relu" )(y)
y = Dropout(0.25)(y)
out1 = keras.layers.add([x,y])
x = Dense(1024, activation="relu")(out1)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output= Dense(1, kernel_initializer='normal', activation='sigmoid')(x)
model = keras.Model(inputs, output, name="ANN")
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy',AUC(),TruePositives(),TrueNegatives(),FalseNegatives(),FalsePositives()])
history = model.fit(X_train, y_train,
          batch_size=609,
          epochs=300,
          validation_data=(X_test, y_test), 
          verbose=2)
plot_model(model, to_file='test3.png',show_shapes= True , show_layer_names=True)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.savefig('test3_accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('test3_loss.png')
plt.show()

model.save('test3.h5')

pd.DataFrame.from_dict(history.history).to_csv('test3.csv',index=False)