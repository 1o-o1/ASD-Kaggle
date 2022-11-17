
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

df1 = pd.read_csv('Autism_Data.arff',na_values='?')
df2 = pd.read_csv('Toddler Autism dataset July 2018.csv',na_values='?')
print(df1.head())
print(df2.head(n=5))



within24_36= pd.get_dummies(df2['Age_Mons']>24,drop_first=True)
within0_12 = pd.get_dummies(df2['Age_Mons']<13,drop_first=True)
male=pd.get_dummies(df2['Sex'],drop_first=True)
ethnics=pd.get_dummies(df2['Ethnicity'],drop_first=True)
jaundice=pd.get_dummies(df2['Jaundice'],drop_first=True)
ASD_genes=pd.get_dummies(df2['Family_mem_with_ASD'],drop_first=True)
ASD_traits=pd.get_dummies(df2['Class/ASD Traits '],drop_first=True)
final_data= pd.concat([within0_12,within24_36,male,ethnics,jaundice,ASD_genes,ASD_traits],axis=1)
final_data.columns=['within0_12','within24_36','male','Latino','Native Indian','Others','Pacifica','White European','asian','black','middle eastern','mixed','south asian','jaundice','ASD_genes','ASD_traits']
print(final_data.head())
X= final_data.iloc[:,:-1]
y= final_data.iloc[:,-1]
print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim= 15))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',AUC(),TruePositives(),TrueNegatives(),FalseNegatives(),FalsePositives()])
history = model.fit(X_train, y_train,
          batch_size=256,
          epochs=100,
          validation_data=(X_test, y_test), 
          verbose=2)
plot_model(model, to_file='test1.png',show_shapes= True , show_layer_names=True)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.savefig('test1_accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('test1_loss.png')
plt.show()


model.save('test1.h5')

pd.DataFrame.from_dict(history.history).to_csv('test1.csv',index=False)