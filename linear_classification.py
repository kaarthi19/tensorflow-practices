# -*- coding: utf-8 -*-
"""Linear Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pzWwM1eK950-NlijJzaT0mUM-Uu67npG
"""

!pip install -q tensorflow==2.1.0
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

type(data)

data.keys()

data.data.shape

data.target

data.target_names

data.target.shape

data.feature_names

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D= X_train.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(D,)),
 tf.keras.layers.Dense(1, activation='sigmoid')                                  
])
#build model
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

#train model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

#model evaluation

print("Train Score", model.evaluate(X_train, y_train))
print("Test Score", model.evaluate(X_test, y_test))

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend

plt.plot(r.history['accuracy'], label=['acc'])
plt.plot(r.history['val_accuracy'], label=['val_acc'])
plt.legend

"""Part 2 making predictions"""

P = model.predict(X_test)
print(P)

import numpy as np
P = np.round(P).flatten()
print(P)

print("Manually calculated accuracy: ", np.mean(P == y_test))
print("Evaluate output: ", model.evaluate(X_test, y_test))

"""Part 3 saving and loading the model"""

model.save('linearclassifier.h5')

!ls -lh

model = tf.keras.models.load_model('linearclassifier.h5')
print(model.layers)
model.evaluate(X_test,y_test)