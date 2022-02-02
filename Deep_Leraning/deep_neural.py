from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris=load_iris()
#x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)


import tensorflow as tf
from tensorflow import keras
ml=keras.models.Sequential()
ml.add(keras.layers.Dense(units=2,activation='relu',input_shape=(4,)))
ml.add(keras.layers.Dense(units=3,activation='relu'))
ml.add(keras.layers.Dense(units=3,activation='sigmoid'))
ml.summary()
ml.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
ml.fit(x_train,y_train)
test_loss,test_accuracy=ml.evaluate(x_test,y_test)
