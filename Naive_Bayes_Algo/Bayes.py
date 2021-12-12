import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [1,2,3]].values
Y = dataset.iloc[:,-1].values
ld = LabelEncoder()
X[:,0] = ld.fit_transform(X[:,0])

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

classif = GaussianNB()
classif.fit(x_train, y_train)
y_pred  =  classif.predict(x_test)

dd= accuracy_score(y_test,y_pred)
print("Accuracy Score=", dd)
