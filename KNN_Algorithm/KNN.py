import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

pf = pd.DataFrame(data.data,columns=data.feature_names)
pf['target'] = data.target

x=data.data
y=data.target
x_train,x_test,y_train,y_test = train_test_split(x,y)

ked = KNeighborsClassifier(n_neighbors=6)
ked.fit(x_train,y_train)
ked.score(x_test,y_test)
y_pred = ked.predict(x_test)

dd = confusion_matrix(y_test,y_pred)
dg = accuracy_score(y_test,y_pred)
print(dd)
print("Accuracy Score=",dg)

st = np.arange(1,10)
train_data,test_data=[],[]
for k in st:
    ket = KNeighborsClassifier(n_neighbors = k)
    ket.fit(x_train,y_train)
    train_data.append(ket.score(x_train,y_train))
    test_data.append(ket.score(x_test,y_test))
   
plt.plot(st,train_data,'r',label='Training Data')
plt.plot(st,test_data,'b',label='Test Data')
plt.title('Training and Testing Score')
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
