import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_pickle("AgesAndHeights.pkl")
df=data[data['Age']>0]  
x,y = df[['Age']].values,df['Height'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=0)
rd = LinearRegression()
rd.fit(x_train,y_train)

plt.scatter(x_test,y_test,c='red',label="AgesAndHeights")
plt.plot(x_test,rd.predict(x_test),c='blue')
plt.title("Age v/s Height")
plt.xlabel("Ages[Years]")
plt.ylabel("Heights[Inches]")
plt.legend()
plt.show()

from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('startups.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, 4].values
cd = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(cd.fit_transform(X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_Pred = reg.predict(X_test)
print("coef of Determination = ",r2_score(Y_test,Y_Pred))
