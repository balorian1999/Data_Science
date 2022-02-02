from sklearn.linear_model import Perceptron
x=[[1,0],[1,1],[2,0]]
y=["boy","girl","boy"]
ml=Perceptron()
ml=ml.fit(x,y)
result=ml.predict([[1,0],[2,0]])
print(result)
