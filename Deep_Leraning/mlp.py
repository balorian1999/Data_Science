from sklearn.neural_network import MLPClassifier
x=[[1,0],[1,1],[2,0]]
y=["boy","girl","boy"]
ml=MLPClassifier(hidden_layer_sizes=(10),activation='relu')
ml=ml.fit(x,y)
result=ml.predict([[1,1],[1,1]])
print(result)
