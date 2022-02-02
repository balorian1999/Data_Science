import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler    
from sklearn import tree


df = pandas.read_csv("iris.csv")
print(df)

d = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
df['species'] = df['species'].map(d)
print(df)

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X = df[features]
y = df['species']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 1)
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
y_predict1 = clf.predict(x_train)
cm = confusion_matrix(y_predict,y_test) 
print("Testing accuracy = ",accuracy_score(y_test,y_predict))
print("Training accuracy = ",accuracy_score(y_train,y_predict1))


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=200)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')
