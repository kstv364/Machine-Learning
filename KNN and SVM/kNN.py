import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?',-99999,inplace=True)

df.drop(['id'],1,inplace=True)

x=np.array(df.drop(['class'],1))
y=np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
with open('KNN.pickle','wb') as f:
  pickle.dump(clf,f)

pickle_in = open('KNN.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(x_test,y_test)

print(accuracy)

predict = clf.predict(x_test)

print(predict)
print(y_test-predict)

errors = abs(y_test - predict)
print(errors)


# errors = [0]*len(predict)

# errors = [1 for i in range(len(y_test)) if y_test[i]==predict[i]]
# print(errors,len(y_test),len(predict))

ids = [i for i in range(1,(len(y_test)+1))]
plt.scatter(ids,errors)

plt.show()
