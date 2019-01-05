import numpy as np
from sklearn import preprocessing,cross_validation,neighbors,svm
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?',-99999,inplace=True)

df.drop(['id'],1,inplace=True)

x=np.array(df.drop(['class'],1))
y=np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = svm.SVC()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)
predict = clf.predict(x_test)

errors = abs(predict-y_test)
ids = [i for i in range(1,(len(y_test)+1))]
#plt.scatter(ids,predict,s=100)
plt.scatter(ids,errors,s=20)
plt.show()
