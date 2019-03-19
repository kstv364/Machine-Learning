import pandas as pd
import numpy as np
import math
import random

def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

def phi(x):
    return 1/(1 + math.exp(-x))

def dphi(x):
    return phi(x)*(1-phi(x))

df = pd.read_csv('data.csv')
#print(df)

np.random.seed(42)
df = df.reindex(np.random.permutation(df.index))
df.reset_index(inplace=True, drop=True)


x,y = to_xy(df,'2')



d = len(x[0])
C = len(y[0])
h = (d + C)//2

v1 = [0]*(h+1)
u1 = [0]*(h+1)
v2 = [0]*(C+1)
u2 = [0]*(C+1)

W1 = [[ random.random() for _ in range(d+1)] for _ in range(h+1)]
W2 = [[ random.random() for _ in range(h+1)] for _ in range(C+1)]

print(W1)
print(W2)

bias = 1

def calculateU1nj(n,j):
    u = 0
    for i in range(1,d+1):
        u += x[n-1][i-1]*W1[j][i]
    u+= bias*W1[j][0]
    return u

def calculateU2nk(n,k):
    u = 0
    for j in range(1,h+1):
        u += v1[j]*W2[k][j]
    u+= bias*W2[k][0]
    return u

eta = 0.01

f = open('Results.txt','w')

def fit(x,y,eta,x_test,y_test,epochs=100): #function to train Neural Net
    c = 0
    N = len(x)
    prevE = 0
    E = 0
    m = 0
    while(epochs>0):

        #eta = eta * (1/(1 + 0.00001*c)) #decrease learning rate
        
        c+=1
        print("Epochs : " ,c)
        f.write("Epoch : \t{}".format(c))
        prevE = E
        E = 0
        for n in range(1,N+1):
            for j in range(1,h+1):
                u1[j] = calculateU1nj(n,j)
                v1[j] = phi(u1[j])
            for k in range(1,C+1):
                u2[k] = calculateU2nk(n,k)
                v2[k] = phi(u2[k])
            e = 0
            for k in range(1,C+1):
                e += (y[n-1][k-1] - v2[k])**2
            
            e/=2

            
            E+=e

            #backpropagation 2nd set of weights
            for k in range(1,C+1):
                for j in range(1,h+1):
                    W2[k][j] = W2[k][j] + eta*(y[n-1][k-1] - v2[k])*dphi(u2[k])*v1[j]
                W2[k][0] = W2[k][0] + eta*(y[n-1][k-1] - v2[k])*dphi(u2[k])*bias

            #backpropagation 1st set of weights
            for j in range(1,h+1):
                for i in range(1,d+1):
                    W1[j][i] = W1[j][i] + eta*sum([(y[n-1][k-1] - v1[k])*dphi(u2[k])*W2[k][j]*dphi(u1[j])*x[n-1][i-1] for k in range(1,C+1)])
                W1[j][0] = W1[j][0] + eta*sum([(y[n-1][k-1] - v1[k])*dphi(u2[k])*W2[k][j]*dphi(u1[j])*bias for k in range(1,C+1)])
        #epochs-=1
        print("Fold ({}) : Error : {}\n".format(fold,E))
        f.write("Error : {}\n".format(E))

        #breaking conditions
        acc = predict(x_test,y_test)


        if(acc==1.0 or(acc>0.9 and c>2000) or c>3000):
            break


        if(E>prevE):
            m+=1

        if(m>100): #tolerance value

            print("Stuck in local minima")
            f.write("Stuck in local minima\n")
            break

        
        

def predict(x,y):
    N = len(x)
    v11 = [0]*(h+1)
    u11 = [0]*(h+1)
    v22 = [0]*(C+1)
    u22 = [0]*(C+1)
    E = 0
    print("predict : ")
    for n in range(1,N+1):
        for j in range(1,h+1):
            u11[j] = sum([ x[n-1][i-1]*W1[j][i] for i in range(1,d+1)]) + bias*W1[j][0]
            v11[j] = phi(u11[j])
        for k in range(1,C+1):
            u22[k] = sum([v11[j]*W2[k][j] for j in range(1,h+1)]) + bias*W2[k][0]
            v22[k] = phi(u22[k])
        

        e = ((np.argmax(y[n-1])) == np.argmax(v22[1:])).astype(int)
        E += e
    E/=N

    return E



from sklearn.model_selection import KFold

kf = KFold(5)
fold = 0
for train, test in kf.split(x):
    fold+=1
    print("Fold #{}\t:".format(fold))
    f.write("Fold #{}\t:\n\n".format(fold))
        
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    fit(x_train,y_train,eta,x_test,y_test)
    E = predict(x_test,y_test)

    print("Accuracy : {}".format(E))
    f.write("\nWeights\t:\n")
    f.write("{}\n{}\n\n".format(W1,W2))
    f.write("Accuracy : {}\n".format(E))
f.close()