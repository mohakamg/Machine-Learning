# Import Liabraries
import numpy as np # Linear Algebra Matrices
import pandas as pd # Dataframe Creation Manipulation
import matplotlib.pyplot as plt # Plotting
from NNDeep import DeepNN

from sklearn import preprocessing

# Import the Data
dataset = pd.read_csv('winequality-red.csv')
dataset['intercept'] = 1 # Add One's to the Dataframe

# Features
X_features = ['intercept', 'residual sugar']
y_features = ['alcohol']

# For Polynomial Fitting
degree = 1
for i in range(2,degree+1):
    dataset[X_features[1] + '^'+str(i)] = dataset[X_features[1]]**degree

dataset.head(5)

# plt.scatter(dataset[X_features[1]],dataset[y_features[0]])
# plt.show()

X = dataset[X_features]
y = dataset[y_features]

X = np.array(X)
y = np.array(y).reshape(len(y),1)

max_abs_scaler = preprocessing.MaxAbsScaler()
X = max_abs_scaler.fit_transform(X)


NN = DeepNN([1,50,100,50,1])
# NN.learn(2000,1e-5,X[:,1:],y,['tanh','identity','identity'], 'least_squares')
NN.learn(1000,1e-6,X[:,1:],y,['tanh','identity','identity','identity'], 'least_squares', 100, 'sgd', 100)

print(NN.think(X[:,1:]))
