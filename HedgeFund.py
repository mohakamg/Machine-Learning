# import libraries
import numpy as np
import pandas as pd
from sklearn import svm
from NNDeep import DeepNN

df = pd.read_csv('deepanalytics_dataset.csv')


y = df['target']
X = df.drop(['data_id','period','target'],axis=1)

X = np.array(X)
y = np.array(y).reshape(len(X),1)

NN = DeepNN([88,200,200,1])
NN.learn(100000, 0.001, X, y, ['reLU','reLU','sigmoid'], 'log_loss', 1000, 'sgd', 1000)

print('Accuracy: ',np.mean(np.round(NN.think(X))==y) * 100)
