# Random Noise Training
from NNDeep import DeepNN
import matplotlib.pyplot as plt
from sklearn import datasets as dt
import numpy as np


np.random.seed(0)
X, y = dt.make_moons(100, noise=0.20)
y = y.reshape(len(X),1)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()
NN = DeepNN([2,50,50,1])

NN.learn(30000, 0.001, X, y, ['reLU','reLU','sigmoid'], 'log_loss')
print('Accuracy: ',np.mean(np.round(NN.think(X))==y) * 100)

print('Weights: ', NN.weights)
