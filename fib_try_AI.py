# Random Noise Training
from NNDeep import DeepNN
import numpy as np

X = np.array([0,1,2,4,5,6,7,8]).reshape(8,1)
y = np.array([0,1,2,24,120,720,5040,40320]).reshape(8,1)

NN = DeepNN([1,100,100,1])
NN.learn(30000,1e-6,X,y,['tanh','identity','identity'], 'least_squares')
print(NN.think(X))
