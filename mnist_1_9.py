from NNDeep import DeepNN
import matplotlib.pyplot as plt
import sklearn
import numpy as np

# Mnist Dataset Classification

# Load Data
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target.reshape(len(X),1)

# Display Some Data
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)))
    plt.title('Training: %i\n' % label, fontsize = 10)
plt.show()

# Split into Training and Testing Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# Preprocess data
y_train_bin = np.zeros([len(y_train),10])
for i in range(len(y_train)):
    if(y_train[i] == 0):
        y_train_bin[i][0] = 1

    elif(y_train[i] == 1):
        y_train_bin[i][1] = 1

    elif(y_train[i] == 2):
        y_train_bin[i][2] = 1

    elif(y_train[i] == 3):
        y_train_bin[i][3] = 1

    elif(y_train[i] == 4):
        y_train_bin[i][4] = 1

    elif(y_train[i] == 5):
        y_train_bin[i][5] = 1

    elif(y_train[i] == 6):
        y_train_bin[i][6] = 1

    elif(y_train[i] == 7):
        y_train_bin[i][7] = 1

    elif(y_train[i] == 8):
        y_train_bin[i][8] = 1

    elif(y_train[i] == 9):
        y_train_bin[i][9] = 1
y_train = y_train_bin

y_test_bin = np.zeros([len(y_test),10])
for i in range(len(y_test)):
    if(y_test[i] == 0):
        y_test_bin[i][0] = 1

    elif(y_test[i] == 1):
        y_test_bin[i][1] = 1

    elif(y_test[i] == 2):
        y_test_bin[i][2] = 1

    elif(y_test[i] == 3):
        y_test_bin[i][3] = 1

    elif(y_test[i] == 4):
        y_test_bin[i][4] = 1

    elif(y_test[i] == 5):
        y_test_bin[i][5] = 1

    elif(y_test[i] == 6):
        y_test_bin[i][6] = 1

    elif(y_test[i] == 7):
        y_test_bin[i][7] = 1

    elif(y_test[i] == 8):
        y_test_bin[i][8] = 1

    elif(y_test[i] == 9):
        y_test_bin[i][9] = 1
y_test = y_test_bin

# Train Mnist Dataset on Training Set
NN = DeepNN([64,50,50,10])
NN.learn(10000, 0.001, x_train, y_train, ['reLU','sigmoid','sigmoid'], 'log_loss', 1000, '', 100)
print('Training Accuracy: ',np.mean(np.round(NN.think(x_train))==y_train) * 100)

# Check Using Testing Set
print('Testing Accuracy: ',np.mean(np.round(NN.think(x_test))==y_test) * 100)

import random
while(input() != 'q'):
    radnomTestImage = random.randint(0,len(y_test))
    randTestX = x_test[radnomTestImage]
    randTestY = y_test[radnomTestImage]

    pred = np.argmax(np.round(NN.think(randTestX.reshape(1,64)))[0])

    plt.imshow(randTestX.reshape(8,8))
    plt.title('Predicted Image By Neural Net: %i\n' % pred, fontsize = 10)
    plt.show()
