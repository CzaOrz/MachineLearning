import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing


"""
h(x) = WX
Cost = -{∑[-y*log(h(x)) + (1-y)*log(1-h(x))]} / m
Cost' = 
"""


lr = 0.001
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def cost(dataX, dataY, ws):
    left = np.multiply(dataY, np.log(sigmoid(dataX * ws)))
    right = np.multiply(1 - dataY, np.log(1 - sigmoid(dataX * ws)))
    return np.sum(left + right) / -(len(dataX))

def gra(dataX, dataY):
    # dataX = preprocessing.scale(dataX)
    dataX = np.mat(dataX)
    dataY = np.mat(dataY)
    m, n = dataX.shape
    ws = np.mat(np.ones((n, 1)))
    for i in range(500):
        h = sigmoid(dataX * ws)
        ws_grad = dataX.T * (h - dataY) / m
        ws = ws - lr * ws_grad
    return ws

if __name__ == '__main__':
    x1 = [[i, random.randint(10, 40)] for i in range(100)]
    x2 = [[i, random.randint(60, 90)] for i in range(100, 200)]
    dataY = [1 for i in range(100)] + [0 for i in range(100)]
    dataX = x1 + x2
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataY = dataY[:, np.newaxis]
    # dataY = np.mat(dataY).T
    dataX = np.concatenate((np.ones((200, 1)), dataX), axis=1)
    # print(dataX.shape, dataY.shape)
    # print(dataY)
    # print(dataX)

    ws = gra(dataX, dataY)
    print(ws)

    xx1 = [i[0] for i in x1]
    yy1 = [i[1] for i in x1]
    xx2 = [i[0] for i in x2]
    yy2 = [i[1] for i in x2]
    plt.scatter(xx1, yy1, c='r')
    plt.scatter(xx2, yy2, c='b')
    x_test = [[-4], [200]]
    y_test = (-ws[0] - x_test * ws[1]) / ws[2]
    plt.plot(x_test, y_test, 'k')
    plt.show()


