import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import LinearRegression


"""
一元线性回归
线性方程： h(x) = kx + b
代价函数： cost = ∑[(h(x)-y)**2] * 1/2m
对代价函数求导：
cost'k = ∑[(h(x)-y)*x] * 1/m
cost'b = ∑[(h(x)-y)*1] * 1/m

当我求出代价函数后，就可以之作为特征参数的修正

多元线程回归参照同上
"""

# y = kx + b
learningRate = 0.0001
k = b = 0
maxIter = 50

# 最小二乘法
def computeError(dataSetX, dataSetY, k, b, baseError=0):
    m = len(dataSetX)
    for i in range(m):
        baseError += (dataSetY[i] - (k * dataSetX[i] + b)) ** 2
    return baseError / (2 * float(m))


def compute(dataSetX, dataSetY, k, b, learningRate, maxIter):
    m = float(len(dataSetX))
    for i in range(maxIter):
        k_t = b_t = 0
        for j in range(int(m)):
            k_t += (1 / m) * ((k * dataSetX[j] + b) - dataSetY[j]) * dataSetX[j]
            b_t += (1 / m) * ((k * dataSetX[j] + b) - dataSetY[j])
        k = k - learningRate * k_t
        b = b - learningRate * b_t
    return k, b


def mulComputeError(dataSetX1, dataSetX2, dataSetY, k1, k2, b, baseError=0):
    m = len(dataSetX1)
    for i in range(m):
        baseError += (dataSetY[i] - (k1 * dataSetX1[i] + k2 * dataSetX2[i] + b)) ** 2
    return baseError / (2 * float(m))


def mulCompute(dataSetX1, dataSetX2, dataSetY, k1, k2, b, learningRate, maxIter):
    m = float(len(dataSetX1))
    for i in range(maxIter):
        k1_t = k2_t = b_t = 0
        for j in range(int(m)):
            k1_t += (1 / m) * ((k1 * dataSetX1[j] + k2 * dataSetX2[j] + b) - dataSetY[j]) * dataSetX1[j]
            k2_t += (1 / m) * ((k1 * dataSetX1[j] + k2 * dataSetX2[j] + b) - dataSetY[j]) * dataSetX2[j]
            b_t += (1 / m) * ((k1 * dataSetX1[j] + k2 * dataSetX2[j] + b) - dataSetY[j])
        k1 = k1 - learningRate * k1_t
        k2 = k2 - learningRate * k2_t
        b = b - learningRate * b_t
    return k1, k2, b




def weight(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat  # 矩阵乘法
    if np.linalg.det(xTx) == 0.0:  # 值为0表示没有逆矩阵
        return
    ws = xTx.I * xMat.T * yMat  # .I就是逆矩阵的意思，这里就是直接套公式了
    return ws



if __name__ == '__main__':
    dataSetX = [random.randint(i, i + 10) for i in range(100)]
    dataSetY = [random.randint(i, i + 10) for i in range(100)]
    k, b = compute(dataSetX, dataSetY, k, b, learningRate, maxIter)
    plt.scatter(dataSetX, dataSetY)
    plt.plot(dataSetX, dataSetY, 'b.')
    plt.plot(np.array(dataSetX), k * np.array(dataSetX) + b, 'r')
    plt.show()
    print(k, b)

    # k1 = k2 = b = 0
    # dataSetX1 = [100, 50, 100, 100, 50, 80, 75, 65, 90, 90]
    # dataSetX2 = [4, 3, 4, 2, 2, 2, 3, 4, 3, 2]
    # dataSetY = [9.3, 4.8, 8.9, 6.5, 4.2, 6.2, 7.4, 6.0, 7.6, 6.1]
    # print(mulComputeError(dataSetX1, dataSetX2, dataSetY, k1, k2, b))
    # k1, k2, b = mulCompute(dataSetX1, dataSetX2, dataSetY, k1, k2, b, learningRate, maxIter)
    # print(mulComputeError(dataSetX1, dataSetX2, dataSetY, k1, k2, b))

    # 标准方程法
    # x_data = np.array([dataSetX]).T
    # X_data = np.concatenate((np.ones((100, 1)), x_data), axis=1)
    # Y_data = np.array([dataSetY]).T
    # ws = weight(X_data, Y_data)
    # print(ws, ws[0], ws[1])
    # plt.plot(dataSetX, dataSetY, 'b.')
    # plt.plot(np.array([[0], [100]]),  np.array([[0], [100]])*ws[1] + ws[0], 'r')
    # plt.show()


"""
100, 50, 100, 100, 50, 80, 75, 65, 90, 90
4, 3, 4, 2, 2, 2, 3, 4, 3, 2
9.3, 4.8, 8.9, 6.5, 4.2, 6.2, 7.4, 6.0, 7.6, 6.1
"""
