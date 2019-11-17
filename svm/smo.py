import random
import numpy as np


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常熟
    :param toler: 容错率
    :param maxIter: 最大循环次数
    :return:
    """
    dataMatrix = np.mat(dataMatIn)  # 将数据集转化为矩阵
    labelMat = np.mat(classLabels).transpose()  # 将标签转置
    b = 0
    m, n = dataMatrix.shape  # 矩阵的shape
    alphas = np.mat(np.zeros((m, 1)))  # 创建m行1列的0矩阵
    iter = 0  # iter是循环次数的意思吗
    while (iter < maxIter):
        alphaPairsChanged = 0  # 用于记录alphas是否已经进行优化
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 我们预测的类别
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei) < -toler) and (alphas[i] < C) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选择第二个alpha
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):  # if-else 用于保证alpha在0-C之间
                    L = max(0, alphas[j] - alphas[i])  # L和H用于将alpha[j]调整到0和C之间
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:  # 若LH相等，则不做处理
                    print("L == H")
                    continue
                # eta表示alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("J not moving enough")
                    continue
                # 对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print(f"iter: {iter}i:{i}, pairs changed {alphaPairsChanged}")
            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
            print(f"iteration number: {iter}")
        return b, alphas


if __name__ == '__main__':
    # dataS = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
    # dataS = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, 0], [1, 2], [1, 2], [1, 1]]
    # label = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    # b, alphas = smoSimple(dataS, label, 0.6, 0.001, 40)
    # # print(b)
    # # print(alphas)
    # for i in range(len(label)):
    #     if alphas[i] > 0:
    #         print(dataS[i], label[i])

    a = np.mat(np.array([1,2,3,4,5]))
    b = np.mat(np.array([1,2]))
    print(a)
    print(b.T)
    # print(a*b.T)
    # print(np.multiply(a,b))
    print(np.multiply(a,b.T))
    print(np.multiply(b.T, a))
