import random
import numpy as np

"""
主要就是求两直线x1,x2之间最近距离d
wx1 + b = 1
wx2 + b = -1
=> w(x1 - x2) = 2
=> |w| * |x1->x2| * cos' = 2
=> |w| * d = 2
=> d = 2/|w|

wx + b >= 1   label1 
wx + b <= -1  label2
=> y(wx + b) >= 1
=> max(2/|w|)
=> min(|w|**2/2)
=> Cost = |w|**2/2  求极值，转化为凸优化问题
可以分为三种：
无约束问题： - 费马定理
带等式约束问题： - 拉格朗日乘子法 -> 目标函数 + (拉格朗日乘子 * 约束条件)转化
带不等式约束问题： - KKT条件
"""
def selectJrand(i, m):  # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):  # 使用辅助函数，对L和H进行调整
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
    iter = 0  # 没有任何alpha改变的情况下遍历数据的次数
    while (iter < maxIter):
        alphaPairsChanged = 0  # 用于记录alphas是否已经进行优化
        for i in range(m):
            # 我们预测的类别 y[i] = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*label[n]*x[n]
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])  # 预测结果与真实结果比对，计算误差Ei

            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            if ((labelMat[i] * Ei) < -toler) and (alphas[i] < C) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = selectJrand(i, m)  # 随机选择第二个alpha
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
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
