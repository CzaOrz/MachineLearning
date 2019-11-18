import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = np.array([
        [1, 3, 3],
        [1, 4, 3],
        [1, 1, 1],
        [1, 0, 2],
    ])
    Y = np.array([
        [1],
        [1],
        [-1],
        [-1],
    ])
    W = (np.random.random(([3, 1])) - 0.5) * 2  # 权值初始化-取值范围-1到1
    lr = 0.11

    res = 0

    def update():
        # 感知器学习规则
        global X,Y,W,lr
        res = np.sign(np.dot(X, W))  # shape: (3,1)
        #
        W_C = lr * (X.T.dot(Y - res)) / int(X.shape[0])
        W = W + W_C

    for i in range(100):
        update()
        print(W)
        print(i)
        res = np.sign(np.dot(X, W))
        if (res == Y).all():
            print("Finished")
            break

    x1 = [3, 4]
    y1 = [3, 3]
    x2 = [1, 0]
    y2 = [1, 2]

    k = -W[1]/W[2]
    d = -W[0]/W[2]
    xdata = (0, 5)
    plt.figure()
    plt.plot(xdata, xdata*k+d, 'r')
    plt.scatter(x1, y1, c='b')
    plt.scatter(x2, y2, c='y')
    plt.show()
