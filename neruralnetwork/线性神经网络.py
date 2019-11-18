# import numpy as np
# import matplotlib.pyplot as plt
#
# if __name__ == '__main__':
#     X = np.array([
#         [1, 0, 3],
#         [1, 4, 3],
#         [1, 1, 1],
#         [1, 0, 2],
#     ])
#     Y = np.array([
#         [1],
#         [1],
#         [-1],
#         [-1],
#     ])
#     W = (np.random.random(([3, 1])) - 0.5) * 2  # 权值初始化-取值范围-1到1
#     lr = 0.11
#
#     res = 0
#
#     def update():  # 解决异或问题需要引入非线性输入
#         global X,Y,W,lr
#         # res = np.sign(np.dot(X, W))  # shape: (3,1)
#         res = np.dot(X, W)
#         W_C = lr * (X.T.dot(Y - res)) / int(X.shape[0])
#         W = W + W_C
#
#     for i in range(100):
#         update()
#         print(W)
#         print(i)
#         res = np.sign(np.dot(X, W))
#         if (res == Y).all():
#             print("Finished")
#             break
#
#     x1 = [3, 4]
#     y1 = [3, 3]
#     x2 = [1, 0]
#     y2 = [1, 2]
#
#     k = -W[1]/W[2]
#     d = -W[0]/W[2]
#     xdata = (0, 5)
#     plt.figure()
#     plt.plot(xdata, xdata*k+d, 'r')
#     plt.scatter(x1, y1, c='b')
#     plt.scatter(x2, y2, c='y')
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = np.array([
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ])
    Y = np.array([
        [-1],
        [1],
        [1],
        [-1],
    ])
    W = (np.random.random([6,1])-0.5)*2
    lr = 0.11
    res = 0


    def update():
        global X, Y, W, lr
        res = np.sign(np.dot(X, W))  # shape: (3,1)
        #
        W_C = lr * (X.T.dot(Y - res)) / int(X.shape[0])
        W = W + W_C
    for i in range(1000):
        update()


    x1 = [0, 1]
    y1 = [1, 0]
    x2 = [0, 1]
    y2 = [0, 1]
    def cal(x, root):
        a = W[5]
        b = W[2] + x * W[4]
        c = W[0] + x * W[1] + x * x * W[3]
        if root == 1:
            return (-b + np.sqrt(b * b - 4 * a * c))/(2*a)
        if root == 2:
            return (-b - np.sqrt(b * b - 4 * a * c))/(2*a)

    xdata = np.linspace(-1, 2)
    plt.figure()
    plt.plot(xdata, cal(xdata, 1), 'r')
    plt.plot(xdata, cal(xdata, 2), 'r')
    plt.scatter(x1, y1, c='b')
    plt.scatter(x2, y2, c='y')
    plt.show()