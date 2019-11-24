if __name__ == '__main__':
    from sklearn import svm
    # x = [[1,1], [3,3], [4,3]]
    # y = [-1, 1, 1]
    # model = svm.SVC(kernel='linear')
    # model.fit(x, y)
    # print(model.support_vectors_)

    import numpy as np
    a = np.mat(np.zeros((5,1)))
    l = np.mat([1,1,1,-1,-1]).T
    print(np.multiply(a, l))




