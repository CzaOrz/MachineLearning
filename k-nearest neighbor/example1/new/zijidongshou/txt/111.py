import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

for i in range(0,10):
    img = cv2.imread('6666_%d.png'%i, cv2.IMREAD_GRAYSCALE)
    print("cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)结果如下：")
    print('大小：{}'.format(img.shape))
    print("类型：%s"%type(img))
    print(img)
    print(type(img))
    np.savetxt('temp_%d.txt'%i,img)