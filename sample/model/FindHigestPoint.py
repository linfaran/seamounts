import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
class FindHighestPoint(object):
    def __init__(self,path_of_image):
        self.path_of_image=path_of_image    #读取图像地址
    def get_highest_point(self):
        img=cv.imread(self.path_of_image)
        print(img.shape)
        for i in range(img.shape[0]):
            y=img[i,:,:]
            print(y.shape)



if __name__ == '__main__':
    path = os.path.join('..', 'data', 'data_0030160180.jpg')
    FindHighestPoint=FindHighestPoint(path)
    FindHighestPoint.get_highest_point()