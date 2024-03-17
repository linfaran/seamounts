import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

class Plot3DImage(object):
    def __init__(self, path_of_image):
        self.path_of_image = path_of_image  # Reading the image path
        self.img = cv.imread(self.path_of_image, cv.IMREAD_GRAYSCALE)  # Reading the image as grayscale

    def plot_img(self):
        if self.img is not None:
            rows, cols = self.img.shape
            # 创建x, y坐标网格
            x = np.linspace(0, cols - 1, cols)
            y = np.linspace(0, rows - 1, rows)
            x, y = np.meshgrid(x, y)
            # 创建3D图形
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # 使用灰度值作为高度z
            z = self.img
            # 绘制3D图形
            ax.plot_surface(x, y, z, cmap='gray')
            # 设置轴标签
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis (Gray Value)')
            # # 启用交互模式
            # plt.ion()
            # 显示图形
            plt.show()

if __name__ == '__main__':
    path = os.path.join('..', 'data', 'area_test4.png')
    plotter = Plot3DImage(path)
    plotter.plot_img()
