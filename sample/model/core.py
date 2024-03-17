import cv2
import numpy as np
import os
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from sample.model import seamounts
import collections
path = os.path.join('..', 'data', 'resize.jpg')
#path = os.path.join('..', 'data', 'grav_32.10.png')
#path = os.path.join('..', 'data', 'retinex.png')
original_image = cv2.imread(path)
#Seamount = collections.namedtuple('Seamount', ['long', 'short','area','long','direction'])

def draw_ellipse(edge_image, original_image, contours):
    list = []  # 存储海山的形态学中心
    list_index = []

    new_picture = np.ones([edge_image.shape[0], edge_image.shape[1], edge_image.shape[2]]) * 255  # 创建一个新的空白画布
    new_picture_ellipse_filtering = np.ones([edge_image.shape[0], edge_image.shape[1], edge_image.shape[2]]) * 255

    TheAngleOfEllipseNeedToClustering = np.empty(shape=(1,))
    TheCentreOfEllipseNeedToClustering = np.empty(shape=(1, 2), dtype=float)
    TheAxisOfEllipseNeedToClustering = np.empty(shape=(1, 2), dtype=float)
    j = 0
    for i in range(len(contours) - 1):  # 绘制椭圆,len(contours)=椭圆的数量

        if (np.vstack(contours[i]).squeeze()).shape[0] > 5:  # 必须大于5个点才能拟合出一个椭圆

            if (directed_hausdorff(np.reshape(contours[i], (contours[i].shape[0], 2)),
                                   np.reshape(contours[i + 1], (contours[i + 1].shape[0], 2)))[0] > 4 and i != len(
                contours) - 2):  # 豪斯多夫(Hausdorff)距离滤波

                ellipse = cv2.fitEllipse(np.vstack(contours[
                                                       i]).squeeze())  # 画出每一个椭圆，遍历边缘矩阵contours，ellipse =  [ (x, y) , (a, b), angle ]，（x, y）代表椭圆中心点的位置，（a, b）代表长短轴长度
                image = cv2.ellipse(new_picture, ellipse, (0, 0, 255), 1)  # shape=(696,736,3)
                geometrical_center = seamounts.get_geometrical_center(contours[i]).flatten()

                list.append([geometrical_center, i])  # 中心坐标和边缘索引
                list_index.append(geometrical_center)

                # cv2.putText(image, str(round(ellipse[2], 2)), (int(ellipse[0][0]), int(ellipse[0][1])),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.35, (255, 0, 0), 1)  # 顺时针旋转,打印椭圆的旋转角度
                # cv2.putText(image, str(j), (int(ellipse[0][0]), int(ellipse[0][1])),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # 顺时针旋转,打印椭圆的旋转角度
                cv2.circle(image, geometrical_center, 1, (255, 0, 0), 2)  # 标记海山中心点
                j = j + 1
                if (min(ellipse[1][0], ellipse[1][1]) / max(ellipse[1][0], ellipse[1][1]) < 0.8):
                    image_ellipse_filtering = cv2.ellipse(new_picture_ellipse_filtering, ellipse, (0, 0, 255), 1)

                    TheCentreOfEllipseNeedToClustering = np.append(TheCentreOfEllipseNeedToClustering,
                                                                   np.array(ellipse[0]).reshape(1, -1), axis=0)
                    TheAxisOfEllipseNeedToClustering = np.append(TheAxisOfEllipseNeedToClustering,
                                                                 np.array(ellipse[0]).reshape(1, -1), axis=0)
                    TheAngleOfEllipseNeedToClustering = np.append(TheAngleOfEllipseNeedToClustering,
                                                                  np.array(ellipse[2]).reshape(1, ))

    list = sorted(list, key=lambda x: np.linalg.norm(x[0] - (0, 0)))
    list_index = sorted(list_index, key=lambda x: np.linalg.norm(x - (0, 0)))
    # dist_index=find_nearest(list_index)
    dist_index = seamounts.find_nearest(list)
    for i in range(1, len(list) - 1):

        min_centre = seamounts.contours_compare(contours[list[i][1]], contours[list[dist_index[3 * i]][1]],
                                      contours[list[dist_index[3 * i + 1]][1]],
                                      contours[list[dist_index[3 * i + 2]][1]])

        if (min_centre == "a"):
            if np.linalg.norm(list[i][0] - list[dist_index[3 * i]][0], axis=0) <= 40:
                cv2.line(image, list[i][0], list[dist_index[3 * i]][0], (255, 0, 0), 1, cv2.LINE_8)
                cv2.line(original_image, list[i][0], list[dist_index[3 * i]][0], (255, 255, 255), 1, cv2.LINE_8)

        if (min_centre == "b"):
            if np.linalg.norm(list[i][0] - list[dist_index[3 * i + 1]][0], axis=0) <= 40:
                cv2.line(image, list[i][0], list[dist_index[3 * i + 1]][0], (255, 0, 0), 1, cv2.LINE_8)
                cv2.line(original_image, list[i][0], list[dist_index[3 * i + 1]][0], (255, 255, 255), 1, cv2.LINE_8)

        if (min_centre == "c"):
            if np.linalg.norm(list[i][0] - list[dist_index[3 * i + 2]][0], axis=0) <= 40:
                cv2.line(image, list[i][0], list[dist_index[3 * i + 2]][0], (255, 0, 0), 1, cv2.LINE_8)
                cv2.line(original_image, list[i][0], list[dist_index[3 * i + 2]][0], (255, 255, 255), 1, cv2.LINE_8)

    TheAngleOfEllipseNeedToClustering = np.delete(TheAngleOfEllipseNeedToClustering, 0).reshape(-1, 1)  # shape=312
    TheCentreOfEllipseNeedToClustering = np.delete(TheCentreOfEllipseNeedToClustering, 0, axis=0)
    TheAxisOfEllipseNeedToClustering = np.delete(TheAxisOfEllipseNeedToClustering, 0, axis=0)

    cluster_group_labels = seamounts.Clustering(image_ellipse_filtering, TheAngleOfEllipseNeedToClustering,
                                      TheCentreOfEllipseNeedToClustering,
                                      TheAxisOfEllipseNeedToClustering)  # 获得每一个椭圆的分类，再画图时按分类给每一个椭圆标上不同的颜色

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.hist(TheAngleOfEllipseNeedToClustering, bins=np.arange(0, 180, 18))
    plt.title('旋转角度分布')
    plt.show()

    return image, original_image, image_ellipse_filtering


if __name__ == '__main__':
    x = original_image.copy()
    original_image_cover_line = cv2.imread(path)
    edge_image = cv2.Canny(seamounts.adaptive_threshold(path), 100, 200)  # 双通道二维图片
    # contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 获取边界
    #
    # dst = cv2.drawContours(x, contours, -1, (255, 0, 0), 1)
    # edge_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)  # 再获取了边界点数据后再将图片转3通道
    #
    # image, original_image_cover_line, image_ellipse_filtering = draw_ellipse(edge_image, original_image_cover_line,
    #                                                                          contours)

    # cv2.imshow('original_image', original_image)
    # # cv2.imshow('original_image_cover_line', original_image_cover_line)
    # # # seamounts.img_show(original_image,original_image_cover_line)
    # # cv2.imshow('angle_image', image)
    # cv2.imshow('edge_image', edge_image)
    # cv2.imshow('image_ellipse_filtering', image_ellipse_filtering)
    # cv2.imshow('dst', dst)
    # cv2.imwrite('dst.png', dst)
    cv2.waitKey(0)