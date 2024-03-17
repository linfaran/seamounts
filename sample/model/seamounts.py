import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections import defaultdict
from heapq import *
from sample.model import parameter
from sample.model import Retinex
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage as ndi
from skimage import morphology,feature

import heapq
import os
para = parameter.para()

def otsu_threshold_in_lab(img_):
    img = img_.copy()
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]

    ret, thresh = cv.threshold(l_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('otsu_threshold_in_lab', img)

def img2Gray_mean(image):
    #平均值法
    h, w = image.shape[:2]
    gray3 = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            gray3[i, j] = (int(image[i, j][0]) + int(image[i, j][1]) + int(image[i, j][2])) / 3
    return gray3

def img2Gray_max(image):
    #最大值法
    h, w = image.shape[:2]
    gray4 = np.zeros((h, w), dtype=np.uint8)  # 创建一个h行w列的二维list
    for i in range(h):
        for j in range(w):
           gray4[i, j] = max(image[i, j][0], image[i, j][1], image[i, j][2])
    return gray4

def img2Gray_component(image):
    #分量法:
    gray6 = image.copy()
    for i in range(gray6.shape[0]):
        for j in range(gray6.shape[1]):
            gray6[i, j] = gray6[i, j, 0]
    return gray6

def img2Gray_weighted(image):
    #加权平均分
    h, w = image.shape[:2]
    gray5= np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # Y = 0．3R + 0．59G + 0．11B
            # 通过cv格式打开的图片，像素格式为 BGR
            gray5[i, j] = 0.3 * image[i, j][2] + 0.11 * image[i, j][0] + 0.59 * image[i, j][1]
    return gray5

def img_tophat(binary_img):
    kernel = np.ones((2, 2), np.uint8)
    cvOpen = cv.morphologyEx(binary_img, cv.MORPH_TOPHAT, kernel)
    return cvOpen

def img_blackhat(binary_img):
    kernel = np.ones((2, 2), np.uint8)
    cvClose = cv.morphologyEx(binary_img, cv.MORPH_BLACKHAT, kernel)
    return cvClose

def img_erode(binary_img,iterations):
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv.erode(binary_img, kernel, iterations=iterations)
    return erosion

def img_dilate(binary_img,iterations):
    kernel = np.ones((2, 2), np.uint8)
    img_dilate = cv.dilate(binary_img, kernel, iterations=iterations)
    return img_dilate

def watershed_(img,binary_img):
    # finding sure foreground area
    dist_transfrom = cv.distanceTransform(binary_img, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transfrom, 0.04 * dist_transfrom.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknow = cv.subtract(binary_img, sure_fg)  # 背景-前景

    ret, maker = cv.connectedComponents(sure_fg)
    maker = maker + 1
    maker[unknow == 255] = 0

    maker = cv.watershed(img, maker)
    maker = maker.astype(np.float32)

    img[maker == -1] = [255, 0, 0]

    return img

def fill_hole(imgray):
    # imgray = cv.imread("binary_img.png")

    # 二值化
    # imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgray[imgray < 100] = 0
    imgray[imgray >= 100] = 255

    # 原图取补得到MASK图像
    mask = 255 - imgray

    # 构造Marker图像
    marker = np.zeros_like(imgray)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255
    marker_0 = marker.copy()

    # 形态学重建
    SE = cv.getStructuringElement(shape=cv.MORPH_CROSS, ksize=(3, 3))
    while True:
        marker_pre = marker
        dilation = cv.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    filling = dst - imgray

    # 显示
    plt.figure(figsize=(12, 6))  # width * height
    plt.subplot(2, 3, 1), plt.imshow(imgray, cmap='gray'), plt.title('src'), plt.axis("off")
    plt.subplot(2, 3, 2), plt.imshow(mask, cmap='gray'), plt.title('Mask'), plt.axis("off")
    plt.subplot(2, 3, 3), plt.imshow(marker_0, cmap='gray'), plt.title('Marker 0'), plt.axis("off")
    plt.subplot(2, 3, 4), plt.imshow(marker, cmap='gray'), plt.title('Marker'), plt.axis("off")
    plt.subplot(2, 3, 5), plt.imshow(dst, cmap='gray'), plt.title('dst'), plt.axis("off")
    plt.subplot(2, 3, 6), plt.imshow(filling, cmap='gray'), plt.title('Holes'), plt.axis("off")
    plt.show()

    return dst

def manual_threshold_segmentation(gray_img,img):
    img_manual_threshold_segmentation=img.copy()
    denoise_img = cv.fastNlMeansDenoising(gray_img, *para.cv2_fastNlMeansDenoising_para.values())
    denoise_img[denoise_img <= 140] = 0  # 灰度值过滤
    cv.imshow('denoise_img', denoise_img)
    binary_img = fill_hole(denoise_img)
    cv.imshow('binary_fill_hole_img', binary_img)
    binary_img = img_dilate(binary_img, 3)
    cv.imshow('img_dilate', binary_img)
    binary_img = img_erode(binary_img, 2)
    cv.imshow('img_erode', binary_img)
    binary_img = fill_hole(binary_img)
    cv.imshow('fill_hole', binary_img)
    # thresh0,sure_bg,sure_fg,gray=watershed(dark_flip_img)
    # cv.imshow('gray', gray)
    watershed_img=watershed_(img_manual_threshold_segmentation,binary_img)   #分水岭算法
    cv.imshow('manual_threshold_segmentation watershed_img', watershed_img)
    return binary_img

def adaptive_threshold_segmentation(img,adaptive_gaussian_threshold):
    adaptive_threshold_segmentation_img=img.copy()
    # 获取图像的维度
    height, width = adaptive_gaussian_threshold.shape
    # 设置最左边和最右边的5列为白色
    adaptive_gaussian_threshold[:, :5] = 255  # 最左边的5列
    adaptive_gaussian_threshold[:, width - 5:] = 255  # 最右边的5列

    def connectedComponentsWithStats(img):
        mask = img
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=4)
        output = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
        for i in range(1, num_labels):
            k = labels == i
            if i==1:
                output[:, :, 0][k] = 255
                output[:, :, 1][k] = 255
                output[:, :, 2][k] = 255
            else :
                output[:, :, 0][k] = 0
                output[:, :, 1][k] = 0
                output[:, :, 2][k] = 0
        return output

    adaptive_gaussian_threshold_dilate=img_erode(adaptive_gaussian_threshold,2)
    cv.imshow('adaptive_gaussian_threshold_dilate', adaptive_gaussian_threshold_dilate)
    #watershed_(adaptive_threshold_segmentation_img, 255-adaptive_gaussian_threshold_dilate)  # 分水岭算法
    output = connectedComponentsWithStats(adaptive_gaussian_threshold_dilate)
    cv.imshow('output0', output)
    output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)  # 转双通道
    watershed_img=watershed_(adaptive_threshold_segmentation_img, 255-output)  # 分水岭算法
    cv.imshow('adaptive_threshold_segmentation watershed_img', watershed_img)
    return output

def otsu_adaptive_threshold_segmentation(img,gray_img):
    otsu_img=img.copy()
    ret, otsu_thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('Otsu Threshold', otsu_thresh)
    otsu_fill_hole=fill_hole(otsu_thresh)
    cv.imshow('otsu_fill_hole', otsu_fill_hole)
    watershed_img=watershed_(otsu_img,otsu_fill_hole)
    cv.imshow('Otsu Threshold watershed_img', watershed_img)
    return otsu_fill_hole

def Thresholding(img):
    # Adaptive Mean Thresholding:
    adaptive_mean_threshold = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                                   cv.THRESH_BINARY, 17, 19)
    cv.imshow('Adaptive Mean Thresholding', adaptive_mean_threshold)
    # Adaptive Gaussian Thresholding:
    adaptive_gaussian_threshold = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv.THRESH_BINARY, 3, 10)
    cv.imshow('Adaptive Gaussian Thresholding', adaptive_gaussian_threshold)

    # Sauvola's Method:
    def sauvola_thresholding(img, window_size=15, k=0.2, R=128):
        mean = cv.boxFilter(img, ddepth=-1, ksize=(window_size, window_size))
        mean_square = cv.boxFilter(img ** 2, ddepth=-1, ksize=(window_size, window_size))
        std = np.sqrt(mean_square - mean ** 2)
        threshold = mean * (1 + k * ((std / R) - 1))
        return (img > threshold).astype(np.uint8) * 255

    sauvola_threshold = sauvola_thresholding(img)
    cv.imshow('Sauvola Thresholding', sauvola_threshold)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization):
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    clahe_img = clahe.apply(img)
    cv.imshow('CLAHE', clahe_img)

    # After applying CLAHE, you can use another thresholding method if needed
    clahe_threshold = cv.adaptiveThreshold(clahe_img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                           cv.THRESH_BINARY, 17, 17)
    cv.imshow('CLAHE with Adaptive Mean Thresholding', clahe_threshold)

    return adaptive_gaussian_threshold

def plot_histogram(img):
    # flatten() 将数组变成一维
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # 计算累积分布图
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

def equalizehist(img):
    equ = cv.equalizeHist(img)
    # res = np.hstack((img, equ))
    # stacking images side-by-side
    cv.imshow('equalizehist', equ)
    plot_histogram(equ)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 应用CLAHE到图像
    cl1 = clahe.apply(img)

    # 显示或保存结果
    cv.imshow('CLAHE Image', cl1)
    return equ,cl1

def contours_compare(x, y, z, f):
    # 将轮廓转换为二维数组
    x, y, z, f = [np.reshape(contour, (contour.shape[0], 2)) for contour in [x, y, z, f]]

    def min_distance(contour1, contour2):
        # 计算轮廓间的最小距离
        return min([np.linalg.norm(point - contour2, axis=1).min() for point in contour1])

    # 计算 x 与其他轮廓的最小距离
    distances = {min_distance(x, contour): label for contour, label in zip([y, z, f], ['a', 'b', 'c'])}

    # 返回最小距离对应的标签
    return distances[min(distances)]

def find_nearest(list):
    length = len(list)
    dist_index = []

    for i in range(length):
        # 创建一个优先队列来存储距离和索引
        distances = []
        for j in range(length):
            if i != j:
                # 计算距离并将其与索引一起放入优先队列中
                dist = np.linalg.norm(list[i][0] - list[j][0])
                heapq.heappush(distances, (dist, j))

        # 获取最近的三个点
        nearest_three = heapq.nsmallest(3, distances)
        dist_index.extend([idx for _, idx in nearest_three])

    return dist_index

def get_geometrical_center(contours):
    mean = np.mean(contours, axis=0)  # 计算每一列的平均值
    geometrical_center = mean.astype(int)  # 转换为整数
    return geometrical_center

def Clustering(img, TheAngleOfEllipseNeedToClustering, TheCentreOfEllipseNeedToClustering,
               TheAxisOfEllipseNeedToClustering):  # 聚类算法
    TheArrayThatNeedToClustring = np.hstack(
        (TheAngleOfEllipseNeedToClustering, TheCentreOfEllipseNeedToClustering, TheAxisOfEllipseNeedToClustering))
    group_size = 3

    # AgglomerativeClustering层次聚类分析

    model = AgglomerativeClustering(n_clusters=group_size, linkage='ward')
    cluster_group = model.fit(TheArrayThatNeedToClustring)
    cnames = ['black', 'blue', 'red']
    for point, gp_id in zip(TheArrayThatNeedToClustring,
                            cluster_group.labels_):  # point会按顺序扫描每一个椭圆，理论上我们只要获取了gp_id就可以区分出每一个椭圆了，因为他是按顺序的
        # 放到 plt 中展示
        plt.scatter(point[0], point[1], s=5, c=cnames[gp_id], alpha=1)
    plt.show()
    return cluster_group.labels_

def img_show(*args):
    columns = [d.name for d in args]
    for i in range(len(args)):
        cv.imshow(columns[i],args[i].value)
    cv.waitKey(0)

def watershed_in_rgb(img_):
    img=img_.copy()
    # 将RGB图像转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 应用Otsu阈值法来获取二值图像
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 形态学操作以去除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 确定前景区域
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 确定未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # 标记标签
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 应用分水岭算法
    markers = cv.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]
    cv.imshow('watershed_in_rgb', img)

def watershed_in_lab(img_):
    save_path = "D:\\software\\Python\\PythonProject\\seamounts_edge_detection\\sample\\data\\plt"
    img=img_.copy()
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]

    ret, thresh = cv.threshold(l_channel, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((1,1), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations=3)

    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.031*dist_transform.max(), 255, 0)      #default=0.03

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [0,0,255]

    # Create a figure to display the images
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # Display and save images
    images = [cv.cvtColor(img_, cv.COLOR_BGR2RGB), l_channel, thresh, opening, sure_bg,
              dist_transform, sure_fg, unknown, cv.cvtColor(img, cv.COLOR_BGR2RGB)]
    titles = ['Original Image', 'L Channel', 'Thresholded Image', 'Opening', 'Sure Background',
              'Distance Transform', 'Sure Foreground', 'Unknown Region', 'Result']
    file_names = ['original_image.jpg', 'l_channel.jpg', 'thresholded_image.jpg', 'opening.jpg',
                  'sure_background.jpg', 'distance_transform.jpg', 'sure_foreground.jpg',
                  'unknown_region.jpg', 'result.jpg']

    for i, (image, title, file_name) in enumerate(zip(images, titles, file_names)):
        axs[i // 3, i % 3].set_title(title)
        cmap = 'gray' if len(image.shape) == 2 else None
        axs[i // 3, i % 3].imshow(image, cmap=cmap)
        save_file_path = os.path.join(save_path, file_name)
        if cmap is None:  # RGB image
            plt.imsave(save_file_path, image)
        else:  # Grayscale image
            plt.imsave(save_file_path, image, cmap='gray')

    plt.tight_layout()
    plt.show()

    # Instead of returning file_names, the function now prints them
    print("Images saved:", file_names)

    # To use this function, you still need to provide an image in BGR format
    # Example: watershed_in_lab(cv.imread('path_to_your_image.jpg'))
    return markers
def adaptive_threshold(path):
    high_res_img_path=os.path.join('..', 'data', 'data_0030160180.jpg')
    img = cv.imread(path)  # 读取
    img_orign = img.copy()
    new_img = np.ones((img.shape[0], img.shape[1], 3)) * 255
    img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    dark_flip_img = img.copy()
    dark_flip_img[np.sum(img, axis=2) <= 120] = [0, 170, 170]  # 暗部翻转
    cv.imshow('dark_flip_img', dark_flip_img)
    lab = cv.cvtColor(dark_flip_img, cv.COLOR_BGR2LAB)
    cv.imshow('lab', lab)

    markers = watershed_in_lab(lab)
    otsu_threshold_in_lab(lab)

    # gray_img = cv.cvtColor(dark_flip_img, cv.COLOR_BGR2GRAY)
    gray_img = img2Gray_weighted(dark_flip_img)
    # gray_img[np.logical_and(110 < gray_img, gray_img <= 120)]=0
    cv.imshow('gray_img', gray_img)
    plot_histogram(gray_img)
    equ, cl1 = equalizehist(gray_img)
    adaptive_gaussian_threshold = Thresholding(gray_img)
    binary_img_manual_threshold_segmentation = manual_threshold_segmentation(gray_img, img)  # 基于手动阈值分割的图像处理
    otsu_binary = otsu_adaptive_threshold_segmentation(img, gray_img)  # 基于大津算法的图像处理
    adaptive_threshold_segmentation_binary = adaptive_threshold_segmentation(img,
                                                                             adaptive_gaussian_threshold)  # 基于高斯自适应阈值函数的图像处理
    img_orign[markers == -1] = [255, 255, 255]
    new_img[markers == -1] = [255, 0, 0]
    high_res_img=cv.imread(high_res_img_path)
    print(high_res_img.shape)# (7200, 4800, 3)
    resized_img = cv.resize(img_orign, (high_res_img.shape[1], high_res_img.shape[0]))
    print(resized_img.shape)
    # for i in range(resized_img.shape[0]):
    #     for j in range(resized_img.shape[1]):
    #         # if np.all(resized_img[i][j] == [255, 255, 255]):
    #         #     high_res_img[i][j]==[255,255,255]
    #         if i<1000 and j<1000:
    #             high_res_img[i][j] == [255, 255, 255]
    white_points_mask = np.all(resized_img > 230, axis=-1)

    # Apply this mask to high_res_img to set those points to white
    high_res_img[white_points_mask] = [255, 255, 255]
    cv.imshow('img_orign_lab_watershed', img_orign)
    cv.imshow('img_orign_lab_watershed_marks', new_img)
    cv.imwrite('img_orign_lab_watershed.png', img_orign)
    cv.imwrite('img_orign_lab_watershed_marks.png', new_img)
    cv.imwrite('high_res_img.png', high_res_img)


