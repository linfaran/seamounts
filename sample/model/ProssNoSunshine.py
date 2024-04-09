import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from sample.model import Retinex
from skimage import measure, filters,exposure,feature
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from skimage import exposure, filters, measure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap
from skimage.feature import canny
import scipy.optimize as so
from sklearn.decomposition import PCA
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes
class ProssNoSunshine(object):

    def __init__(self, path_of_img_no_shadow,path_of_img_with_shadow,path_of_img_with_ramp_shader):
        self.path_of_img_no_shadow = path_of_img_no_shadow  # Reading the image path
        self.img = cv.imread(self.path_of_img_no_shadow)  # Reading the image as grayscale
        self.path_of_img_with_shadow = path_of_img_with_shadow
        self.img_with_shadow=cv.imread(self.path_of_img_with_shadow)
        self.path_of_img_with_ramp_shader = path_of_img_with_ramp_shader
        self.img_with_ramp_shader = cv.imread(self.path_of_img_with_ramp_shader)
    def resize_img_show(self,img,scale):
        cv.namedWindow('Window', cv.WINDOW_NORMAL)
        cv.resizeWindow('Window', int(img.shape[1] / scale), int(img.shape[0] / scale))
        cv.imshow('Window', img)
    def enhance_image_contrast(self, img):
        # 计算左上角区域的大小（图像的1/4大小）
        height, width = img.shape[:2]
        square_size = min(height, width) // 4

        # Calculate the dimensions for the bottom-left quarter
        quarter_height = height // 6
        quarter_width = width // 6

        # 分割原始图片为两个部分
        top_left_square = img[:square_size, :square_size].copy()
        rest_of_image = img.copy()

        # 对左上角区域进行处理
        top_left_processed = self.process_part(top_left_square, 50, 86)

        # 对其余部分进行处理（可能使用不同的low和high值）
        rest_of_image_processed = self.process_part(rest_of_image, 38,86)

        # 将处理后的左上角区域合并回到其余部分
        rest_of_image_processed[:square_size, :square_size] = top_left_processed

        rest_of_image_processed[height - quarter_height:, :quarter_width] = 0

        # Define and process the arbitrary quadrilateral
        quad_coords=[(3513,1186),(3756,1316),(3210,2083),(2990,1896)]
        mask = np.zeros_like(img, dtype=np.uint8)
        cv.fillPoly(mask, [np.array(quad_coords)], (255, 255, 255))
        quad_area = cv.bitwise_and(img, mask)
        quad_processed = self.process_part(quad_area, 48, 86)
        rest_of_image_processed = cv.bitwise_and(rest_of_image_processed, cv.bitwise_not(mask))
        rest_of_image_processed = cv.add(rest_of_image_processed, quad_processed)

        return rest_of_image_processed
    def process_part(self,img,low,high):
        # Convert the image to the HSV color space
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Define range of green color in HSV
        lower_green = np.array([low, 0, 0])
        upper_green = np.array([high, 255, 255])
        # Threshold the HSV image to get only green colors
        mask = cv.inRange(hsv, lower_green, upper_green)
        # Invert the mask to get non-green areas
        mask_inv = cv.bitwise_not(mask)
        # Bitwise-AND mask and original image to remove green background
        res = cv.bitwise_and(img, img, mask=mask_inv)
        # Convert the result to grayscale
        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        # Apply thresholding to get binary image
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        # Find contours (connected components) in the thresholded image
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Draw contours on the original image
        cv.drawContours(res, contours, -1, (0, 255, 0), 3)
        # Convert BGR to RGB for matplotlib
        #result_img_rgb = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        # Display the result using matplotlib
        return res
    def gray_binary_img(self,img):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        ret, res = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        return res
    def connected_component_analysis(self,image_gray):
        # Reapplying adaptive histogram equalization with a lower clip limit to enhance contrast
        image_eq_enhanced = exposure.equalize_adapthist(image_gray, clip_limit=0.01)

        # Convert the enhanced image back to uint8 format
        image_eq_enhanced_uint8 = (image_eq_enhanced * 255).astype(np.uint8)

        # Use a threshold to create a binary image, threshold chosen by inspecting the histogram
        thresh_enhanced = filters.threshold_otsu(image_eq_enhanced_uint8)
        binary_image_enhanced = image_eq_enhanced_uint8 > thresh_enhanced

        # Perform connected component analysis to label the regions
        labeled_image_enhanced, num_features_enhanced = measure.label(binary_image_enhanced, background=0,
                                                                      return_num=True)
        props_enhanced = measure.regionprops(labeled_image_enhanced)
        areas = [prop.area for prop in props_enhanced]
        # Plot the labeled image with bounding boxes around the highlighted areas
        fig, ax = plt.subplots(figsize=(15, 15), dpi=500)
        ax.imshow(image_eq_enhanced_uint8, cmap='gray')  # Show the enhanced contrast image
        ax.imshow(labeled_image_enhanced, cmap='nipy_spectral', alpha=0.5)  # Overlay the labeled regions
        # Draw the bounding boxes in red
        for prop in props_enhanced:
            ax.text(
                prop.centroid[1], prop.centroid[0],
                f'Area {prop.area:.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=2, color='white'

            )
            minr, minc, maxr, maxc = prop.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=0.1)
            ax.add_patch(rect)

        ax.axis('off')
        plt.show()
    def edge_analysis(self,image_gray):
        # Assuming image_gray is the grayscale image loaded from file
        # Apply adaptive histogram equalization with a lower clip limit to enhance contrast
        image_eq_enhanced = exposure.equalize_adapthist(image_gray, clip_limit=0.01)

        # Convert the enhanced image back to uint8 format
        image_eq_enhanced_uint8 = (image_eq_enhanced * 255).astype(np.uint8)

        # Use a threshold to create a binary image, threshold chosen by inspecting the histogram
        thresh_enhanced = filters.threshold_otsu(image_eq_enhanced_uint8)
        binary_image_enhanced = image_eq_enhanced_uint8 > thresh_enhanced

        # Perform connected component analysis to label the regions
        labeled_image_enhanced, num_features_enhanced = measure.label(binary_image_enhanced.astype(np.uint8),
                                                                      background=0,
                                                                      return_num=True)
        props_enhanced = measure.regionprops(labeled_image_enhanced)

        # Plot the labeled image with bounding boxes around the highlighted areas
        fig, ax = plt.subplots(figsize=(15, 15), dpi=500)
        ax.imshow(image_eq_enhanced_uint8, cmap='gray')  # Show the enhanced contrast image

        # Draw the bounding boxes and get edge information in red
        for prop in props_enhanced:
            # Get the coordinates of the region's bounding box
            minr, minc, maxr, maxc = prop.bbox
            region = binary_image_enhanced[minr:maxr, minc:maxc]
            # Pad the region to avoid border effects during edge detection
            region_padded = np.pad(region, 1, mode='constant', constant_values=0)
            # Detect edges using the Canny edge detector
            edges = feature.canny(region_padded, sigma=1)
            # Get the coordinates of the edge pixels and adjust for the padding
            edge_coords = np.column_stack(np.where(edges)) - 1
            # Adjust for the location of the region within the larger image
            edge_coords[:, 0] += minr
            edge_coords[:, 1] += minc
            # Plot the edges
            ax.plot(edge_coords[:, 1], edge_coords[:, 0], 'r.', markersize=0.1)

        ax.axis('off')
        plt.show()

    def connected_component_analysis_and_ellipse_fitting(self,image_gray):
        # Reapplying adaptive histogram equalization with a lower clip limit to enhance contrast
        image_eq_enhanced = exposure.equalize_adapthist(image_gray, clip_limit=0.01)

        # Convert the enhanced image back to uint8 format
        image_eq_enhanced_uint8 = (image_eq_enhanced * 255).astype(np.uint8)

        # Use a threshold to create a binary image, threshold chosen by inspecting the histogram
        thresh_enhanced = filters.threshold_otsu(image_eq_enhanced_uint8)
        binary_image_enhanced = image_eq_enhanced_uint8 > thresh_enhanced

        # Perform connected component analysis to label the regions
        labeled_image_enhanced, num_features_enhanced = measure.label(binary_image_enhanced, background=0,
                                                                      return_num=True)
        props_enhanced = measure.regionprops(labeled_image_enhanced)

        # Prepare color map for ellipse orientation
        cmap = LinearSegmentedColormap.from_list('red_to_blue', ['red', 'blue'])

        # Plot the labeled image with bounding boxes and ellipses
        fig, ax = plt.subplots(figsize=(15, 15), dpi=5000)
        ax.imshow(image_eq_enhanced_uint8, cmap='gray')  # Show the enhanced contrast image

        for prop in props_enhanced:
            # 获取椭圆的中心、轴长和旋转角度
            y0, x0 = prop.centroid
            orientation = prop.orientation
            major_axis_length = prop.major_axis_length
            minor_axis_length = prop.minor_axis_length

            # 计算OpenCV的椭圆参数
            center = (int(x0), int(y0))
            axes = (int(major_axis_length / 2), int(minor_axis_length / 2))
            angle = np.degrees(orientation)
            # 绘制椭圆
            cv.ellipse(self.img, center, axes, angle, 0, 360, (255, 0, 0), 4)

        # 将修改后的图像转换回灰度格式（如果需要）
        # image_modified_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 显示或保存图像
        self.resize_img_show(self.img,scale=6)

    def rotate_vector(self,vector, angle_degree):
        # 将角度转换为弧度
        angle_rad = np.radians(angle_degree)
        # 定义旋转矩阵
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        # 应用旋转矩阵
        return np.dot(rotation_matrix, vector)

    def get_angle(self,first_component):
        """
        计算与水平轴之间的角度（0-180度）。

        :param first_component: 包含两个元素的列表或数组，表示二维向量。
        :return: 返回与水平轴之间的角度（度）。
        """
        # 使用 arctan2 计算角度（以弧度为单位）
        angle_rad = np.arctan2(first_component[1], first_component[0])

        # 将弧度转换为度
        angle_deg = np.degrees(angle_rad)

        # 逆时针旋转90度
        angle_deg += 90

        # 调整角度使其在0到360度范围内
        if angle_deg >= 360:
            angle_deg -= 360
        elif angle_deg < 0:
            angle_deg += 360
        return angle_deg

    def chain_code_analysis(self,edges):
        """
        简化版链码分析函数，计算图像边缘的链码，并估计主要朝向。
        """
        # 查找边缘上的轮廓，这里假设edges是一个二值化的边缘图像
        contours = find_contours(edges, level=0.5)

        if not contours:
            return 0  # 如果没有找到轮廓，返回0作为默认朝向

        # 选取最长的轮廓进行分析
        longest_contour = max(contours, key=len)

        # 计算每一步的方向变化（链码）
        deltas = np.diff(longest_contour, axis=0)
        directions = np.arctan2(deltas[:, 0], deltas[:, 1])

        # 将方向转换为角度
        degrees = np.rad2deg(directions)

        # 计算所有角度的直方图，找到最常见的方向
        hist, _ = np.histogram(degrees, bins=np.arange(-180, 181, 10), density=True)
        most_common_direction = np.argmax(hist) * 10 - 180

        return most_common_direction

    def pca_analysis_of_concave_polygon_orientation(self, image_gray):
        # Reapplying adaptive histogram equalization with a lower clip limit to enhance contrast
        image_eq_enhanced = exposure.equalize_adapthist(image_gray, clip_limit=0.01)
        # Convert the enhanced image back to uint8 format
        image_eq_enhanced_uint8 = (image_eq_enhanced * 255).astype(np.uint8)
        # Use a threshold to create a binary image
        thresh_enhanced = filters.threshold_otsu(image_eq_enhanced_uint8)
        binary_image_enhanced = image_eq_enhanced_uint8 > thresh_enhanced
        binary_image_filled = binary_fill_holes(binary_image_enhanced)  # 填充多边形内部
        edges = feature.canny(binary_image_filled)  # 使用Canny边缘检测
        # Perform connected component analysis to label the regions
        labeled_image_enhanced, num_features_enhanced = measure.label(binary_image_enhanced, background=0,
                                                                      return_num=True)
        props_enhanced = measure.regionprops(labeled_image_enhanced)
        # Define figure size and DPI
        fig_size = 6  # You can adjust this size
        dpi_val = 100  # You can adjust this DPI value

        # Calculate fontsize proportional to figure size and DPI
        fontsize_proportional = fig_size * dpi_val / 50  # Adjust denominator for scaling

        # Plot the labeled image with adjusted size and resolution
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi_val)
        ax.imshow(image_eq_enhanced_uint8, cmap='gray')
        angles_list = []
        for prop in props_enhanced:
            chain_orientation = self.chain_code_analysis(edges)
            # Get the pixel coordinates of the region
            coords = prop.coords

            # Apply PCA analysis
            pca = PCA(n_components=2)
            pca.fit(coords)
            first_component = pca.components_[0]

            # Correcting direction to ensure it's between 9 o'clock and 3 o'clock direction
            # This makes sure the angle is measured clockwise from the 9 o'clock direction
            angle = np.degrees(np.arctan2(-first_component[0], first_component[1]))

            # Get PCA direction and position
            center = prop.centroid
            vector_length = 50  # Adjust vector length for visualization
            if angle > 0:
                adjusted_vector = np.array([-first_component[1], first_component[0]])  # Reverse direction
            else:
                adjusted_vector = np.array([first_component[1], -first_component[0]])

            if angle > 0:
                angle -= 180
            angles_list.append(-angle)
            vector = adjusted_vector * vector_length

            # Plot PCA principal component direction
            plt.quiver(center[1], center[0], vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='red')

            text_pos = (center[1] + vector[0], center[0] + vector[1])
            plt.text(text_pos[0], text_pos[1], f"{-angle:.2f}°", color='blue', fontsize=fontsize_proportional)

            # 计算链码方向的向量（示例代码，根据需要调整长度和方向）
            chain_vector_length = 50  # 可以调整以适合你的可视化需求
            chain_dx = chain_vector_length * np.cos(np.radians(chain_orientation))
            chain_dy = chain_vector_length * np.sin(np.radians(chain_orientation))

            # 绘制链码分析的方向
            plt.quiver(center[1], center[0], chain_dx, chain_dy, angles='xy', scale_units='xy', scale=1, color='green',
                       width=0.005)

            # 你可以选择在向量末端添加文本以标示方向
            plt.text(center[1] + chain_dx, center[0] + chain_dy, f"Chain: {chain_orientation:.2f}°", color='green',
                     fontsize=fontsize_proportional / 2)

        plt.axis('equal')
        plt.show()

        # After the loop, plot the histogram of angles
        plt.figure(figsize=(10, 6))
        plt.hist(angles_list, bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of Polygon Orientations')
        plt.xlabel('Orientation Angle (Degrees)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    def color_regions_by_orientation(self, image_gray):
        # Reapplying adaptive histogram equalization with a lower clip limit to enhance contrast
        image_eq_enhanced = exposure.equalize_adapthist(image_gray, clip_limit=0.01)

        # Convert the enhanced image back to uint8 format
        image_eq_enhanced_uint8 = (image_eq_enhanced * 255).astype(np.uint8)

        # Use a threshold to create a binary image
        thresh_enhanced = filters.threshold_otsu(image_eq_enhanced_uint8)
        binary_image_enhanced = image_eq_enhanced_uint8 > thresh_enhanced

        # Perform connected component analysis to label the regions
        labeled_image_enhanced, num_features_enhanced = measure.label(binary_image_enhanced, background=0,
                                                                      return_num=True)
        props_enhanced = measure.regionprops(labeled_image_enhanced)

        # Create an image to display the colored regions
        colored_regions = np.zeros((image_eq_enhanced_uint8.shape[0], image_eq_enhanced_uint8.shape[1], 3), dtype=np.uint8)

        # Define a color map
        n_colors = 18  # 180° / 10° = 18 sections
        cmap = plt.cm.get_cmap('hsv', n_colors)

        for prop in props_enhanced:
            # 获取区域的像素坐标
            coords = prop.coords

            # 应用PCA分析
            pca = PCA(n_components=2)
            pca.fit(coords)
            first_component = pca.components_[0]

            # 确保方向在0-180°范围内
            if first_component[1] < 0:
                first_component = -first_component

            # 计算朝向角度
            angle = np.arctan2(first_component[1], first_component[0])
            angle_degrees = np.degrees(angle) % 180

            # 选择颜色
            color_index = int(angle_degrees / 10)  # 每10°分一个区间
            color = cmap(color_index)[:3]  # 提取RGB颜色

            # 对应颜色填充连通区域
            for coord in coords:
                colored_regions[coord[0], coord[1]] = np.array(color) * 255

        # 显示结果
        plt.imshow(colored_regions, cmap='hsv')
        plt.axis('off')

        # 创建色标
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 180))
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(0, 180, n_colors + 1), boundaries=np.arange(-5, 186, 10))
        cbar.set_label('Orientation Angle (Degrees)')

        plt.show()
    def color_regions_by_orientation_with_colorbar_adjusted(self, image_gray):
        # Reapplying adaptive histogram equalization with a lower clip limit to enhance contrast
        image_eq_enhanced = exposure.equalize_adapthist(image_gray, clip_limit=0.01)

        # Convert the enhanced image back to uint8 format
        image_eq_enhanced_uint8 = (image_eq_enhanced * 255).astype(np.uint8)
        # Use a threshold to create a binary image
        thresh_enhanced = filters.threshold_otsu(image_eq_enhanced_uint8)
        binary_image_enhanced = image_eq_enhanced_uint8 > thresh_enhanced

        # Perform connected component analysis to label the regions
        labeled_image_enhanced, num_features_enhanced = measure.label(binary_image_enhanced, background=0,
                                                                      return_num=True)
        props_enhanced = measure.regionprops(labeled_image_enhanced)

        # 创建一个三通道的图像以显示彩色区域
        colored_regions = np.zeros((image_eq_enhanced_uint8.shape[0], image_eq_enhanced_uint8.shape[1], 3),
                                   dtype=np.uint8)

        # Define a color map
        n_colors = 18  # 180° / 10° = 18 sections
        cmap = plt.cm.get_cmap('hsv', n_colors)

        for prop in props_enhanced:
            # 获取区域的像素坐标
            coords = prop.coords

            # 应用PCA分析
            pca = PCA(n_components=2)
            pca.fit(coords)
            first_component = pca.components_[0]

            # 确保方向在0-180°范围内
            if first_component[0] > 0:
                first_component = -first_component

            # 计算朝向角度，相对于x轴负半轴
            angle = np.arctan2(first_component[1], -first_component[0])
            angle_degrees = np.degrees(angle)
            if angle_degrees < 0:
                angle_degrees += 360
            angle_degrees = (angle_degrees + 270) % 360  # 调整基准线至x轴负半轴

            # 限制角度在0-180°范围内
            angle_degrees = angle_degrees % 180

            # 选择颜色
            color_index = int(angle_degrees / 10)  # 每10°分一个区间
            color = cmap(color_index)[:3]  # 提取RGB颜色

            # 对应颜色填充连通区域
            for coord in coords:
                colored_regions[coord[0], coord[1]] = (np.array(color) * 255).astype(np.uint8)

        # 显示结果和色标
        plt.imshow(colored_regions)
        plt.axis('off')

        # 创建色标
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 180))
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(0, 180, n_colors + 1), boundaries=np.arange(-5, 186, 10))
        cbar.set_label('Orientation Angle (Degrees)')

        plt.show()

    def edge_detection(self,img):
        edges = feature.canny(img)
        # Display the edges using matplotlib
        # plt.figure(figsize=(12, 12))
        # plt.imshow(edges, cmap='gray')
        # plt.title('Edges of White Regions')
        # plt.axis('off')
        # plt.show()
        return edges
    def plot_edge_in_orign_img(self,edges):
        img_copy=self.img_with_shadow.copy()
        img_copy[edges != 0] = [255, 255, 255]
        # Now, let's plot the edges on the blank image using matplotlib
        plt.figure(figsize=(12, 12))
        plt.imshow(cv.cvtColor(img_copy, cv.COLOR_BGR2RGB))  # Convert to RGB for correct color display
        plt.title('Extracted Edges on Blank Image')
        plt.axis('off')  # Hide axes
        plt.show()
    def plot_img(self):
        if self.img is not None:
            img_output = self.enhance_image_contrast(self.img)
            #self.resize_img_show(img_output, scale=6)
            img_output_gray_binary=self.gray_binary_img(img_output)
            #self.connected_component_analysis(img_output_gray_binary)
            # edges=self.edge_detection(img_output_gray_binary)
            # self.plot_edge_in_orign_img(edges)
            # self.connected_component_analysis_and_ellipse_fitting(img_output_gray_binary)
            self.pca_analysis_of_concave_polygon_orientation(img_output_gray_binary)
            # self.color_regions_by_orientation(img_output_gray_binary)
            # self.color_regions_by_orientation_with_colorbar_adjusted(img_output_gray_binary)

if __name__ == '__main__':
    path_of_img_no_shadow = os.path.join('..', 'data', 'data_0030160180_sun_diseable.jpg')
    path_of_img_with_shadow=os.path.join('..', 'data', 'data_0030160180.jpg')
    path_of_img_with_ramp_shader=os.path.join('..', 'data', 'data_0030160180_sun_diseable_ramp.jpg')
    result = ProssNoSunshine(path_of_img_no_shadow,path_of_img_with_shadow,path_of_img_with_ramp_shader)
    result.plot_img()
    cv.waitKey(0)
    cv.destroyAllWindows()
