import cv2
import numpy as np
from PIL import Image

# Save the image and mask to disk
image_filename = 'D:/software/Python/PythonProject/seamounts_edge_detection/sample/data/imgs/'
mask_filename = 'D:/software/Python/PythonProject/seamounts_edge_detection/sample/data/masks/'


def generate_abstract_image_and_mask(size=(256, 256), num_shapes=3, max_radius=50, blur_intensity=25):
    # Create a blue background (BGR format)
    background_color = (191, 144, 92)  # A soft blue color in BGR format
    image = np.full((size[1], size[0], 3), background_color, dtype=np.uint8)        #先生成一幅背景底图

    # Initialize mask to zeros
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)     #生成一张蒙版图片

    for _ in range(num_shapes):     #需要循环操作但不需要引入循环变量于是引入下划线
        # Random center position
        center = (np.random.randint(blur_intensity, size[0] - blur_intensity),
                  np.random.randint(blur_intensity, size[1] - blur_intensity))      #确定随机中心，函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
        # Random radius
        radius = np.random.randint(max_radius // 2, max_radius)     #

        # Draw a white circle on the mask
        cv2.circle(mask, center, radius, (255), -1)     #绘制椭圆
    cv2.imshow('mask', mask)
    # Blur the mask to create a smooth transition
    mask_blurred = cv2.GaussianBlur(mask, (0, 0), blur_intensity).astype(float) / 255.0
    cv2.imshow('mask_blurred',mask_blurred)
    # Create a transition layer using the blurred mask
    transition_layer = np.stack([mask_blurred] * 3, axis=-1)
    cv2.imshow('transition_layer', transition_layer)
    # Transition from the background color to white using the transition layer
    white_area = np.full((size[1], size[0], 3), (255, 255, 255), dtype=np.uint8)
    cv2.imshow('white_area', white_area)
    image = np.uint8(background_color * (1 - transition_layer) + white_area * transition_layer)
    cv2.imshow('image', image)
    cv2.imshow('np.uint8(mask_blurred * 255)', np.uint8(mask_blurred * 255))
    return image, np.uint8(mask_blurred * 255)

for i in range(1, 2):        #批量生成数据集
    # Generate the image and mask
    image, mask = generate_abstract_image_and_mask()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Convert the NumPy array mask to a PIL image
    # ret, binary = cv2.threshold(mask, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    # images_save_path = image_filename + str(i)+'.png'
    # mask_image=Image.fromarray(binary)
    # maks_save_path = mask_filename  + str(i)+'.gif'
    # cv2.imwrite(images_save_path,image)
    # mask_image.save(maks_save_path,'GIF  ')