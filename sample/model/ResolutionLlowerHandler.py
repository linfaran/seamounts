import cv2
import os
path_of_img_with_shadow=os.path.join('..', 'data', 'img_orign_lab_watershed.png')
img=cv2.imread(path_of_img_with_shadow)
pic=cv2.resize(img,(4800,7200))
cv2.imwrite('resize_img_orign_lab_watershed.jpg',pic)


