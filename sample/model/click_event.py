import cv2
import os
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map the coordinates back to the original image's dimensions
        scale_x, scale_y = img.shape[1] / resized_img.shape[1], img.shape[0] / resized_img.shape[0]
        original_x, original_y = int(x * scale_x), int(y * scale_y)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Original: ({original_x}, {original_y}), Resized: ({x}, {y})'
        cv2.putText(resized_img, text, (10, 30), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', resized_img)
path_of_img_no_shadow = os.path.join('..', 'data', 'data_0030160180_sun_diseable.jpg')
img = cv2.imread(path_of_img_no_shadow)  # Replace with your image path
scale_percent = 30  # percentage of original size

width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

cv2.imshow('image', resized_img)
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()