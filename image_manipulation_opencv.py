import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from PIL import Image

test = cv2.imread("checkerboard.jpg", 0)
img_bgr= cv2.imread("landscape.jpg", cv2.IMREAD_COLOR)

print(test.shape)
plt.imshow(test, cmap="gray")
plt.show()


test_color = cv2.imread("checkerboard.jpg", 1) # OpenCV stores images in BGR format
plt.imshow(test_color)
plt.show()

test_color_reversed = test_color[:, :, ::-1] # reversing order of color channels
plt.imshow(test_color_reversed)
plt.show()

b, g, r = cv2.split(img_bgr)

plt.figure(figsize=(20,5))
plt.subplot(141)
plt.imshow(r, cmap="gray")
plt.title("Red Channel")
plt.subplot(142)
plt.imshow(g, cmap="gray")
plt.title("Green Channel")
plt.subplot(143)
plt.imshow(b, cmap="gray")
plt.title("Blue Channel")

img_merged = cv2.merge((b, g, r))
plt.subplot(144)
plt.imshow(img_merged[:, :, ::-1])
plt.title("Merged Output")
plt.show()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)

plt.figure(figsize=(20,5))
plt.subplot(141)
plt.imshow(h, cmap="gray")
plt.title("H Channel")
plt.subplot(142)
plt.imshow(s, cmap="gray")
plt.title("S Channel")
plt.subplot(143)
plt.imshow(v, cmap="gray")
plt.title("V Channel")
plt.subplot(144)
plt.imshow(img_rgb)
plt.title("Original")
plt.show()

# Modified channel H
h_new = h + 10
img_merged = cv2.merge((h_new, s, v))
img_rgb = cv2.cvtColor(img_merged, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(20,5))
plt.subplot(141)
plt.imshow(h, cmap="gray")
plt.title("H Channel")
plt.subplot(142)
plt.imshow(s, cmap="gray")
plt.title("S Channel")
plt.subplot(143)
plt.imshow(v, cmap="gray")
plt.title("V Channel")
plt.subplot(144)
plt.imshow(img_rgb)
plt.title("Modified Image")
plt.show()

cv2.imwrite("morskie_oko.png", img_bgr)

print(test[0,0])

test_copy = test.copy()
test_copy[2:10, 2:10] = 200

plt.imshow(test_copy, cmap="gray")
plt.show()

img_landscape_rgb = img_bgr[:, :, ::-1]

plt.imshow(img_landscape_rgb)
plt.show()

cropped = img_landscape_rgb[0:200, 200:500]
plt.imshow(cropped)
plt.show()

cropped_2x = cv2.resize(cropped, None, fx=2, fy=2)
plt.imshow(cropped_2x)
plt.show()

desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)

resized_cropped = cv2.resize(cropped, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped)
plt.show()

aspect_ratio = desired_width / cropped.shape[1]
desired_height = int(cropped.shape[0] * aspect_ratio)
dim = (desired_width, desired_height)

resized_aspect_cropped = cv2.resize(cropped, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_aspect_cropped)
plt.show()

img_landscape_rgb_flip_horiz = cv2.flip(img_landscape_rgb, 1)
img_landscape_rgb_flip_vert = cv2.flip(img_landscape_rgb, 0)
img_landscape_rgb_flip_both = cv2.flip(img_landscape_rgb, -1)

plt.figure(figsize=(20,5))
plt.subplot(141)
plt.imshow(img_landscape_rgb_flip_horiz)
plt.title("Horizontal Flip")
plt.subplot(142)
plt.imshow(img_landscape_rgb_flip_vert)
plt.title("Vertical Flip")
plt.subplot(143)
plt.imshow(img_landscape_rgb_flip_both)
plt.title("Both Direction Flip")
plt.subplot(144)
plt.imshow(img_landscape_rgb)
plt.title("Original Image")
plt.show()

img_test = cv2.imread("tiger.jpg", cv2.IMREAD_COLOR)
plt.imshow(img_test[:,:,::-1])
plt.show()

image_line = img_test.copy()
cv2.line(image_line, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA)
plt.imshow(image_line[:,:,::-1])
plt.show()

image_circle = img_test.copy()
cv2.circle(image_circle, (100, 300), 50, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
plt.imshow(image_circle[:,:,::-1])
plt.show()

image_rect = img_test.copy()
cv2.rectangle(image_rect, (50, 100), (200, 250), (255, 0, 255), thickness=5, lineType=cv2.LINE_8)
plt.imshow(image_rect[:,:,::-1])
plt.show()

image_text = img_test.copy()
text = "Tiger Laying in Nature"
font_scale = 1.5
font_face = cv2.FONT_HERSHEY_PLAIN
font_color = (0, 255, 0)
font_thickness = 2

cv2.putText(image_text, text, (0, 420), font_face,
            font_scale, font_color, font_thickness, cv2.LINE_AA)
plt.imshow(image_text[:,:,::-1])
plt.show()