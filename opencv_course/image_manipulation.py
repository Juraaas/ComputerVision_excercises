import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from PIL import Image

test = cv2.imread("images/checkerboard.jpg", 0)
img_bgr= cv2.imread("images/landscape.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

# Brightness modification

matrix = np.ones(img_rgb.shape, dtype="uint8") * 50

img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)

plt.figure(figsize=(18,5))
plt.subplot(131)
plt.imshow(img_rgb_darker)
plt.title("Darker")
plt.subplot(132)
plt.imshow(img_rgb)
plt.title("Original")
plt.subplot(133)
plt.imshow(img_rgb_brighter)
plt.title("Brighter")
plt.show()

# Contrast modification

matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb), matrix2))
img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))

plt.figure(figsize=(18,5))
plt.subplot(131)
plt.imshow(img_rgb_darker)
plt.title("Lower Contrast")
plt.subplot(132)
plt.imshow(img_rgb)
plt.title("Original")
plt.subplot(133)
plt.imshow(img_rgb_brighter)
plt.title("Higher Contrast")
plt.show()

img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))
img_rgb_lower = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))

plt.figure(figsize=(18,5))
plt.subplot(131)
plt.imshow(img_rgb_lower)
plt.title("Lower Contrast")
plt.subplot(132)
plt.imshow(img_rgb)
plt.title("Original")
plt.subplot(133)
plt.imshow(img_rgb_higher)
plt.title("Higher Contrast")
plt.show()

# Image Thresholding

img_read = cv2.imread("patrick.jpg", cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(18,5))
plt.subplot(121)
plt.imshow(img_read, cmap="gray")
plt.title("Original")
plt.subplot(122)
plt.imshow(img_thresh, cmap="gray")
plt.title("Thresholded")
plt.show()

retval, img_thresh_glob1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)
retval, img_thresh_glob2 = cv2.threshold(img_read, 130, 255, cv2.THRESH_BINARY)
img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 7)

plt.figure(figsize=(18,15))
plt.subplot(221)
plt.imshow(img_read, cmap="gray")
plt.title("Original")
plt.subplot(222)
plt.imshow(img_thresh_glob1, cmap="gray")
plt.title("Thresholded (global: 50)")
plt.subplot(223)
plt.imshow(img_thresh_glob2, cmap="gray")
plt.title("Thresholded (global: 130)")
plt.subplot(224)
plt.imshow(img_thresh_adp, cmap="gray")
plt.title("Thresholded (adaptive)")
plt.show()

# Bitwise operations

img_rec = cv2.imread("images/rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread("images/circle.jpg", cv2.IMREAD_GRAYSCALE)
desired_width = 882
desired_height = 360
dim = (desired_width, desired_height)
cir_res = cv2.resize(img_cir, dsize=dim, interpolation=cv2.INTER_AREA)
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.imshow(img_rec, cmap="gray")
plt.subplot(122)
plt.imshow(img_cir, cmap="gray")
plt.show()
result = cv2.bitwise_and(img_rec, cir_res, mask=None)

plt.imshow(result, cmap="gray")
plt.show()

result = cv2.bitwise_or(img_rec, cir_res, mask=None)
plt.imshow(result, cmap="gray")
plt.show()

result = cv2.bitwise_xor(img_rec, cir_res, mask=None)
plt.imshow(result, cmap="gray")
plt.show()

# Application: Logo Manipulation

img_bgr = cv2.imread("images/coke.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(img_rgb.shape)
logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]

img_background_bgr = cv2.imread("images/background.jpg")
img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)

aspect_ratio = logo_w / img_background_rgb.shape[1]
dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))

img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(img_mask, cmap="gray")
plt.show()

img_mask_inv = cv2.bitwise_not(img_mask)
plt.imshow(img_mask_inv, cmap="gray")
plt.show()

img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)
plt.imshow(img_background)
plt.show()

img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inv)
plt.imshow(img_foreground)
plt.show()

# Merge foreground and background
result = cv2.add(img_background, img_foreground)
plt.imshow(result)
plt.show()
cv2.imwrite("logo_final.png", result[:,:,::-1])
