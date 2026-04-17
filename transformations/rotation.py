import numpy as np
import math
import matplotlib.pyplot as plt
from interpolation import bilinear

def rotate_nearest(img, angle):
    h, w, c = img.shape

    cx = w // 2
    cy = h // 2

    output = np.zeros_like(img)

    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)

    for i in range(h):
        for j in range(w):

            x = j - cx
            y = i - cy

            src_x = x * cos_a - y * sin_a
            src_y = x * sin_a + y * cos_a

            src_j = src_x + cx
            src_i = src_y + cy

            if 0 <= src_i < h and 0 <= src_j < w:
                output[i, j] = img[src_i, src_j]
            else:
                output[i, j] = [0, 0, 0]

    plt.imshow(output)
    plt.show()

    return 0

def rotate_bilinear(img, angle):
    h, w, c = img.shape

    cx = w // 2
    cy = h // 2

    output = np.zeros_like(img)

    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)

    for i in range(h):
        for j in range(w):

            x = j - cx
            y = i - cy

            src_x = x * cos_a - y * sin_a
            src_y = x * sin_a + y * cos_a

            src_j = src_x + cx
            src_i = src_y + cy

            if 0 <= src_i < h and 0 <= src_j < w:
                output[i, j] = bilinear(img, src_i, src_j)
            else:
                output[i, j] = [0, 0, 0]

    plt.imshow(output)
    plt.show()

    return output