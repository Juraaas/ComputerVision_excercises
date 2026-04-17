import numpy as np
import matplotlib.pyplot as plt
from interpolation import bilinear

def scaling_nearest(img, scale):
    h, w, c = img.shape

    n_width = int(w * scale)
    n_height = int(h * scale)

    output = np.zeros((n_height, n_width, 3), dtype=np.uint8)

    for i in range(n_height):
        for j in range(n_width):

            src_i = int(i / scale)
            src_j = int(j / scale)

            if 0 <= src_i < h and 0 <= src_j < w:
                output[i, j] = img[src_i, src_j]
            else:
                output[i, j] = [0, 0, 0]

    plt.imshow(output)
    plt.show()

    return output


def scaling_bilinear(img, scale):
    h, w, c = img.shape

    n_width = int(w * scale)
    n_height = int(h * scale)
    
    output = np.zeros((n_height, n_width, 3), dtype=np.uint8)

    for i in range(n_height):
        for j in range(n_width):

            src_i = i / scale
            src_j = j / scale

            if 0 <= src_i < h and 0 <= src_j < w:
                output[i, j] = bilinear(img, src_i, src_j)
            else:
                output[i, j] = [0, 0, 0]

    plt.imshow(output)
    plt.show()

    return output