import numpy as np
import matplotlib.pyplot as plt
from transformations.v1_basic.interpolation import bilinear, nearest

def affine_mode(img, transform_matrix, output_shape, interpolation="nearest", mode="black"):
    h, w, c = img.shape
    output = np.zeros((output_shape), dtype=np.uint8)
    inverse = np.linalg.inv(transform_matrix)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            x, y = j, i
            p_output = np.array([x, y, 1])
            src_x, src_y, src_w = np.dot(inverse, p_output)
            src_x = src_x / src_w
            src_y = src_y / src_w
            if mode == "clamp":
                src_x = np.clip(src_x, 0, w - 1)
                src_y = np.clip(src_y, 0, h - 1)
            elif mode == "black":
                if src_x < 0 or src_x >= w or src_y < 0 or src_y >= h:
                    continue
            else:
                raise ValueError("mode must be 'black' or 'clamp'")

            if interpolation == "nearest":
                output[i, j] = nearest(img, src_y, src_x)
            elif interpolation == "bilinear":
                output[i, j] = bilinear(img, src_y, src_x)
            else:
                raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    plt.imshow(output)
    plt.show()        
    return output