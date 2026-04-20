import numpy as np
import matplotlib.pyplot as plt
from transformations.v1_basic.interpolation import bilinear

def affine(img, transform_matrix, output_shape):
    h, w, c = img.shape
    output = np.zeros((output_shape), dtype=np.uint8)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            x, y = j, i
            p_output = np.array([x, y, 1])
            src_x, src_y, src_w = np.dot(np.linalg.inv(transform_matrix), p_output)
            src_x = src_x / src_w
            src_y = src_y / src_w
            if 0 <= src_y < h and 0 <= src_x < w:
                output[i, j] = bilinear(img, src_y, src_x)
            else:
                output[i, j] = [0, 0, 0] 
    plt.imshow(output)
    plt.show()        
    return output