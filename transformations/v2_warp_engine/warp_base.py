import numpy as np
import matplotlib.pyplot as plt
from transformations.v1_basic.interpolation import bilinear

def warp(img, transformation, output_shape):
    h, w, c = img.shape
    output = np.zeros((output_shape), dtype=np.uint8)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            src_i, src_j = transformation(i, j)
            if 0 <= src_i < h and 0 <= src_j < w:
                output[i, j] = bilinear(img, src_i, src_j)
            else:
                output[i, j] = [0, 0, 0]
    plt.imshow(output)
    plt.show()        
    return output