import numpy as np
import matplotlib.pyplot as plt

def translate_forward(img, tx, ty):
    h, w, c = img.shape

    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):

            output[i + tx, j + ty] = img[i, j]

    plt.imshow(output)
    plt.show()

    return output

def translate_backward(img, tx, ty):
    h, w, c = img.shape

    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):

            src_i = i - tx
            src_j = j - ty

            if 0 <= src_i < h and 0 <= src_j < w:
                output[i, j] = img[i - tx, j - ty]
            else:
                output[i, j] = [0, 0, 0]

    plt.imshow(output)
    plt.show()
    
    return output