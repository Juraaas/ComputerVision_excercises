import numpy as np
import math

def bilinear(img, src_i, src_j):
    h, w, c = img.shape

    i0 = math.floor(src_i)
    j0 = math.floor(src_j)

    i1 = i0 + 1
    j1 = j0 + 1

    i0 = np.clip(i0, 0, h - 1)
    i1 = np.clip(i1, 0, h - 1)
    j0 = np.clip(j0, 0, w - 1)
    j1 = np.clip(j1, 0, w - 1)

    di = src_i - math.floor(src_i)
    dj = src_j - math.floor(src_j)

    TL = img[i0, j0]
    TR = img[i0, j1]
    BL = img[i1, j0]
    BR = img[i1, j1]

    w_tl = (1 - di) * (1 - dj)
    w_tr = (1 - di) * dj
    w_bl = di * (1 - dj)
    w_br = di * dj

    pixel = (
        TL * w_tl +
        TR * w_tr +
        BL * w_bl +
        BR * w_br
    )

    return np.clip(pixel, 0, 255).astype(np.uint8)


def nearest(img, src_i, src_j):
    h, w, c = img.shape
    src_i = round(src_i)
    src_j = round(src_j)

    src_i = max(0, min(src_i, h-1))
    src_j = max(0, min(src_j, w-1))
    return img[src_i, src_j]