import numpy as np
import matplotlib.pyplot as plt
import cv2
import filters

def show(img, title="Image"):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def get_pixel(img, x, y, mode="zero"):
    h, w, _ = img.shape
    if x < 0 or y < 0 or x >= h or y >= w:
        if mode == "zero":
            pixel = np.array([0.0, 0.0, 0.0])
            return pixel
        if mode == "clamp":
            x = min(max(x, 0), h-1)
            y = min(max(y, 0), w-1)
            pixel = img[x, y]
            return pixel
    else:
        pixel = img[x, y]
        return pixel

def pad_image(img, pad, mode="zero"):
    h, w, c = img.shape
    padded = np.zeros((h+2*pad, w+2*pad, c),dtype=img.dtype)

    if mode == "zero":
        padded[pad:pad+h, pad:pad+w] = img

    elif mode == "clamp":
        padded[pad:pad+h, pad:pad+w] = img

        padded[:pad, pad:pad+w] = img[0:1, :]
        padded[pad+h:, pad:pad+w] = img[-1:, :]

        padded[:, :pad] = padded[:, pad:pad+1]
        padded[:, pad+w:] = padded[:, pad+w-1:pad+w]

        padded[:pad, :pad] = img[0, 0]
        padded[:pad, -pad:] = img[0, -1]
        padded[-pad:, :pad] = img[-1, 0]
        padded[-pad:, -pad:] = img[-1, -1] 
    
    else:
        return "Invalid padding mode"
        
    return padded
            
def conv2d(img, kernel, padding="clamp"):
    h, w, c = img.shape
    k_h, k_w = kernel.shape

    if k_h != k_w:
        return "Invalid kernel"
    
    output = np.zeros((h, w, 3), dtype=np.uint8)
    offset_h = k_h // 2
    padded = pad_image(img, offset_h, padding)
    for i in range(h):
        for j in range(w):
            res = np.zeros(3, dtype=np.float32)
            for ki in range(k_h):
                for kj in range(k_w):
                    x = i+ki
                    y = j+kj
                    pixel = padded[x,y]
                    res += pixel * kernel[ki][kj]
            output[i, j] = np.clip(res, 0, 255)
    return output

img_bgr = cv2.imread("images/patrick.jpg", cv2.IMREAD_COLOR)
img_test = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#show(conv2d(img_test, filters.blur_3, padding="zero"), "Blur 3x3")
#show(conv2d(img_test, filters.edge, padding="zero"), "Edge")
#show(conv2d(img_test, filters.sharpen, padding="zero"), "Sharpen")

img_blur = conv2d(img_test, filters.gaussian_3)
gx = conv2d(img_blur, filters.sobel_x).astype(np.float32)
gy = conv2d(img_blur, filters.sobel_y).astype(np.float32)
edge = np.sqrt(gx**2 + gy**2)
edge = (edge / edge.max()) * 255
edge = edge.astype(np.uint8)

show(img_blur)
show(gx)
show(gy)
show(edge)