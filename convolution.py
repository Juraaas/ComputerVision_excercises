import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((200,200, 3), dtype=np.uint8)
for i in range(80, 120):
    for j in range(80, 120):
        image[i, j] = [255, 255, 255]
        image[i+40, j+40] = [255, 255, 255]

image2 = np.zeros((200, 200, 3), dtype=np.uint8)
image2[60:140, 60:140] = [255, 255, 255]
image2[:, 100] = [255, 0, 0]
image2[100, :] = [0, 255, 0]
image2[0:10, 0:10] = [0, 0, 255]

image3 = np.zeros((50,50,3))
image3[0,0] = [255,255,255]

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
            
def conv2d(img, kernel, padding="clamp"):
    h, w, c = img.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    offset = 1
    for i in range(h):
        for j in range(w):
            res = np.array([0.0, 0.0, 0.0])
            for ki in range(3):
                for kj in range(3):
                    x = i+ki-offset
                    y = j+kj-offset
                    pixel = get_pixel(img, x, y, padding)
                    res += pixel * kernel[ki][kj]
            res = np.clip(res, 0, 255).astype(np.uint8)
            output[i, j] = res
    plt.imshow(output)
    plt.show()

    return output

kernel_same = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
])

kernel_blur = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
]) / 9

kernel_edge = np.array([
 [1,0,-1],
 [1,0,-1],
 [1,0,-1]
])

print(conv2d(image3, kernel_blur, padding="zero"))
print(conv2d(image3, kernel_blur, padding="clamp"))