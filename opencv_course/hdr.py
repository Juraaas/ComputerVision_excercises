import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Capture multiple exposures 

def read_images_and_times():
    filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
    times = np.array([1 / 30.0, 0.25, 2.5, 15.0], dtype=np.float32)

    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
    return images, times

# Step 2: Align images

images, times = read_images_and_times()

alignMTB = cv2.createAlignMTB() # Median threshold bitmap
alignMTB.process(images, images)

# Step 3: Estimate Camera Response Function

calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

x = np.arange(256, dtype=np.uint8)
y = np.squeeze(responseDebevec)

ax = plt.figure(figsize=(30, 10))
plt.title("Debevec Inverse Camera Response Function", fontsize=24)
plt.xlabel("Measured Pixel Value", fontsize=22)
plt.ylabel("Calibrated Intensity", fontsize=22)
plt.xlim([0, 260])
plt.grid()
plt.plot(x, y[:, 0], "b", x, y[:, 1], "g", x, y[:, 2], "r")
plt.show()

# Step 4: Merge exposure into an HDR Image

mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

# Step 5: Tonemapping

tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago

plt.figure(figsize=(20,10))
plt.imshow(np.clip(ldrDrago, 0, 1))
plt.axis("off")
plt.show()

cv2.imwrite("ldrDrago.jpg", ldrDrago * 255)
print("saved drago")


tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)

plt.figure(figsize=(20,10))
plt.imshow(np.clip(ldrReinhard, 0, 1))
plt.axis("off")
plt.show()

cv2.imwrite("ldrReinhard.jpg", ldrReinhard * 255)
print("saved reinhard")

tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = 3 * ldrMantiuk

plt.figure(figsize=(20,10))
plt.imshow(np.clip(ldrMantiuk, 0, 1))
plt.axis("off")
plt.show()

cv2.imwrite("ldrMantiuk.jpg", ldrMantiuk * 255)
print("saved mantiuk")