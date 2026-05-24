import numpy as np

same = np.array([
    [0,0,0],
    [0,1,0],
    [0,0,0]
])

blur_3 = np.ones((3,3)) / 9
blur_5 = np.ones((5,5)) / 25
blur_7 = np.ones((7,7)) / 49

sharpen = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

edge = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

sobel_x = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

sobel_y = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])

emboss = np.array([
    [-2,-1,0],
    [-1,1,1],
    [0,1,2]
])

gaussian_3 = np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
]) / 16

