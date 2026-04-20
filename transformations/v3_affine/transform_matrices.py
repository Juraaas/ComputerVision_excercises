import numpy as np

def translate_mat(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ])

def rotation_mat(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])

def scale_mat(scale):
    return np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ])

def compute_output_shape(M, w, h):
    corners = [[0,0,1], [0,h,1], [w,0,1], [w,h,1]]
    xs = []
    ys = []
    for corner in corners:
        res_x, res_y, res_w = np.dot(M, corner)
        res_x /= res_w
        res_y /= res_w
        xs.append(res_x)
        ys.append(res_y)
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    shift_x = -x_min
    shift_y = -y_min
    T_shift = translate_mat(shift_x, shift_y)
    M_final = T_shift @ M
    return (width, height, 3), M_final