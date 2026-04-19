import math

def translate(tx, ty):
    def transform(i, j):
        return i - ty, j - tx
    return transform

def rotation(cx, cy, angle):
    def transform(i, j):
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        
        x = j - cx
        y = i - cy

        src_x = x * cos_a - y * sin_a
        src_y = x * sin_a + y * cos_a

        return src_y + cy, src_x + cx
    return transform

def scale(factor):
    def transform(i, j):
        return i / factor, j / factor
    return transform