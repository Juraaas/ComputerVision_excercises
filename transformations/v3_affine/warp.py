from v3_affine.affine import affine
from v3_affine.transform_matrices import compute_output_shape

def warp(img, M, interpolation="bilinear", mode="black"):
    h, w, _ = img.shape

    output_shape, M_final = compute_output_shape(M, w, h)

    return affine(img, M_final, output_shape, interpolation, mode)