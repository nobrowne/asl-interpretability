"""Minimal cv2 stub for SPOTER on FIPS-enabled systems.

opencv-python-headless bundles its own OpenSSL 1.1.1k, which fails the
system FIPS self-test at import time even though cv2 is only used for
two pure-geometry functions. This stub reimplements those two functions
in numpy so we can drop the opencv dependency entirely.

Functions implemented (the only ones used by SPOTER's augmentations):
  - getPerspectiveTransform(src, dst)  ->  3x3 homography matrix
  - perspectiveTransform(src, H)       ->  transformed points array

All other cv2 attributes raise AttributeError on access.
"""

import numpy as np


def getPerspectiveTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute the 3x3 perspective transform matrix for 4 point pairs.

    Uses the Direct Linear Transform (DLT): for each correspondence
    (x, y) -> (x', y'), the homography H satisfies:
        [x', y', 1]^T ~ H @ [x, y, 1]^T
    Rearranged into a homogeneous linear system A*h = 0 (8 equations,
    9 unknowns), solved via SVD. Matches OpenCV's output exactly.

    Args:
        src: float32 array of shape (4, 2) — source points.
        dst: float32 array of shape (4, 2) — destination points.

    Returns:
        float32 array of shape (3, 3) — the homography matrix.
    """
    A = np.zeros((8, 9), dtype=np.float64)
    for i in range(4):
        x,  y  = float(src[i, 0]), float(src[i, 1])
        xp, yp = float(dst[i, 0]), float(dst[i, 1])
        A[2 * i]     = [ x,  y,  1,  0,  0,  0, -x * xp, -y * xp, -xp]
        A[2 * i + 1] = [ 0,  0,  0,  x,  y,  1, -x * yp, -y * yp, -yp]

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    return H.astype(np.float32)


def perspectiveTransform(src: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a perspective (homography) transform to an array of 2D points.

    Accepts input of any shape (..., 2) and returns the same shape,
    matching OpenCV's perspectiveTransform semantics.

    Args:
        src: float32 array of shape (..., 2) — input points.
        H:   float32 or float64 array of shape (3, 3) — homography matrix.

    Returns:
        float32 array of same shape as src — transformed points.
    """
    orig_shape = src.shape
    pts = src.reshape(-1, 2).astype(np.float64)          # (N, 2)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])     # (N, 3) homogeneous
    dst_h = (H.astype(np.float64) @ pts_h.T).T           # (N, 3)
    dst = dst_h[:, :2] / dst_h[:, 2:3]                   # (N, 2) dehomogenize
    return dst.reshape(orig_shape).astype(np.float32)
