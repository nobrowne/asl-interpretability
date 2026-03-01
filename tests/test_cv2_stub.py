"""Tests for the numpy cv2 stub (stubs/cv2.py).

Verifies that getPerspectiveTransform and perspectiveTransform produce
correct results for cases with known analytic answers, and that they're
mutually consistent (round-trip). These tests run on CPU with no
dependencies beyond numpy and pytest.

Run via SLURM:
    sbatch slurm/tests.sh
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the stub is imported, not a real cv2 install.
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _TESTS_DIR.parent

sys.path.insert(0, str(_ANALYSIS_DIR / "stubs"))

import cv2  # noqa: E402 — must be the stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Reference implementation: apply 3x3 H to (N, 2) pts, return (N, 2)."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # (N, 3)
    dst_h = (H @ pts_h.T).T                           # (N, 3)
    return dst_h[:, :2] / dst_h[:, 2:3]               # (N, 2)


def _four_corners() -> np.ndarray:
    """Unit-square corners as (4, 2) float32 — a convenient default src."""
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# getPerspectiveTransform — known analytic cases
# ---------------------------------------------------------------------------


def test_identity_transform() -> None:
    """src == dst should return the identity matrix (up to float32 tolerance)."""
    pts = _four_corners()
    H = cv2.getPerspectiveTransform(pts, pts)

    assert H.shape == (3, 3)
    assert H.dtype == np.float32
    np.testing.assert_allclose(H, np.eye(3, dtype=np.float32), atol=1e-5)


def test_pure_translation() -> None:
    """A translation by (dx, dy) should produce the translation homography."""
    dx, dy = 0.4, 0.3
    src = _four_corners()
    dst = src + np.array([dx, dy], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    expected = np.array(
        [[1, 0, dx],
         [0, 1, dy],
         [0, 0,  1]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(H, expected, atol=1e-5)


def test_pure_scale() -> None:
    """Uniform scaling by factor s should produce diag(s, s, 1)."""
    s = 2.5
    src = _four_corners()
    dst = (src * s).astype(np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    expected = np.diag([s, s, 1.0]).astype(np.float32)
    np.testing.assert_allclose(H, expected, atol=1e-5)


def test_spoter_squeeze_augmentation() -> None:
    """Reproduce SPOTER's squeeze augmentation and verify H analytically.

    SPOTER computes:
        src  = [(0,1), (1,1), (0,0), (1,0)]
        dest = [(L,1), (1-R,1), (L,0), (1-R,0)]
    where L = move_left, R = move_right.

    This is a pure x-scaling + x-translation (affine, no true perspective):
        x' = L + x*(1 - L - R)
        y' = y
    So H = [[1-L-R, 0, L], [0, 1, 0], [0, 0, 1]].
    """
    move_left, move_right = 0.1, 0.15
    src = np.array([(0, 1), (1, 1), (0, 0), (1, 0)], dtype=np.float32)
    dst = np.array(
        [
            (move_left,          1),
            (1 - move_right,     1),
            (move_left,          0),
            (1 - move_right,     0),
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src, dst)

    scale_x = 1.0 - move_left - move_right
    expected = np.array(
        [[scale_x, 0, move_left],
         [0,       1, 0        ],
         [0,       0, 1        ]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(H, expected, atol=1e-5)


def test_recover_arbitrary_homography() -> None:
    """Build a non-trivial homography, warp 4 points, then recover H via DLT.

    Tests that getPerspectiveTransform inverts perspectiveTransform correctly
    for a true projective (non-affine) transform.
    """
    H_true = np.array(
        [[1.2,  0.3, 0.1],
         [0.1,  0.9, 0.2],
         [0.05, 0.02, 1.0]],
        dtype=np.float64,
    )

    src = _four_corners().astype(np.float64)
    dst = _apply_homography(H_true, src).astype(np.float32)
    src = src.astype(np.float32)

    H_recovered = cv2.getPerspectiveTransform(src, dst)

    # Normalize both for comparison (H defined up to scale).
    H_true_norm = (H_true / H_true[2, 2]).astype(np.float32)
    np.testing.assert_allclose(H_recovered, H_true_norm, atol=1e-4)


def test_output_shape_and_dtype() -> None:
    """getPerspectiveTransform always returns float32 shape (3, 3)."""
    src = _four_corners()
    dst = src.copy()
    dst[0] += 0.1
    H = cv2.getPerspectiveTransform(src, dst)
    assert H.shape == (3, 3)
    assert H.dtype == np.float32


# ---------------------------------------------------------------------------
# perspectiveTransform — known analytic cases
# ---------------------------------------------------------------------------


def test_perspective_transform_identity() -> None:
    """Identity homography leaves all points unchanged."""
    H = np.eye(3, dtype=np.float32)
    pts = np.random.default_rng(0).random((10, 1, 2)).astype(np.float32)
    result = cv2.perspectiveTransform(pts, H)
    np.testing.assert_allclose(result, pts, atol=1e-6)


def test_perspective_transform_translation() -> None:
    """Translation homography shifts every point by (dx, dy)."""
    dx, dy = 3.0, -1.5
    H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
    pts = np.array([[[0.0, 0.0]], [[1.0, 2.0]], [[-0.5, 0.5]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pts, H)
    expected = pts + np.array([dx, dy], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_perspective_transform_preserves_shape() -> None:
    """Output shape must equal input shape for any (..., 2) input."""
    H = np.eye(3, dtype=np.float32)
    for shape in [(1, 1, 2), (5, 1, 2), (3, 4, 2), (20, 54, 2)]:
        pts = np.zeros(shape, dtype=np.float32)
        assert cv2.perspectiveTransform(pts, H).shape == shape


def test_perspective_transform_spoter_landmark_shape() -> None:
    """Test the exact call shape SPOTER uses: (T, num_body_landmarks, 2)."""
    T, B = 62, 8  # typical frame count, BODY_IDENTIFIERS count
    H = np.eye(3, dtype=np.float32)
    landmarks = np.random.default_rng(1).random((T, B, 2)).astype(np.float32)
    result = cv2.perspectiveTransform(landmarks, H)
    assert result.shape == (T, B, 2)
    assert result.dtype == np.float32


def test_perspective_transform_origin_lookup() -> None:
    """Test the specific (1, 1, 2) origin-lookup pattern SPOTER uses.

    SPOTER does:
        augmented_zero = cv2.perspectiveTransform(
            np.array([[[0, 0]]], dtype=np.float32), mtx
        )[0][0]
    and compares the result elementwise. Verify shape and indexing work.
    """
    dx, dy = 0.1, 0.2
    H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
    result = cv2.perspectiveTransform(np.array([[[0.0, 0.0]]], dtype=np.float32), H)
    assert result.shape == (1, 1, 2)
    zero_mapped = result[0][0]  # shape (2,)
    np.testing.assert_allclose(zero_mapped, [dx, dy], atol=1e-6)


def test_perspective_transform_output_dtype() -> None:
    """Output is always float32 regardless of input/H dtype."""
    H = np.eye(3, dtype=np.float64)
    pts = np.ones((4, 1, 2), dtype=np.float64)
    result = cv2.perspectiveTransform(pts, H)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------


def test_roundtrip_get_then_apply() -> None:
    """getPerspectiveTransform then perspectiveTransform recovers dst points."""
    src = _four_corners()
    dst = np.array([[0.1, 0.9], [0.8, 0.85], [0.75, 0.05], [0.05, 0.1]],
                   dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    recovered = cv2.perspectiveTransform(src.reshape(4, 1, 2), H).reshape(4, 2)

    np.testing.assert_allclose(recovered, dst, atol=1e-5)


def test_roundtrip_arbitrary_projective() -> None:
    """Round-trip with a true projective (non-affine) transform."""
    H_true = np.array(
        [[0.9, 0.1, 0.05],
         [0.0, 1.1, 0.10],
         [0.02, 0.01, 1.0]],
        dtype=np.float32,
    )
    src = np.array([[0.1, 0.2], [0.8, 0.15], [0.75, 0.85], [0.2, 0.9]],
                   dtype=np.float32)
    dst = cv2.perspectiveTransform(src.reshape(4, 1, 2), H_true).reshape(4, 2)

    H_recovered = cv2.getPerspectiveTransform(src, dst)
    recovered = cv2.perspectiveTransform(src.reshape(4, 1, 2), H_recovered).reshape(4, 2)

    np.testing.assert_allclose(recovered, dst, atol=1e-5)
