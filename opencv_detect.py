import timeit
import math

import cv2
import numpy as np


def extract_patches(img, kpts, N, mag_factor):
    """
    Rectifies patches around openCV keypoints, and returns patches tensor
    """
    patches = []
    for kp in kpts:
        x, y = kp.pt
        s = kp.size
        # a = kp.angle
        # print(s * mag_factor)
        a = 0

        s = mag_factor * s / N
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix(
            [
                [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
                [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y],
            ]
        )

        patch = cv2.warpAffine(img, M, (N, N), flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        patches.append(patch)

    return patches


def detector_runtime_test(img, detector):
    _t0 = timeit.default_timer()
    keypoints = detector.detect(img, None)
    _t1 = timeit.default_timer()
    if len(keypoints) > 0:
        _ = extract_patches(img, keypoints, 32, mag_factor=12)
    _t2 = timeit.default_timer()
    print(f"Detected {len(keypoints)} keypoints in {_t1 - _t0}")
    print(f"Extracted {len(keypoints)} patches in {_t2 - _t1}")


if __name__ == "__main__":
    img = cv2.cvtColor(cv2.imread("ceres_test.png"), cv2.COLOR_BGR2GRAY)[..., np.newaxis]

    # Test SIFT.
    detector = cv2.ORB_create(nfeatures=5000)
    detector_runtime_test(img, detector)

    # Test ORB.
