import timeit
import math

import cv2
import numpy as np

NUM_ITERATIONS = 20


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
        cos = 1 # math.cos(a * math.pi / 180.0)
        sin = 0 # math.sin(a * math.pi / 180.0)

        M = np.matrix(
            [
                [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
                [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y],
            ]
        )

        patch = cv2.warpAffine(img, M, (N, N), flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        patches.append(patch)

    return patches


def detector_runtime_test(img, detector, mag_factor=16):
    t_detect = []
    t_extract = []
    for i in range(NUM_ITERATIONS):
        _t0 = timeit.default_timer() * 1000
        keypoints = detector.detect(img, None)
        _t1 = timeit.default_timer() * 1000
        # print(len(keypoints))
        # if len(keypoints) > 0:
        patches = extract_patches(img, keypoints, 32, mag_factor=mag_factor)
        _t2 = timeit.default_timer() * 1000
        if i < 10:
            continue
        t_detect.append(_t1 - _t0)
        t_extract.append(_t2 - _t1)
        print(_t1 - _t0)
        print(_t2 - _t1)
        print(np.array(patches).shape)
        print(keypoints[0].size)
    print(f"Detected {len(keypoints)} keypoints in {np.mean(t_detect)}")
    print(f"Extracted {len(keypoints)} patches in {np.mean(t_extract)}")


if __name__ == "__main__":
    img = cv2.cvtColor(cv2.imread("ceres_test.png"), cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    print(img.shape)

    # # Test SIFT.
    # print("\nTESTING SIFT")
    # for nfeatures in [500, 1000, 5000]:
    #     detector = cv2.SIFT_create(nfeatures=nfeatures)
    #     detector_runtime_test(img, detector, mag_factor=16)
    #     del detector

    # Test ORB.
    print("\nTESTING ORB")
    for nfeatures in [500, 1000, 5000]:
        detector = cv2.ORB_create(nfeatures=nfeatures)
        detector_runtime_test(img, detector, mag_factor=1)
        del detector

