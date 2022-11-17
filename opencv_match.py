import timeit
import math

import cv2
import numpy as np


def matching_runtime_test(num_features=500, dim_desc=128, norm_type=cv2.NORM_L2, num_trials=50, burn_in=10):

    # Create dummy descriptors.
    desc1 = np.random.rand(num_features, dim_desc).astype(np.float32)
    desc2 = np.random.rand(num_features, dim_desc).astype(np.float32)
    # print(desc1.shape)
    # print(desc1.shape)
    # print(desc1[0])
    if norm_type == cv2.NORM_HAMMING:
        desc1 = np.packbits(np.round(desc1).astype(int), axis=1)
        desc2 = np.packbits(np.round(desc2).astype(int), axis=1)

    # Create matcher.
    bf = cv2.BFMatcher(norm_type, crossCheck=True)

    runtimes = []
    for i in range(num_trials):
        _t0 = timeit.default_timer()
        _ = bf.match(desc1, desc2)
        _t1 = timeit.default_timer()
        if i < burn_in:
            continue
        runtimes.append(_t1 - _t0)
    print(f"Matched {num_features} features in {np.mean(runtimes) * 1000} ms")


if __name__ == "__main__":

    # Test full precision 128-dimensional descriptors, i.e., SIFT & DidymosNet
    num_features = 500
    dim_desc = 128
    norm_type = cv2.NORM_L2
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test full precision 256-dimensional descriptors, i.e., SuperPoint
    dim_desc = 256
    norm_type = cv2.NORM_L2
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test binary 256-dimensional descriptors, i.e., ORB & DidymosNet
    dim_desc = 256
    norm_type = cv2.NORM_HAMMING
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test full precision 128-dimensional descriptors, i.e., SIFT & DidymosNet
    num_features = 1000
    dim_desc = 128
    norm_type = cv2.NORM_L2
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test full precision 256-dimensional descriptors, i.e., SuperPoint
    dim_desc = 256
    norm_type = cv2.NORM_L2
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test binary 256-dimensional descriptors, i.e., ORB & DidymosNet
    dim_desc = 256
    norm_type = cv2.NORM_HAMMING
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test full precision 128-dimensional descriptors, i.e., SIFT & DidymosNet
    num_features = 5000
    dim_desc = 128
    norm_type = cv2.NORM_L2
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test full precision 256-dimensional descriptors, i.e., SuperPoint
    dim_desc = 256
    norm_type = cv2.NORM_L2
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")

    # Test binary 256-dimensional descriptors, i.e., ORB & DidymosNet
    dim_desc = 256
    norm_type = cv2.NORM_HAMMING
    print(f"Testing d={dim_desc}, n={num_features}, norm_type={norm_type}")
    matching_runtime_test(num_features, dim_desc, norm_type)
    print("\n")
