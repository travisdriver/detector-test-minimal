import timeit

import cv2
import numpy as np
import torch

from superpoint import SuperPoint


# img = cv2.cvtColor(cv2.imread("ceres_test.png"), cv2.COLOR_BGR2GRAY)[None, None, ...]
# img = torch.from_numpy(img.astype(np.float32) / 255.0)
img = torch.randn((1, 1, 1024, 1024)).cpu()

print(img.shape)

net = SuperPoint({}).cpu()
net.eval()
runtimes = []
for _ in range(10):
    _t0 = timeit.default_timer()
    out = net({"image": img})
    _t1 = timeit.default_timer()
    runtimes.append(_t1 - _t0)
    print(runtimes[-1] * 1000)

print(np.mean(runtimes) * 1000)
