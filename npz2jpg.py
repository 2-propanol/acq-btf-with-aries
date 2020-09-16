import numpy as np
import cv2

src = np.load("out.npz")['arr_0']
print(src.shape, src.dtype)

for i, out in enumerate(src):
    cv2.imwrite(f"out_{i}.png", out)