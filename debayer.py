import cv2

FILENAME = "acq.png"

src = cv2.imread(FILENAME, 0)
dst = cv2.cvtColor(src, cv2.COLOR_BAYER_BG2BGR)

cv2.imwrite("out.png", dst)
