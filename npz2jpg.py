"""acq_btf.pyで得た.btf.npzを.jpgに変換する"""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BTF_FILENAME = "SweatNavyBlue-1D.btf.npz"
JPEG_OUTPUT_DIRECTORY = "SweatNavyBlue-1D"


def main() -> None:
    npz = np.load(BTF_FILENAME)
    btf = npz["images"]

    Path(JPEG_OUTPUT_DIRECTORY).mkdir(exist_ok=True)

    for i, src in tqdm(enumerate(btf), total=len(btf)):
        dst = np.clip(src*255,0,255)
        cv2.imwrite(f"{JPEG_OUTPUT_DIRECTORY}/{i}.jpg", dst)

if __name__ == "__main__":
    main()
