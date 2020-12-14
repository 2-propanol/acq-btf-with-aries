"""acq_btf.pyで得た.btf.npzを.jpgに変換する"""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BTF_FILENAME = "carbon_1D179.btf.npz"
JPEG_OUTPUT_DIRECTORY = "carbon_1D179"

EXTRACT_WITH_TLPLTVPV = False
EXTRACT_WITH_SEQUENTIAL = True


def main() -> None:
    npz = np.load(BTF_FILENAME)
    btf = npz["images"]
    angles = npz["angles"]

    Path(JPEG_OUTPUT_DIRECTORY).mkdir(exist_ok=True)


    for i, (src, angle) in tqdm(enumerate(zip(btf, angles)), total=len(btf)):
        dst = np.clip(src * 255, 0, 255)
        if EXTRACT_WITH_SEQUENTIAL:
            cv2.imwrite(f"{JPEG_OUTPUT_DIRECTORY}/{i}.png", dst)
        if EXTRACT_WITH_TLPLTVPV:
            cv2.imwrite(f"{JPEG_OUTPUT_DIRECTORY}/tl{angle[0]:.0f}_pl{angle[1]:.0f}_tv{angle[2]:.0f}_pv{angle[3]:.0f}.png", dst)


if __name__ == "__main__":
    main()
