from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from calib_utils import calib_by_points, calibed_rmse, obj_and_img_points_from_csv

CSV_FILENAME_TO_LOAD_CORRESPONDS = "corresponds.csv"
NPY_FILENAME_TO_SAVE_CAMERA_MATRIX = "camera_matrix.npy"

CAM_GAIN = 5
CAM_AVERAGE = 3
CAM_EXPOSURE_US = 50000


def mannual_calib():
    # キャリブレーションを行い、評価を表示し、カメラ行列をnpyに保存
    corresponds = obj_and_img_points_from_csv(CSV_FILENAME_TO_LOAD_CORRESPONDS)
    print("Valid points:", len(corresponds))

    cam_mat = calib_by_points(corresponds)
    print("camera matrix:\n", cam_mat)

    rmse, diff = calibed_rmse(cam_mat, corresponds)
    print("RMSE:", rmse)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(diff[:, 0], diff[:, 1])
    fig.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot(
        corresponds[:, 0],
        corresponds[:, 1],
        corresponds[:, 2],
        marker="o",
        linestyle="None",
    )
    plt.show()

    np.save(NPY_FILENAME_TO_SAVE_CAMERA_MATRIX, cam_mat)
    print(f"saved camera matrix to [{NPY_FILENAME_TO_SAVE_CAMERA_MATRIX}]")


if __name__ == "__main__":
    if Path(NPY_FILENAME_TO_SAVE_CAMERA_MATRIX).exists():
        print(f"file: [{NPY_FILENAME_TO_SAVE_CAMERA_MATRIX}] is already exists.")
    else:
        mannual_calib()
