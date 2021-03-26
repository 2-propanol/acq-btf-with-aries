from time import sleep
from typing import Any
from pathlib import Path

import cv2
import EasyPySpin
import matplotlib.pyplot as plt
import numpy as np
from aries import Aries
from mpl_toolkits.mplot3d import Axes3D
from nptyping import NDArray
from tqdm import tqdm

from calib_utils import (
    _AR_ID_TO_WORLD_XYZ_40X40,
    optimize_id_to_xyz,
    raw_xyz_to_cam_mat,
)

TRY_XYZS = 50
FILENAME_TO_SAVE_CORRESPONDS = "corresponds_20210326-7.npy"
FILENAME_TO_SAVE_CAMERA_MATRIX = "camera_matrix_20210326-7.npy"

# `None`でカメラから、`str`でファイルから対応点を取得
FILENAME_TO_LOAD_CORRESPONDS = None
# FILENAME_TO_LOAD_CORRESPONDS = "corresponds_20210326-7.npy"

CAM_GAIN = 5
CAM_AVERAGE = 3
CAM_EXPOSURE_US = 25000

PAN_ROTATE_RANGE = 140
TILT_ROTATE_RANGE = 70
ROLL_ROTATE_RANGE = 45
USE_U_AXIS = False

# 以下内部用定数
aruco = cv2.aruco
ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


def test_ar_reader() -> bool:
    """ARマーカーが認識できているか確認し、撮影を始めるか決める

    Returns:
        bool: qキーでFalse、sキーでTrueが返る
    """
    cap = EasyPySpin.VideoCaptureEX(0)
    # cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE_US)
    # cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    # cap.set(cv2.CAP_PROP_GAMMA, 1.0)
    cap.average_num = CAM_AVERAGE

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

        # ARマーカー検出
        corners, ids, rejected_corners = aruco.detectMarkers(frame, ar_dict)
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        aruco.drawDetectedMarkers(frame, rejected_corners, borderColor=(0, 0, 255))

        if len(corners) > 0:
            # 各ARマーカーについて
            for ar_corner in corners:
                # カメラに写った中心座標を計算
                ar_center = ar_corner[0].mean(axis=0)
                frame = cv2.circle(
                    frame, tuple(ar_center.astype(np.int)), 7, (0, 0, 255), -1
                )

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("[q] to quit(cancel), [s] to start calibration.", frame)
        key = cv2.waitKey(50)
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == ord("s"):
            cap.release()
            cv2.destroyAllWindows()
            return True


def get_corresponds() -> NDArray[(Any, 5), np.float]:
    """4軸ステージを動かして、ARマーカーの画像座標とステージ位置の対応点を得る

    Returns:
        ndarray: n*5のndarray。各行は、ARマーカーのID(0:0)、
                 画像上のARマーカーの中心座標(1:3)、ステージ位置(3:6)が入る。
    """
    if not (0 <= PAN_ROTATE_RANGE <= 180):
        print("invalid `PAN_ROTATE_RANGE`")
    if not (0 <= TILT_ROTATE_RANGE <= 90):
        print("invalid `PAN_ROTATE_RANGE`")
    if not (0 < PAN_ROTATE_RANGE <= 360):
        print("invalid `PAN_ROTATE_RANGE`")

    stage = Aries()

    # U軸固定モード(`USE_U_AXIS=False`)の場合、
    # 事前にカメラから素材が見えるようにしておく。
    if not USE_U_AXIS:
        pos = list(stage.position)
        if -10 < pos[3] < 10:
            u = -10
            pos[3] = u
            stage.position = pos
        else:
            u = pos[3]

    # カメラ初期設定
    cap = EasyPySpin.VideoCaptureEX(0)
    # cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE_US)
    # cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    # cap.set(cv2.CAP_PROP_GAMMA, 1.0)
    cap.average_num = CAM_AVERAGE

    # 対応点の対象をランダムに決定
    ## 1st half: tilt上側, 2nd half: tilt下側
    xyzs_1st_half = np.random.rand((TRY_XYZS + 1) // 2, 3)
    xyzs_1st_half[:, 0] = xyzs_1st_half[:, 0] * PAN_ROTATE_RANGE - PAN_ROTATE_RANGE / 2
    xyzs_1st_half[:, 1] = xyzs_1st_half[:, 1] * TILT_ROTATE_RANGE / 2 + (
        90 - TILT_ROTATE_RANGE / 2
    )
    xyzs_1st_half[:, 2] = xyzs_1st_half[:, 2] * 45
    ## pan順に並び変える
    xyzs_1st_half = xyzs_1st_half[np.argsort(xyzs_1st_half[:, 0])]

    xyzs_2nd_half = np.random.rand(TRY_XYZS // 2, 3)
    xyzs_2nd_half[:, 0] = xyzs_2nd_half[:, 0] * PAN_ROTATE_RANGE - PAN_ROTATE_RANGE / 2
    xyzs_2nd_half[:, 1] = xyzs_2nd_half[:, 1] * TILT_ROTATE_RANGE / 2 + (
        90 - TILT_ROTATE_RANGE
    )
    xyzs_2nd_half[:, 2] = xyzs_2nd_half[:, 2] * 45
    ## pan順に並び変える
    xyzs_2nd_half = xyzs_2nd_half[np.argsort(xyzs_2nd_half[:, 0])[::-1]]

    xyzs = np.concatenate((xyzs_1st_half, xyzs_2nd_half))

    # Ariesの精度は小数点以下3桁まで
    xyzs = xyzs.round(decimals=2)

    corresponds = []
    schedule = tqdm(xyzs)
    for xyz in schedule:
        if USE_U_AXIS:
            # xyzから、写りが良くなるであろうuを決める
            if xyz[0] >= 0:
                u = np.clip(xyz[0], 10, 80)
            else:
                u = np.clip(xyz[0], -80, -10)

        schedule.set_description(
            f"[Valid points: {len(corresponds):3d}, "
            + f"X: {xyz[0]:3.0f}, Y: {xyz[1]:2.0f}, "
            + f"Z: {xyz[2]:3.0f}, U: {u:3.0f}]"
        )

        stage.position = (*xyz, u)
        stage.sleep_until_stop()
        sleep(1)

        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

        # ARマーカー検出
        # corners: list[NDArray[(同一IDのARマーカーの個数, 4, 2), np.float32]]
        # ids: Optional[NDArray[(検出したARマーカーIDの個数, 1), np.int32]]
        corners, ids, _ = aruco.detectMarkers(frame, ar_dict)

        # 何も検出されなければ次のポジションへ
        # `ids = None`による"TypeError: 'NoneType' object is not iterable"の回避
        if len(corners) == 0:
            continue

        # 各ARマーカーについて
        for ar_corner, ar_id in zip(corners, ids):
            # 予期していないIDが検出された場合無視する
            if ar_id[0] >= len(_AR_ID_TO_WORLD_XYZ_40X40):
                continue

            # カメラに写った中心座標を計算
            ar_center = ar_corner[0].mean(axis=0)

            # 対応点として記録
            corresponds.append(np.concatenate([ar_id, ar_center, xyz]))

    cap.release()
    del stage

    corresponds = np.array(corresponds)
    print("Valid points:", len(corresponds))
    return corresponds


if __name__ == "__main__":
    # カメラ行列ファイルがあれば何もしない。
    if Path(FILENAME_TO_SAVE_CAMERA_MATRIX).exists():
        print(f"file: [{FILENAME_TO_SAVE_CAMERA_MATRIX}] is already exists.")
        exit()

    # 対応点ファイルがあればそれを使う。なければカメラで撮影して取得する。
    if FILENAME_TO_LOAD_CORRESPONDS:
        corresponds = np.load(FILENAME_TO_LOAD_CORRESPONDS)
        print(f"loaded raw corresponding points from [{FILENAME_TO_LOAD_CORRESPONDS}]")
    else:
        if not test_ar_reader():
            exit()
        corresponds = get_corresponds()
        np.save(FILENAME_TO_SAVE_CORRESPONDS, corresponds)
        print(f"saved raw corresponding points to [{FILENAME_TO_SAVE_CORRESPONDS}]")

    # 対応点からカメラ行列を計算する。
    ar_id = corresponds[:, 0].astype(np.int)
    ar_center = corresponds[:, 1:3]
    stage_pos = corresponds[:, 3:6]
    id_to_xyz = optimize_id_to_xyz(ar_id, ar_center, stage_pos)

    cam_mat, rmse, diff = raw_xyz_to_cam_mat(
        ar_id, ar_center, stage_pos, id_to_xyz
    )

    print("camera matrix:\n", cam_mat)
    print("RMSE:", rmse)

    np.save(FILENAME_TO_SAVE_CAMERA_MATRIX, cam_mat)
    print(f"saved camera matrix to [{FILENAME_TO_SAVE_CAMERA_MATRIX}]")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(diff[:, 0], diff[:, 1])
    plt.show()
