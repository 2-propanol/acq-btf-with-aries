from pathlib import Path

import cv2
import EasyPySpin
import numpy as np
from aries import Aries
from tqdm import tqdm

from calib_utils import rot_matrix_from_pan_tilt_roll

TRY_XYZS = 15
FILENAME_TO_SAVE_CORRESPONDS = "corresponds.npy"

CAM_GAIN = 0
CAM_AVERAGE = 3
CAM_EXPOSURE_US = 3500


aruco = cv2.aruco
ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

_AR_ID_TO_WORLD_XYZ = np.array(
        (
            (-0.7, -0.7, 0.0),
            (0.0, -0.7, 0.0),
            (0.7, -0.7, 0.0),
            (-0.7, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 0.0),
            (-0.7, 0.7, 0.0),
            (0.0, 0.7, 0.0),
            (0.7, 0.7, 0.0),
        )
    )


def test_ar_reader():
    cap = EasyPySpin.VideoCaptureEX(0)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE_US)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    # cap.set(cv2.CAP_PROP_GAMMA, 1.0)
    cap.average_num = CAM_AVERAGE

    while True:
        # 撮影
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

        # 手動の二値化試したけどあんまり良くならなかった
        corners, ids, rejected_corners = aruco.detectMarkers(frame, ar_dict)
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        aruco.drawDetectedMarkers(frame, rejected_corners, borderColor=(0, 0, 255))

        if len(corners) > 0:
            # 各ARマーカーについて
            for ar_corner, ar_id in zip(corners, ids):
                # カメラに写った中心座標を計算
                ar_center = ar_corner[0].mean(axis=0)

                frame = cv2.circle(
                    frame, tuple(ar_center.astype(np.int)), 7, (0, 0, 255), -1
                )

        cv2.imshow("Press [q] to start calibration.", frame)
        if cv2.waitKey(100) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def auto_calib():
    stage = Aries()

    # カメラ初期設定
    cap = EasyPySpin.VideoCaptureEX(0)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE_US)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    # cap.set(cv2.CAP_PROP_GAMMA, 1.0)
    cap.average_num = CAM_AVERAGE

    xyzs = np.random.rand(TRY_XYZS, 3)
    xyzs[:, 0] = xyzs[:, 0] * 120 - 60
    xyzs[:, 1] = xyzs[:, 1] * 60 + 30
    xyzs[:, 2] = xyzs[:, 2] * 45

    # Ariesの精度は小数点以下3桁まで
    xyzs = xyzs.round(decimals=2)

    # pan順に並び変える
    xyzs = xyzs[np.argsort(xyzs[:, 0])]

    corresponds = []
    schedule = tqdm(xyzs)
    for xyz in schedule:
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

        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

        # ARマーカー検出
        # （手動の二値化を試したが、あまり良くならなかった）
        corners, ids, _ = aruco.detectMarkers(frame, ar_dict)

        if len(corners) > 0:
            # 各ARマーカーについて
            for ar_corner, ar_id in zip(corners, ids):
                # ARマーカーの中心点は価値が低い(常に画像中心にある)ので廃棄
                if ar_id[0] == 4:
                    continue

                # ARマーカーの世界座標を計算
                world_rot = rot_matrix_from_pan_tilt_roll(*stage.position[0:3])
                world_xyz = _AR_ID_TO_WORLD_XYZ[ar_id[0]]
                world_xyz = world_rot @ world_xyz

                # カメラに写った中心座標を計算
                ar_center = ar_corner[0].mean(axis=0)

                # 対応点として記録
                corresponds.append(np.append(world_xyz, ar_center))

    # 取得できた対応点の数を表示し、npyに保存
    print("Valid points:", len(corresponds))
    np.save(FILENAME_TO_SAVE_CORRESPONDS, np.array(corresponds))

    cap.release()


if __name__ == "__main__":
    test_ar_reader()
    if Path(FILENAME_TO_SAVE_CORRESPONDS).exists():
        print(f"file: [{FILENAME_TO_SAVE_CORRESPONDS}] is already exists.")
    else:
        auto_calib()
