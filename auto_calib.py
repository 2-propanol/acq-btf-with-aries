from pathlib import Path

import cv2
import EasyPySpin
import matplotlib.pyplot as plt
import numpy as np
from aries import Aries
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from calib_utils import calib_by_points, calibed_rmse, rot_matrix_from_pan_tilt_roll

TRY_XYZS = 20
FILENAME_TO_SAVE_CORRESPONDS = "camera_matrix.npy"

CAM_GAIN = 5
CAM_AVERAGE = 3
CAM_EXPOSURE_US = 50000

PAN_ROTATE_RANGE = 120
TILT_ROTATE_RANGE = 60
ROLL_ROTATE_RANGE = 45
USE_U_AXIS = False

# 以下内部用定数
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

_AR_ID_TO_WORLD_XYZ_40X40 = np.array(
    (
        (-0.85, -0.85, 0.0),
        (-0.50, -0.85, 0.0),
        (0.00, -0.85, 0.0),
        (0.35, -0.85, 0.0),
        (0.75, -0.85, 0.0),
        (-0.85, -0.45, 0.0),
        (-0.50, -0.35, 0.0),
        (-0.05, -0.50, 0.0),
        (0.30, -0.35, 0.0),
        (0.85, -0.40, 0.0),
        (-0.85, 0.10, 0.0),
        (-0.45, 0.05, 0.0),
        (-0.05, -0.05, 0.0),
        (0.30, 0.00, 0.0),
        (0.65, -0.05, 0.0),
        (-0.75, 0.50, 0.0),
        (-0.40, 0.40, 0.0),
        (-0.05, 0.30, 0.0),
        (0.40, 0.35, 0.0),
        (0.80, 0.30, 0.0),
        (-0.80, 0.85, 0.0),
        (-0.40, 0.75, 0.0),
        (-0.05, 0.85, 0.0),
        (0.35, 0.70, 0.0),
        (0.80, 0.80, 0.0),
    )
)


def test_ar_reader():
    cap = EasyPySpin.VideoCaptureEX(0)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE_US)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
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

        cv2.imshow("Press [q] to start calibration.", frame)
        if cv2.waitKey(100) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def auto_calib():
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
    cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE_US)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
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

            # ARマーカーの世界座標を計算
            world_rot = rot_matrix_from_pan_tilt_roll(*stage.position[0:3])
            world_xyz = _AR_ID_TO_WORLD_XYZ_40X40[ar_id[0]]
            world_xyz = world_rot @ world_xyz

            # カメラに写った中心座標を計算
            ar_center = ar_corner[0].mean(axis=0)

            # 対応点として記録
            corresponds.append(np.append(world_xyz, ar_center))

    cap.release()
    del stage

    # キャリブレーションを行い、評価を表示し、カメラ行列をnpyに保存
    corresponds = np.array(corresponds)
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

    np.save(FILENAME_TO_SAVE_CORRESPONDS, cam_mat)
    print(f"saved camera matrix to [{FILENAME_TO_SAVE_CORRESPONDS}]")


if __name__ == "__main__":
    test_ar_reader()
    if Path(FILENAME_TO_SAVE_CORRESPONDS).exists():
        print(f"file: [{FILENAME_TO_SAVE_CORRESPONDS}] is already exists.")
    else:
        auto_calib()
