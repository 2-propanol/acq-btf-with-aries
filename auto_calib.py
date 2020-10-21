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
    stage = Aries()

    # カメラ初期設定
    cap = EasyPySpin.VideoCaptureEX(0)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE_US)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    # cap.set(cv2.CAP_PROP_GAMMA, 1.0)
    cap.average_num = CAM_AVERAGE

    # 対応点の対象をランダムに決定
    ## 1st half: 60 <= tilt < 90
    xyzs_1st_half = np.random.rand((TRY_XYZS + 1) // 2, 3)
    xyzs_1st_half[:, 0] = xyzs_1st_half[:, 0] * 120 - 60
    xyzs_1st_half[:, 1] = xyzs_1st_half[:, 1] * 30 + 60
    xyzs_1st_half[:, 2] = xyzs_1st_half[:, 2] * 45
    ## pan順に並び変える
    xyzs_1st_half = xyzs_1st_half[np.argsort(xyzs_1st_half[:, 0])]

    ## 2nd half: 30 <= tilt < 60
    xyzs_2nd_half = np.random.rand(TRY_XYZS // 2, 3)
    xyzs_2nd_half[:, 0] = xyzs_2nd_half[:, 0] * 120 - 60
    xyzs_2nd_half[:, 1] = xyzs_2nd_half[:, 1] * 30 + 30
    xyzs_2nd_half[:, 2] = xyzs_2nd_half[:, 2] * 45
    ## pan順に並び変える
    xyzs_2nd_half = xyzs_2nd_half[np.argsort(xyzs_2nd_half[:, 0])[::-1]]

    xyzs = np.concatenate((xyzs_1st_half, xyzs_2nd_half))

    # Ariesの精度は小数点以下3桁まで
    xyzs = xyzs.round(decimals=2)

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
