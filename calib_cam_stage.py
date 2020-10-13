"""カメラとステージ位置のキャリブレーション"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation


def rot_matrix_from_pan_tilt_roll(pan: float, tilt: float, roll: float) -> np.ndarray:
    """ステージの回転による移動先の世界座標を得るための行列を求める。

    世界座標の原点は素材の中心、カメラから素材の向きをz軸の正の方向とした右手座標系。

    Example:
        >>> rot_matrix_from_pan_tilt_roll(0,90,0)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

        >>> rot_matrix_from_pan_tilt_roll(45,45,45)
        array([[ 0.14644661, -0.85355339,  0.5       ],
               [ 0.5       ,  0.5       ,  0.70710678],
               [-0.85355339,  0.14644661,  0.5       ]])

        >>> rot_matrix_from_pan_tilt_roll(45,90,0) @ np.array([1,1,0])
        array([ 0.70710678,  1.        , -0.70710678])

    Args:
        pan (float): [-90, 90]
        tilt (float): [0, 90]
        roll (float): [0, 360)

    Returns:
        ndarray: dtype=np.float, shape=(3, 3)
    """
    # degree を radian に変換
    pan_radian = np.deg2rad(pan)
    tilt_radian = np.deg2rad(tilt - 90)
    roll_radian = np.deg2rad(roll)

    # ステージ回転軸
    pan_axis = np.array([0, 1, 0])
    tilt_axis = np.array([1, 0, 0])
    roll_axis = np.array([0, 0, 1])

    # pan軸回転
    rot_pan = Rotation.from_rotvec(pan_axis * pan_radian)
    tilt_axis = rot_pan.apply(tilt_axis)
    roll_axis = rot_pan.apply(roll_axis)

    # tilt軸回転
    rot_tilt = Rotation.from_rotvec(tilt_axis * tilt_radian)
    roll_axis = rot_tilt.apply(roll_axis)

    # roll軸回転
    rot_roll = Rotation.from_rotvec(roll_axis * roll_radian)

    rot = rot_roll.as_matrix() @ rot_tilt.as_matrix() @ rot_pan.as_matrix()
    return rot


def obj_and_img_points_from_csv(filepath: str) -> np.ndarray:
    """キャリブレーション用csvから、世界座標とカメラ座標の組を返す。

    Args:
        filepath (str): pan, tilt, roll, corner, x, yの列を持つcsvファイルのパス。行数N。

    Returns:
        ndarray: dtype=np.float, shape=(N, 5)。
                 世界座標は[:, 0:3]、カメラ座標は[:, 3:5]。
    """

    df = pd.read_csv(filepath)
    obj_and_img_points = np.empty((len(df), 5))
    for i, row in enumerate(df.itertuples()):
        if row.corner == "lt":
            objpoint = np.array([-1, -1, 0])
        elif row.corner == "rt":
            objpoint = np.array([1, -1, 0])
        elif row.corner == "lb":
            objpoint = np.array([-1, 1, 0])
        elif row.corner == "rb":
            objpoint = np.array([1, 1, 0])
        else:
            raise NameError("corner name other than lt, rt, lb, rb")

        rot = rot_matrix_from_pan_tilt_roll(row.pan, row.tilt, row.roll)
        objpoint = rot @ objpoint
        imgpoint = row.x, row.y
        obj_and_img_points[i] = (*objpoint, *imgpoint)
    return obj_and_img_points


def no_calib(
    pictured_size: Tuple[float, float], center_point: Tuple[float, float]
) -> np.ndarray:
    """正面画像で写った素材の大きさと中心点から、世界座標とカメラ座標の変換行列を求める。

    ・画角0度（テレセントリック）
    ・素材面とイメージセンサーが完全に並行
    ・歪みなし
    の理想的な状態であれば、対応点を求めることなしに変換行列が求まる。

    Args:
        pictured_size (float, float): 正面画像で写った素材の大きさ
        center_point (float, float): 正面画像で写った素材の中心点

    Returns:
        ndarray: dtype=np.float, shape=(3, 4)
    """
    C = np.array(
        [
            [pictured_size[0] / 2, 0, 0, center_point[0]],
            [0, pictured_size[1] / 2, 0, center_point[1]],
            [0, 0, 1, 0],
        ]
    )
    return C


def calib_by_points(obj_and_img_points: np.ndarray) -> np.ndarray:
    """世界座標とカメラ座標の対応点からカメラパラメータ行列を求める。

    Args:
        obj_and_img_points (ndarray): dtype=np.float, shape=(N, 5)

    Returns:
        ndarray: dtype=np.float, shape=(3, 4)
    """
    N = len(obj_and_img_points)
    A = np.zeros((N * 2, 11))
    b = np.zeros((N * 2))
    for i in range(N):
        X, Y, Z = obj_and_img_points[i][0:3]
        u, v = obj_and_img_points[i][3:5]

        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -X * u, -Y * u, -Z * u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -X * v, -Y * v, -Z * v]
        b[2 * i] = u
        b[2 * i + 1] = v

    C, _, _, _ = lstsq(A, b)
    C = np.append(C, 1.0).reshape((3, 4))
    return C


def test_calib(camera_matrix: np.ndarray, obj_and_img_points: np.ndarray) -> None:
    """キャリブレーションの精度を確認する"""
    diff_sum = np.array((0, 0))
    for obj_img in obj_and_img_points:
        predicted_imgpoint = camera_matrix @ np.array(
            (obj_img[0], obj_img[1], obj_img[2], 1.0)
        )
        predicted_imgpoint = predicted_imgpoint / predicted_imgpoint[2]
        diff = np.array(
            ((obj_img[3] - predicted_imgpoint[0]), (obj_img[4] - predicted_imgpoint[1]))
        )
        # print("true imgpoint:", np.array((obj_img[3],obj_img[4])).round(1))
        # print("         diff:", diff.round(1))
        diff_sum = diff_sum + diff ** 2
    print("n: ", len(obj_and_img_points))
    print("SD: ", (diff_sum / len(obj_and_img_points)) ** 0.5)
