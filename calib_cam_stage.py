"""カメラとステージ位置のキャリブレーション"""
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from nptyping import NDArray
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation


def rot_matrix_from_pan_tilt_roll(
    pan: float, tilt: float, roll: float
) -> NDArray[(3, 3), np.float]:
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
        ndarray: 回転行列。dtype=np.float, shape=(3, 3)
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


def obj_and_img_points_from_csv(filepath: str) -> NDArray[(Any, 5), np.float]:
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
) -> NDArray[(3, 4), np.float]:
    """正面画像で写った素材の大きさと中心点から、世界座標とカメラ座標の変換行列を求める。

    ・画角0度（テレセントリック）
    ・素材面とイメージセンサーが完全に並行
    ・歪みなし
    の理想的な状態であれば、対応点を求めることなしに変換行列が求まる。

    Args:
        pictured_size (float, float): 正面画像で写った素材の大きさ
        center_point (float, float): 正面画像で写った素材の中心点

    Returns:
        ndarray: カメラパラメータ行列。dtype=np.float, shape=(3, 4)
    """
    C = np.array(
        [
            [pictured_size[0] / 2, 0, 0, center_point[0]],
            [0, pictured_size[1] / 2, 0, center_point[1]],
            [0, 0, 1, 0],
        ]
    )
    return C


def calib_by_points(
    obj_and_img_points: NDArray[(Any, 5), np.float]
) -> NDArray[(Any, 4), np.float]:
    """世界座標とカメラ座標の対応点からカメラパラメータ行列を求める。

    Args:
        obj_and_img_points (ndarray): dtype=np.float, shape=(N, 5)

    Returns:
        ndarray: カメラパラメータ行列。dtype=np.float, shape=(3, 4)
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


def wrap_homogeneous_dot(
    matrix: NDArray[(3, 4), np.float],
    objpoints: Union[NDArray[(Any, 3), np.float], NDArray[3, np.float]],
) -> Union[NDArray[(Any, 2), np.float], NDArray[2, np.float]]:
    """3x1行列 = 3x4行列 @ 4x1行列のラッパー。3点をから2点を得る。

    Args:
        matrix (ndarray): 3x4カメラパラメータ行列
        objpoints (ndarray): 世界座標X, Y, Z。単数でも複数でも良い。

    Return:
        ndarray: カメラ座標x, y。dtype=np.float, shape=(N, 2) or (2,)
    """
    if objpoints.ndim == 1:
        if objpoints.shape[0] == 3:
            imgpoint = matrix @ np.append(objpoints, 1.0)
            imgpoint = imgpoint / imgpoint[2]
            return imgpoint[0:2]

    elif objpoints.ndim == 2:
        if objpoints.shape[1] == 3:
            objpoints = np.vstack((objpoints.T, np.ones(objpoints.shape[0])))
            imgpoints = matrix @ objpoints
            imgpoints = imgpoints / imgpoints[2, :]
            return imgpoints.T[:, 0:2]

    raise ValueError(
        "`matrix` shape must be (3,) or (N, 3). inputed shape is ", objpoints.shape
    )


def test_calib(
    camera_matrix: NDArray[(3, 4), np.float],
    obj_and_img_points: NDArray[(Any, 5), np.float],
) -> None:
    """キャリブレーションの精度を確認する"""
    N = len(obj_and_img_points)

    true_imgpoint = obj_and_img_points[:, 3:5]
    pred_imgpoint = wrap_homogeneous_dot(camera_matrix, obj_and_img_points[:, 0:3])
    diff = true_imgpoint - pred_imgpoint
    diff_SD = (np.sum(diff ** 2, axis=0) / N) ** 0.5
    print("n: ", N)
    print("SD: ", diff_SD)
