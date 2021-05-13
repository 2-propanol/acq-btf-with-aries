"""カメラとステージ位置のキャリブレーション"""
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from nptyping import NDArray
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation

# ARマーカーIDからARマーカーの座標を得るための定数
## 20x20画素ARマーカープレート
_AR_ID_TO_WORLD_XYZ_20X20 = np.array(
    (
        # fmt: off
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

## 40x40画素ARマーカープレート
_AR_ID_TO_WORLD_XYZ_40X40 = np.array(
    (
        # fmt: off
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


def rot_matrix_from_pan_tilt_roll(
    pan: float, tilt: float, roll: float
) -> NDArray[(3, 3), np.float64]:
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
        ndarray: 回転行列。dtype=np.float64, shape=(3, 3)
    """
    # degree を radian に変換
    pan_radian = np.deg2rad(pan)
    tilt_radian = np.deg2rad(tilt - 90)
    roll_radian = np.deg2rad(roll)

    # ステージ回転軸
    pan_axis = np.array([0, 1, 0])
    tilt_axis = np.array([1, 0, 0])
    roll_axis = np.array([0, 0, 1])

    # 各軸の回転行列
    rot_pan = Rotation.from_rotvec(pan_axis * pan_radian)
    rot_tilt = Rotation.from_rotvec(tilt_axis * tilt_radian)
    rot_roll = Rotation.from_rotvec(roll_axis * roll_radian)

    # roll, tilt, panの順に回転
    rot = rot_pan.as_matrix() @ rot_tilt.as_matrix() @ rot_roll.as_matrix()
    return rot


def obj_and_img_points_from_csv(filepath: str) -> NDArray[(Any, 5), np.float64]:
    """キャリブレーション用csvから、世界座標とカメラ座標の組を返す。

    Args:
        filepath (str): pan, tilt, roll, corner, x, yの列を持つcsvファイルのパス。行数N。

    Returns:
        ndarray: dtype=np.float64, shape=(N, 5)。
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
) -> NDArray[(3, 4), np.float64]:
    """正面画像で写った素材の大きさと中心点から、世界座標とカメラ座標の変換行列を求める。

    ・画角0度（テレセントリック）
    ・素材面とイメージセンサーが完全に並行
    ・歪みなし
    の理想的な状態であれば、対応点を求めることなしに変換行列が求まる。

    Args:
        pictured_size (float, float): 正面画像で写った素材の大きさ
        center_point (float, float): 正面画像で写った素材の中心点

    Returns:
        ndarray: カメラパラメータ行列。dtype=np.float64, shape=(3, 4)
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
    obj_and_img_points: NDArray[(Any, 5), np.float64]
) -> NDArray[(Any, 4), np.float64]:
    """世界座標とカメラ座標の対応点からカメラパラメータ行列を求める。

    Args:
        obj_and_img_points (ndarray): dtype=np.float64, shape=(N, 5)

    Returns:
        ndarray: カメラパラメータ行列。dtype=np.float64, shape=(3, 4)
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
    matrix: NDArray[(3, 4), np.float64],
    objpoints: Union[NDArray[(Any, 3), np.float64], NDArray[3, np.float64]],
) -> Union[NDArray[(Any, 2), np.float64], NDArray[2, np.float64]]:
    """3x1行列 = 3x4行列 @ 4x1行列のラッパー。3点をから2点を得る。

    Args:
        matrix (ndarray): 3x4カメラパラメータ行列
        objpoints (ndarray): 世界座標X, Y, Z。単数でも複数でも良い。

    Returns:
        ndarray: カメラ座標x, y。dtype=np.float64, shape=(N, 2) or (2,)
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
        "`objpoints` shape must be (3,) or (N, 3). inputed shape is ", objpoints.shape
    )


def calibed_rmse(
    camera_matrix: NDArray[(3, 4), np.float64],
    obj_and_img_points: NDArray[(Any, 5), np.float64],
) -> Tuple[NDArray[(2,), np.float64], NDArray[(Any, 2), np.float64]]:
    """キャリブレーションの精度を確認する（RMSEとdiffを返す）

    Args:
        camera_matrix (ndarray): 3x4カメラパラメータ行列
        obj_and_img_points (ndarray): dtype=np.float64, shape=(N, 5)

    Returns:
        tuple: 「xとyについてのRMSE」と「各対応点に対する再投影誤差」のタプル
    """
    N = len(obj_and_img_points)

    true_imgpoint = obj_and_img_points[:, 3:5]
    pred_imgpoint = wrap_homogeneous_dot(camera_matrix, obj_and_img_points[:, 0:3])
    diff = true_imgpoint - pred_imgpoint
    rmse = (np.sum(diff ** 2, axis=0) / N) ** 0.5
    return rmse, diff


def raw_xyz_to_cam_mat(
    ar_id: NDArray[(Any,), np.int32],
    ar_center: NDArray[(Any, 2), np.float64],
    stage_pos: NDArray[(Any, 3), np.float64],
    id_to_xyz: NDArray[(Any, 3), np.float64],
) -> Tuple[
    NDArray[(3, 4), np.float64],
    NDArray[(2,), np.float64],
    NDArray[(Any, 2), np.float64],
]:
    """ARマーカーとステージ位置から3x4カメラパラメータ行列を得る

    Args:
        ar_id (ndarray): ARマーカーIDの配列
        ar_center (ndarray): 画像上のARマーカーの中心座標(x, y)
        stage_pos (ndarray): ステージ位置(pan, tilt, roll)
        id_to_xyz (ndarray): ARマーカーIDから世界座標を得る配列

    Returns:
        tuple: 「3x4カメラパラメータ行列」と
               「xとyについてのRMSE」と「各対応点に対する再投影誤差」のタプル
    """
    obj_and_img_points = []
    for aid, center, pos in zip(ar_id, ar_center, stage_pos):
        # ARマーカーの世界座標を計算
        world_rot = rot_matrix_from_pan_tilt_roll(*pos)
        world_xyz = id_to_xyz[aid]
        world_xyz = world_rot @ world_xyz
        # 対応点として記録
        obj_and_img_points.append(np.append(world_xyz, center))

    obj_and_img_points = np.array(obj_and_img_points)
    cam_mat = calib_by_points(obj_and_img_points)
    rmse, diff = calibed_rmse(cam_mat, obj_and_img_points)
    return cam_mat, rmse, diff


def optimize_id_to_xyz(
    ar_id: NDArray[(Any,), np.int32],
    ar_center: NDArray[(Any, 2), np.float64],
    stage_pos: NDArray[(Any, 3), np.float64],
):
    """対応点情報から最適なARマーカー世界座標を求める

    Args:
        ar_id (ndarray): ARマーカーIDの配列
        ar_center (ndarray): 画像上のARマーカーの中心座標(x, y)
        stage_pos (ndarray): ステージ位置(pan, tilt, roll)

    Returns:
        ndarray: ARマーカーIDから世界座標を得る配列
    """

    def golden_search(
        xyz_index: int,
        error: float,
        low: float,
        high: float,
        id_to_xyz_base: np.ndarray,
    ) -> Tuple[float, NDArray[(2,), np.float64]]:
        """最適なARマーカー世界座標オフセットを黄金分割探索により求める

        Args:
            xyz_index (int): x, y, zについて探索する場合は0, 1, 2を入れる
            error (float): 最大誤差
            low (float): 探索下限
            high (float): 探索上限
            id_to_xyz_base (np.ndarray): ARマーカーIDから世界座標を得る配列

        Returns:
            tuple: 「最適ARマーカー世界座標オフセット」と「xとyについてのRMSE」のタプル
        """
        PHI = (1.0 + 5 ** (1 / 2)) / 2

        def offset_to_rmse(offset: float, index: int) -> NDArray[(2,), np.float64]:
            """`index`を`offset`したときのRMSEを求める

            Args:
                offset (float): ARマーカー世界座標オフセット
                index (int): x, y, zについて探索する場合は0, 1, 2を入れる

            Returns:
                ndarray: xとyについてのRMSE
            """
            id_to_xyz_opt = np.copy(id_to_xyz_base)
            id_to_xyz_opt[:, index] = id_to_xyz_opt[:, index] + offset
            _, rmse, _ = raw_xyz_to_cam_mat(ar_id, ar_center, stage_pos, id_to_xyz_opt)
            return rmse

        x_low = low
        x_high = high
        x1 = (x_low * PHI + x_high) / (1.0 + PHI)
        x2 = (x_low + x_high * PHI) / (1.0 + PHI)
        rmse_1 = offset_to_rmse(x1, xyz_index)
        rmse_2 = offset_to_rmse(x2, xyz_index)
        while x_low + error < x_high:
            if np.sum(rmse_1) < np.sum(rmse_2):
                x_high = x2
                x2 = x1
                x1 = (x_low * PHI + x_high) / (1.0 + PHI)
                rmse_2 = rmse_1
                rmse_1 = offset_to_rmse(x1, xyz_index)
            else:
                x_low = x1
                x1 = x2
                x2 = (x_low + x_high * PHI) / (1.0 + PHI)
                rmse_1 = rmse_2
                rmse_2 = offset_to_rmse(x2, xyz_index)
        return x1, rmse_1

    id_to_xyz_optimized = np.copy(_AR_ID_TO_WORLD_XYZ_40X40)
    applied_offset = [0, 0, 0]

    for err, halfwidth in zip((1e-4, 1e-6, 1e-8), (0.04, 0.005, 0.001)):
        best_z_offset, best_rmse = golden_search(
            2, err, -halfwidth, halfwidth, id_to_xyz_optimized
        )
        print(f"best Z offset is {best_z_offset:+.6f}, RMSE:{np.sum(best_rmse):.6f}")
        id_to_xyz_optimized[:, 2] = id_to_xyz_optimized[:, 2] + best_z_offset
        applied_offset[2] += best_z_offset

        best_x_offset, best_rmse = golden_search(
            0, err, -halfwidth, halfwidth, id_to_xyz_optimized
        )
        print(f"best X offset is {best_x_offset:+.6f}, RMSE:{np.sum(best_rmse):.6f}")
        id_to_xyz_optimized[:, 0] = id_to_xyz_optimized[:, 0] + best_x_offset
        applied_offset[0] += best_x_offset

        best_y_offset, best_rmse = golden_search(
            1, err, -halfwidth, halfwidth, id_to_xyz_optimized
        )
        print(f"best Y offset is {best_y_offset:+.6f}, RMSE:{np.sum(best_rmse):.6f}")
        id_to_xyz_optimized[:, 1] = id_to_xyz_optimized[:, 1] + best_y_offset
        applied_offset[1] += best_y_offset

    print(
        f"applied offset is ("
        + f"{applied_offset[0]:+.6f}, "
        + f"{applied_offset[1]:+.6f}, "
        + f"{applied_offset[2]:+.6f})"
    )

    return id_to_xyz_optimized
