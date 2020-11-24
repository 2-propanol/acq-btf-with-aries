"""Aries4軸ステージでBTFを撮影する"""
from pathlib import Path
from typing import Any, Tuple

import colour_demosaicing as cd
import cv2
import EasyPySpin
import numpy as np
import pandas as pd
from aries import Aries
from nptyping import Float16, NDArray
from tqdm import trange

from acq_presets import preset_4d_169shots
from calib_utils import rot_matrix_from_pan_tilt_roll, wrap_homogeneous_dot
from transcoord import tlpltvpv_to_xyzu

NPZ_FILENAME_TO_SAVE = "ARMarker-4D.btf.npz"
NPY_FILENAME_FOR_CAMERA_MATRIX = "camera_matrix.npy"

# 露光時間（単位:us）
ACQ_T_MIN_US = 2500
ACQ_T_MAX_US = 25000
ACQ_T_REF_US = 10000

ACQ_GAIN = 0
ACQ_AVERAGE = 3

# クロップ後画像サイズ
IMG_SIZE = (512, 512)

_WORLD_LT_RT_LB_RB = ((-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0))


def schedule_acq(
    to_acq: NDArray[(Any, 4), Float16]
) -> Tuple[NDArray[(Any, 4), Float16], NDArray[(Any, 4), Float16]]:
    """ステージ移動時間がなるべく短くなるように撮影順序を決める。

    現在は"light", "pan", "tilt", "roll"の順にソートしているだけ

    Args:
        to_acq: 撮影したいtlpltvpvの組が入った`ndarray`, shape=(Any, 4)

    Returns:
        tlpltvpvの入った`ndarray`とxyzuの入った`ndarray`のタプル
    """
    to_acq_tlpltvpv = pd.DataFrame(
        to_acq,
        columns=["tl", "pl", "tv", "pv"],
    )
    to_acq_xyzu = pd.DataFrame(
        [tlpltvpv_to_xyzu(*tlpltvpv) for tlpltvpv in to_acq],
        columns=["pan", "tilt", "roll", "light"],
    )
    to_acq_xyzu = to_acq_xyzu.astype(np.float32).round(4)

    df_tlpltvpv_xyzu = pd.concat([to_acq_tlpltvpv, to_acq_xyzu], axis=1)
    acq_order = df_tlpltvpv_xyzu.sort_values(["light", "pan", "tilt", "roll"])

    acq_order_tlpltvpv = acq_order[["tl", "pl", "tv", "pv"]].values
    acq_order_xyzu = acq_order[["pan", "tilt", "roll", "light"]].values
    return acq_order_tlpltvpv, acq_order_xyzu


def main() -> int:
    if Path(NPZ_FILENAME_TO_SAVE).exists():
        print(f"file: [{NPZ_FILENAME_TO_SAVE}] already exists.")
        return 2
    if not Path(NPY_FILENAME_FOR_CAMERA_MATRIX).exists():
        print(f"file: [{NPY_FILENAME_FOR_CAMERA_MATRIX}] does not exist.")
        return 4

    print("Waiting Aries.")
    stage = Aries()

    # 撮影する角度を決める
    print("Optimizing acquiring order.")
    to_acq = preset_4d_169shots()
    scheduled_tlpltvpv, scheduled_xyzu = schedule_acq(to_acq)

    # stageを動かす
    stage.safety_stop = False
    stage.safety_u_axis = False
    stage.position = scheduled_xyzu[0]

    # カメラの初期設定
    cap = EasyPySpin.VideoCaptureEX(0)
    if not cap.isOpened():
        print("Acq-BTF: Camera device error.")
        return 1
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.average_num = ACQ_AVERAGE

    # カメラパラメータ行列読み込み
    obj_to_img_mat = np.load(NPY_FILENAME_FOR_CAMERA_MATRIX)
    dst_imgpoints = np.array(
        ((0, 0), (IMG_SIZE[0], 0), (0, IMG_SIZE[1]), (IMG_SIZE[0], IMG_SIZE[1])),
        dtype=np.float32
    )

    # 画像保存用のメモリ確保
    frames = np.empty((len(scheduled_xyzu), *IMG_SIZE, 3), dtype=np.float32)
    tqdm_xyzu = trange(len(scheduled_xyzu))
    for i in tqdm_xyzu:
        # tqdm更新
        xyzu = scheduled_xyzu[i]
        tqdm_xyzu.set_description(
            f"[X: {xyzu[0]:3.0f}, Y: {xyzu[1]:2.0f}, "
            + f"Z: {xyzu[2]:3.0f}, U: {xyzu[3]:3.0f}]"
        )

        # stageが動き切るまで待って撮影
        stage.sleep_until_stop()
        is_valid, frame = cap.readHDR(t_min=ACQ_T_MIN_US, t_max=ACQ_T_MAX_US, t_ref=ACQ_T_REF_US)

        # stageを動かす
        if i + 1 < len(scheduled_xyzu):
            stage.position = scheduled_xyzu[i + 1]

        if not is_valid:
            print("capture failed: ", scheduled_tlpltvpv[i])
            frames[i] = np.zeros((*IMG_SIZE, 3))

        # デモザイキング
        frame = cd.demosaicing_CFA_Bayer_bilinear(frame, pattern="BGGR")

        # 素材の四隅がカメラのどこに写るか計算する
        world_rot = rot_matrix_from_pan_tilt_roll(xyzu[0], xyzu[1], xyzu[2])
        material_edges = np.array(_WORLD_LT_RT_LB_RB, dtype=np.float) * 0.9
        material_edges = world_rot @ material_edges.T
        src_imgpoints = wrap_homogeneous_dot(obj_to_img_mat, material_edges.T)

        # 射影変換を行い、素材の正面画像を得る
        img_to_img_mat = cv2.getPerspectiveTransform(src_imgpoints.astype(np.float32), dst_imgpoints)
        frame = cv2.warpPerspective(
            frame, img_to_img_mat, IMG_SIZE, flags=cv2.INTER_CUBIC
        )

        # 画像を保存
        frames[i] = frame.astype(np.float32)

    cap.release()
    del stage

    print("compressing npz")
    # TODO: 露光時間などのconditionsをnpzに記録する
    np.savez_compressed(
        NPZ_FILENAME_TO_SAVE,
        images=frames,
        angles=scheduled_tlpltvpv,
    )
    return 0


if __name__ == "__main__":
    main()
