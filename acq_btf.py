"""Aries4軸ステージでBTFを撮影する"""
from itertools import product
from typing import Any, Tuple

import colour_demosaicing as cd
import cv2
import EasyPySpin
import numpy as np
import pandas as pd
from aries import Aries
from nptyping import NDArray, Float16
from tqdm import tqdm

from transcoord import tlpltvpv_to_xyzu, xyzu_to_tlpltvpv

NPZ_FILENAMETO_SAVE = "SweatNavyBlue-1D.btf.npz"

# 露光時間（単位:us）
ACQ_T_MIN_US = 25000
ACQ_T_MAX_US = 250000
ACQ_T_REF_US = 100000

ACQ_GAIN = 0
ACQ_AVERAGE = 2

# クロップ座標左右上下(奇数可)
CROP_L = 800
CROP_R = 1312
CROP_T = 380
CROP_B = 892

cenetr = (CROP_L + CROP_R) / 2
acq_img_size = (CROP_B - CROP_T, CROP_R - CROP_L)


def acq_1d(start: float, stop: float, num: int) -> NDArray[(Any, 4), Float16]:
    """光源位置の1軸変化のみ取得。引数は`np.linspace`準拠。

    Returns:
        撮影したいtlpltvpvの組が入った`ndarray`: shape=(num, 4), dtype=np.float16
    """
    to_acq = np.zeros((num, 4), dtype=np.float16)
    to_acq_tl = np.linspace(start, stop, num=num)
    to_acq[:, 0] = to_acq_tl
    return to_acq


def acq_2d() -> NDArray[(Any, 4), Float16]:
    """光源位置の2軸変化のみ取得。

    Returns:
        撮影したいtlpltvpvの組が入った`ndarray`: shape=(num, 4), dtype=np.float16
    """
    # to_acq_tl = np.concatenate(
    #     (
    #         np.full(24, 15),
    #         np.full(24, 30),
    #         np.full(24, 45),
    #         np.full(24, 60),
    #         np.full(24, 75),
    #     )
    # )
    # to_acq_pl = np.concatenate(
    #     (
    #         # np.arange(10, 89),
    #         # np.arange(10, 89)
    #         np.linspace(0, 345, int(360 / 15)),
    #         np.linspace(0, 345, int(360 / 15)),
    #         np.linspace(0, 345, int(360 / 15)),
    #         np.linspace(0, 345, int(360 / 15)),
    #         np.linspace(0, 345, int(360 / 15))
    #         # np.linspace(10.0, 89.5, int(80 / 0.5)), np.linspace(10.0, 89.5, int(80 / 0.5))
    #     )
    # )

    return np.array([(0, 0, 0, 0)], dtype=np.float16)


def acq_4d() -> NDArray[(Any, 4), Float16]:
    """光源・カメラ位置の4軸変化を取得。

    Returns:
        撮影したいtlpltvpvの組が入った`ndarray`: shape=(num, 4), dtype=np.float16
    """
    # THETA_PHI_PAIR_TO_ACQ = (
    #     (0, 0),
    #     (30, 0),
    #     (30, 90),
    #     (30, 180),
    #     (30, 270),
    #     (60, 0),
    #     (60, 45),
    #     (60, 90),
    #     (60, 135),
    #     (60, 180),
    #     (60, 225),
    #     (60, 270),
    #     (60, 315),
    # )
    THETA_PHI_PAIR_TO_ACQ = (
        (0, 0),
        (45, 0),
        (45, 90),
        (45, 180),
        (45, 270),
    )

    # to_acq_tlpltvpv = product(THETA_PHI_PAIR_TO_ACQ, THETA_PHI_PAIR_TO_ACQ)
    # to_acq_tlpltvpv = pd.DataFrame(
    #     [light + view for light, view in to_acq_tlpltvpv],
    #     columns=["tl", "pl", "tv", "pv"],
    # )

    return np.array([(0, 0, 0, 0)], dtype=np.float16)


def schedule_acq(
    to_acq: NDArray[(Any, 4), Float16]
) -> Tuple[NDArray[(Any, 4), Float16], NDArray[(Any, 4), Float16]]:
    """ステージ移動時間がなるべく短くなるように撮影順序を決める。

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
        [tlpltvpv_to_xyzu(*tlpltvpv) for tlpltvpv in to_acq_tlpltvpv.values],
        columns=["pan", "tilt", "roll", "light"],
    )
    to_acq_xyzu = to_acq_xyzu.round(4)

    df_tlpltvpv_xyzu = pd.concat([to_acq_tlpltvpv, to_acq_xyzu], axis=1)
    acq_order = df_tlpltvpv_xyzu.sort_values(["light", "pan", "tilt", "roll"])

    acq_order_tlpltvpv = acq_order[["tl", "pl", "tv", "pv"]].values
    acq_order_xyzu = acq_order[["pan", "tilt", "roll", "light"]].values
    return acq_order_tlpltvpv, acq_order_xyzu


def main() -> None:
    stage = Aries()

    to_acq = acq_1d(-89, 89, 89 + 89 + 1)
    scheduled_tlpltvpv, scheduled_xyzu = schedule_acq(to_acq)

    stage.safety_stop = False
    stage.safety_u_axis = False
    stage.position = scheduled_xyzu[0]  # stageが動いている間にカメラの初期設定をする

    cap = EasyPySpin.VideoCaptureEX(0)
    if not cap.isOpened():
        print("Acq-BTF: Camera device error.")
        return 1

    cap.set(cv2.CAP_PROP_GAIN, 0)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.average_num = 2
    #TODO: CAP_FRAME > acq_img_size

    frames = np.empty((len(scheduled_xyzu), *acq_img_size, 3), dtype=np.float32)
    tqdm_xyzu = tqdm(scheduled_xyzu)
    for i, xyzu in enumerate(tqdm_xyzu):
        tqdm_xyzu.set_description(
            f"[X: {xyzu[0]:3.0f}, Y: {xyzu[1]:2.0f}, "
            + f"Z: {xyzu[2]:3.0f}, U: {xyzu[3]:3.0f}]"
        )

        stage.sleep_until_stop()
        is_valid, frame = cap.readHDR(t_min=25000, t_max=250000, t_ref=100000)
        stage.position = xyzu  # stageが動いている間にデモザイキングとクロップ処理をする

        frame = cd.demosaicing_CFA_Bayer_bilinear(frame, pattern="BGGR")
        # TODO: 射影変換
        if is_valid:
            frames[i] = frame[CROP_T:CROP_B, CROP_L:CROP_R].astype(np.float32)
        else:
            print("capture failed: ", scheduled_tlpltvpv[i])
            frames[i] = np.zeros((*acq_img_size, 3))

    print("compressing npz")
    # TODO: 露光時間などのconditionsをnpzに記録する
    np.savez_compressed(
        NPZ_FILENAMETO_SAVE,
        images=frames,
        angles=scheduled_tlpltvpv,
    )


if __name__ == "__main__":
    main()
