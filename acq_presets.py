from itertools import product
from typing import Any

import numpy as np
from nptyping import Float16, NDArray


def acq_1d_light(start: float, stop: float, num: int) -> NDArray[(Any, 4), Float16]:
    """光源位置の1軸変化のみ取得。引数は`np.linspace`準拠。

    Returns:
        ndarray, shape=(`num`, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    to_acq = np.zeros((num, 4), dtype=np.float16)
    to_acq_tl = np.linspace(start, stop, num=num)
    to_acq[:, 0] = to_acq_tl
    return to_acq


def acq_1d_view(start: float, stop: float, num: int) -> NDArray[(Any, 4), Float16]:
    """視点位置の1軸変化のみ取得。引数は`np.linspace`準拠。

    Returns:
        ndarray, shape=(`num`, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    to_acq = np.zeros((num, 4), dtype=np.float16)
    to_acq_tv = np.linspace(start, stop, num=num)
    to_acq[:, 2] = to_acq_tv
    return to_acq


def acq_4d(light_theta_phi, view_theta_phi) -> NDArray[(Any, 4), Float16]:
    """光源・視点位置の4軸変化を取得。各光源位置について各視点位置で撮影する。

    Args:
        view_theta_phi (ndarray): 光源位置
        light_theta_phi (ndarray): 視点位置

    Returns:
        ndarray, shape=(view*light, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    to_acq = product(light_theta_phi, view_theta_phi)
    to_acq = np.array([light + view for light, view in to_acq], dtype=np.float16)
    return to_acq


def non_yokogiri_1D(start: float, stop: float, num: int) -> NDArray[(Any, 4), Float16]:
    to_acq = np.zeros((num, 4), dtype=np.float16)
    to_acq_tl = np.linspace(start, stop, num=num)
    to_acq[:, 0] = to_acq_tl
    to_acq[:, 2] = to_acq_tl + 10
    return to_acq


def preset_non_yokogiri_33shots():
    return non_yokogiri_1D(-85, 75, (85 // 5) + (75 // 5) + 1)


def preset_non_yokogiri_169shots():
    return non_yokogiri_1D(-89, 79, 89 + 79 + 1)


def preset_non_yokogiri_1699shots():
    return non_yokogiri_1D(-89.9, 79.9, 899 + 799 + 1)


def preset_1d_light_35shots() -> NDArray[(Any, 4), Float16]:
    """光源位置の1軸変化のみ取得。35枚(-85度から85度まで5度間隔)プリセット。

    Returns:
        ndarray, shape=(179, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    return acq_1d_light(-85, 85, (85 // 5) + (85 // 5) + 1)


def preset_1d_light_359shots() -> NDArray[(Any, 4), Float16]:
    """光源位置の1軸変化のみ取得。359枚(-89.5度から89.5度まで0.5度間隔)プリセット。

    Returns:
        ndarray, shape=(359, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    return acq_1d_light(-89.5, 89.5, 179 + 179 + 1)


def preset_1d_light_179shots() -> NDArray[(Any, 4), Float16]:
    """光源位置の1軸変化のみ取得。179枚(-89度から89度まで1度間隔)プリセット。

    Returns:
        ndarray, shape=(179, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    return acq_1d_light(-89, 89, 89 + 89 + 1)


def preset_2d_light_24shots() -> NDArray[(Any, 4), Float16]:
    """光源位置の2軸変化のみ取得。24枚(傾斜角45度固定・方位角15度間隔)プリセット。

    Returns:
        ndarray, shape=(24, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    theta_and_phidiff_list = ((45, 15),)
    pairlist = []
    for theta, phidiff in theta_and_phidiff_list:
        for i in range(int(360 / phidiff)):
            pairlist.append((theta, phidiff * i))

    return acq_4d(pairlist, ((0, 0),))


def preset_2d_light_120shots() -> NDArray[(Any, 4), Float16]:
    """光源位置の2軸変化のみ取得。120枚(傾斜角・方位角、各15度間隔)プリセット。

    Returns:
        ndarray, shape=(120, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    theta_and_phidiff_list = ((15, 15), (30, 15), (45, 15), (60, 15), (75, 15))
    pairlist = []
    for theta, phidiff in theta_and_phidiff_list:
        for i in range(int(360 / phidiff)):
            pairlist.append((theta, phidiff * i))

    return acq_4d(pairlist, ((0, 0),))


def preset_4d_25shots() -> NDArray[(Any, 4), Float16]:
    """光源・視点位置の4軸変化を取得。25枚(5x5)プリセット。

    Returns:
        ndarray, shape=(25, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    pairlist = (
        (10, 0),
        (30, 0),
        (30, 90),
        (30, 180),
        (30, 270),
    )

    return acq_4d(pairlist, pairlist)


def preset_4d_169shots() -> NDArray[(Any, 4), Float16]:
    """光源・視点位置の4軸変化を取得。169枚(13x13)プリセット。

    Returns:
        ndarray, shape=(169, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    theta_and_phidiff_list = ((30, 90), (60, 45))
    pairlist = [(0, 0)]
    for theta, phidiff in theta_and_phidiff_list:
        for i in range(int(360 / phidiff)):
            pairlist.append((theta, phidiff * i))

    return acq_4d(pairlist, pairlist)


def preset_4d_625shots() -> NDArray[(Any, 4), Float16]:
    """光源・視点位置の4軸変化を取得。625枚(25x25)プリセット。

    Returns:
        ndarray, shape=(625, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    theta_and_phidiff_list = ((30, 90), (45, 45), (60, 30))
    pairlist = [(0, 0)]
    for theta, phidiff in theta_and_phidiff_list:
        for i in range(int(360 / phidiff)):
            pairlist.append((theta, phidiff * i))

    return acq_4d(pairlist, pairlist)


def preset_4d_6561shots() -> NDArray[(Any, 4), Float16]:
    """光源・視点位置の4軸変化を取得。6561枚(81x81)プリセット。

    Returns:
        ndarray, shape=(6561, 4), dtype=np.float16: 撮影したいtlpltvpvの組
    """
    theta_and_phidiff_list = ((15, 60), (30, 30), (45, 20), (60, 18), (75, 15))
    pairlist = [(0, 0)]
    for theta, phidiff in theta_and_phidiff_list:
        for i in range(int(360 / phidiff)):
            pairlist.append((theta, phidiff * i))

    return acq_4d(pairlist, pairlist)
