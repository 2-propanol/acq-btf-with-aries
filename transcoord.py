"""coordinate transformation"""
import warnings
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def tlpltvpv_to_xyzu(
    tl: float, pl: float, tv: float, pv: float
) -> Tuple[float, float, float, float]:
    """BTF座標系の`tl`,`pl`,`tv`,`pv`を、aries座標系の`pan`,`tilt`,`roll`,`light`に変換する。

    Args:
        tl (float): [0, 90]
        pl (float): [0, 360)
        tv (float): [0, 90]
        pv (float): [0, 360)

    Returns:
        Tuple[float, float, float, float]: `pan`,`tilt`,`roll`,`light`
    """
    # 動径r=1としたときの球座標と見立てる
    # 素材の法線を(x,y,z)=(0,0,1)に固定したときの、カメラと光源との単位ベクトル。
    tl = np.deg2rad(tl, dtype=np.float64)
    pl = np.deg2rad(pl, dtype=np.float64)
    tv = np.deg2rad(tv, dtype=np.float64)
    pv = np.deg2rad(pv, dtype=np.float64)
    light = np.array([np.sin(tl) * np.cos(pl), np.sin(tl) * np.sin(pl), np.cos(tl)])
    camera = np.array([np.sin(tv) * np.cos(pv), np.sin(tv) * np.sin(pv), np.cos(tv)])
    material = np.array([0, 0, 1])

    # `camera`が[0,0,1]と一致するように回転させたときの、光源と素材の法線のベクトルを求める。
    if not np.allclose(camera, np.array([0, 0, 1])):
        axis = np.cross(camera, np.array([0, 0, 1]))
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.inner(camera, np.array([0, 0, 1])))
        rot = Rotation.from_rotvec(axis * angle)
        light = rot.apply(light)
        material = rot.apply(material)

        # 回転しても各ベクトルは長さ1。
        assert np.isclose(
            np.linalg.norm(light), 1
        ), "The 1st rotated light vector is not a unit vector."
        assert np.isclose(
            np.linalg.norm(material), 1
        ), "The 1st rotated material vector is not a unit vector."
        assert np.allclose(
            rot.apply(camera), np.array([0, 0, 1])
        ), "The 1st rotated camera vector is not [0, 0, 1]."

    # `camera`をそのままに、`light`がxz平面上にあるように全体を回転させる。
    if not np.isclose(light[1], 0):
        angle = np.sign(light[1]) * np.arccos(
            np.inner(np.array([light[0], light[1]]), np.array([1, 0]))
            / np.linalg.norm(np.array([light[0], light[1]]))
        )
        roll = np.rad2deg(angle) % 360
        rot = Rotation.from_rotvec(np.array([0, 0, 1]) * -angle)
        light = rot.apply(light)
        material = rot.apply(material)
        assert np.isclose(
            np.linalg.norm(light), 1
        ), "The 2nd rotated light vector is not a unit vector."
        assert np.isclose(
            light[1], 0
        ), "The 2nd rotated light vector is not in the xz plane."
        assert np.isclose(
            np.linalg.norm(material), 1
        ), "The 2nd rotated material vector is not a unit vector."
    else:
        roll = np.float32(0.0)

    # `material`がxz平面より上にあるようにする。
    if material[1] < 0:
        # rot = Rotation.from_rotvec(np.array([0, 0, 1]) * np.pi)
        # light = rot.apply(light)
        # material = rot.apply(material)
        light = light * np.array([-1, -1, 1])
        material = material * np.array([-1, -1, 1])
        roll = np.float32((roll + 180) % 360)
    assert material[1] >= 0, "The `tilt` value will over 90."

    # `pan`は、`material`をxz平面に投射したときのz軸とのなす角。
    pan = np.sign(material[0]) * np.arccos(
        material[2] / np.linalg.norm(np.array([material[0], material[2]]))
    )

    # `tilt`は、`material`とxz平面のなす角。
    if np.isclose(material[1], 0):
        # 浮動小数点誤差によるnan回避（ex. arccos(1.0000000000000002 / 1.0)）
        tilt = np.float32(0.0)
    else:
        material_xz = np.array([material[0], 0, material[2]])
        tilt = np.arccos(np.inner(material, material_xz) / np.linalg.norm(material_xz))

    # `light`は、(既にxz平面上にある)`light`とz軸とのなす角。
    if np.isclose(light[2], 1):
        # 浮動小数点誤差によるnan回避（ex. arccos(1.0000000000000002)）
        light = np.float32(0.0)
    else:
        light = np.sign(light[0]) * np.arccos(light[2])

    pan = -np.rad2deg(pan)
    tilt = 90 - np.rad2deg(tilt)
    light = -np.rad2deg(light)

    return (
        pan.astype(np.float32),
        tilt.astype(np.float32),
        roll.astype(np.float32),
        light.astype(np.float32),
    )


def xyzu_to_tlpltvpv(
    pan: float, tilt: float, roll: float, light: float
) -> Tuple[float, float, float, float]:
    """aries座標系の`pan`,`tilt`,`roll`,`light`を、BTF座標系の`tl`,`pl`,`tv`,`pv`に変換する。

    Args:
        pan (float): [-90, 90]
        tilt (float): [0, 90]
        roll (float): [0, 360)
        light (float): [-180, 180]

    Returns:
        Tuple[float, float, float, float]: `tl`,`pl`,`tv`,`pv`
    """
    # degree を radian に変換
    pan_radian = np.deg2rad(-pan, dtype=np.float64)
    tilt_radian = np.deg2rad(tilt - 90, dtype=np.float64)
    roll_radian = np.deg2rad(roll, dtype=np.float64)
    light_radian = np.deg2rad(-light, dtype=np.float64)

    # カメラを(x,y,z)=(0,0,1)に固定したときの、光源と素材の法線の単位ベクトル。
    camera_rect = np.array([0, 0, 1])
    light_rect = np.array([np.sin(light_radian), 0, np.cos(light_radian)])
    rot = Rotation.from_rotvec(np.array([0, 1, 0]) * pan_radian)
    material_rect = rot.apply(np.array([0, 0, 1]))
    tilt_axis = rot.apply(np.array([1, 0, 0]))
    rot = Rotation.from_rotvec(tilt_axis * tilt_radian)
    material_rect = rot.apply(material_rect)

    # 素材の法線を軸にカメラと光源をroll分回転させる。
    rot = Rotation.from_rotvec(material_rect * roll_radian)
    camera_rect = rot.apply(camera_rect)
    light_rect = rot.apply(light_rect)

    # 回転しても各ベクトルは長さ1。
    assert np.isclose(
        np.linalg.norm(camera_rect), 1
    ), "The 1st rotated camera vector is not a unit vector."
    assert np.isclose(
        np.linalg.norm(light_rect), 1
    ), "The 1st rotated light vector is not a unit vector."

    # 素材の法線をz軸と一致するように回転させたときの、カメラと光源のベクトルを求める。
    if not np.allclose(material_rect, np.array([0, 0, 1])):
        axis = np.cross(material_rect, np.array([0, 0, 1]))
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.inner(material_rect, np.array([0, 0, 1])))
        rot = Rotation.from_rotvec(axis * angle)
        camera_rect = rot.apply(camera_rect)
        light_rect = rot.apply(light_rect)

        # 回転しても各ベクトルは長さ1。
        assert np.isclose(
            np.linalg.norm(camera_rect), 1
        ), "The 2nd rotated camera vector is not a unit vector."
        assert np.isclose(
            np.linalg.norm(light_rect), 1
        ), "The 2nd rotated light vector is not a unit vector."
        assert np.allclose(
            rot.apply(material_rect), np.array([0, 0, 1])
        ), "The rotated material vector is not [0, 0, 1]."

    # z軸から見たときのtl, pl。
    if np.isclose(light_rect[2], 1):
        tl = np.float32(0.0)
        pl = np.float32(0.0)
    else:
        tl = np.rad2deg(np.arccos(light_rect[2]))
        pl = np.sign(light_rect[1]) * np.rad2deg(
            np.arccos(light_rect[0] / np.linalg.norm([light_rect[0], light_rect[1]]))
        )
        pl %= 360

    # z軸から見たときのtv, pv。
    if np.isclose(camera_rect[2], 1):
        tv = np.float32(0.0)
        pv = np.float32(0.0)
    else:
        tv = np.rad2deg(np.arccos(camera_rect[2]))
        pv = np.sign(camera_rect[1]) * np.rad2deg(
            np.arccos(camera_rect[0] / np.linalg.norm([camera_rect[0], camera_rect[1]]))
        )
        pv %= 360

    # 傾斜角度が90度を超えているときに警告を出す。
    if __debug__:
        if tl > 90:
            warnings.warn("The `tl` value is over 90.", RuntimeWarning)
        if tv > 90:
            warnings.warn("The `tv` value is over 90.", RuntimeWarning)

    return (
        tl.astype(np.float32),
        pl.astype(np.float32),
        tv.astype(np.float32),
        pv.astype(np.float32),
    )
