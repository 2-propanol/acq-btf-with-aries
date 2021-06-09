"""カメラ位置調整とキャリブレーション結果の確認"""
import cv2
import EasyPySpin
import numpy as np
from aries import Aries

import calib_utils


CAM_ID: int = 0
VIEW_SCALE: float = 0.5

NPY_FILENAME_FOR_CAMERA_MATRIX = "camera_matrix_20201214.npy"

WORLD_LT_RT_LB_RB = np.array(((-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0)))
WORLD_LT_RT_LB_RB = WORLD_LT_RT_LB_RB + np.array((-0.005, 0.004, 0.02))
# WORLD_LT_RT_LB_RB = WORLD_LT_RT_LB_RB + np.array((-0.005, 0.004, 0.0))
WORLD_LT_RT_LB_RB = WORLD_LT_RT_LB_RB * 1

IMG_SIZE = (512, 512)


def main():
    cap = EasyPySpin.VideoCapture(0)
    if not cap.isOpened():
        print("Calib-Cam-Pos: Camera device error.")
        return 1

    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 横長画像を想定(cam_width > cam_height)
    left_border = (cam_width - cam_height) // 2
    right_border = cam_width - left_border
    max_object_size = int(cam_height / 2 ** 0.5)
    top_border = (cam_height - max_object_size) // 2
    bottom_border = cam_height - top_border
    dst_imgpoints = np.array(
        ((0, 0), (IMG_SIZE[0], 0), (0, IMG_SIZE[1]), (IMG_SIZE[0], IMG_SIZE[1])),
        dtype=np.float32,
    )

    cam_mat = np.load(NPY_FILENAME_FOR_CAMERA_MATRIX)

    stage = Aries()

    while True:
        rot = calib_utils.rot_matrix_from_pan_tilt_roll(*stage.position[0:3])

        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

        # 射影変換後の画像を表示(`WORLD_LT_RT_LB_RB`と露出の補正用)
        world_lt_rt_lb_rb = rot @ np.array(WORLD_LT_RT_LB_RB).T
        camera_lt_rt_lb_rb = calib_utils.wrap_homogeneous_dot(
            cam_mat, world_lt_rt_lb_rb.T
        ).astype(np.float32)
        img_to_img_mat = cv2.getPerspectiveTransform(camera_lt_rt_lb_rb, dst_imgpoints)
        frame_perspectived = cv2.warpPerspective(
            frame, img_to_img_mat, IMG_SIZE, flags=cv2.INTER_CUBIC
        )
        cv2.imshow(f"perspective", frame_perspectived)

        # カメラ画像を表示(カメラ位置の補正用)
        frame_preview = frame.copy()

        ## 中心ガイド線
        cv2.line(
            frame_preview,
            (left_border, top_border),
            (right_border, top_border),
            (255, 255, 255),
            5,
        )
        cv2.line(
            frame_preview,
            (left_border, bottom_border),
            (right_border, bottom_border),
            (255, 255, 255),
            5,
        )

        ## 最大素材サイズガイド線
        cv2.line(
            frame_preview,
            (left_border, 0),
            (right_border, cam_height),
            (255, 255, 255),
            5,
        )
        cv2.line(
            frame_preview,
            (right_border, 0),
            (left_border, cam_height),
            (255, 255, 255),
            5,
        )

        ## ステージ位置から推定した画像上の四隅点(右上が赤)
        camera_lt_rt_lb_rb = camera_lt_rt_lb_rb.astype(np.int32)
        camera_lt = tuple(camera_lt_rt_lb_rb[0])
        camera_rt = tuple(camera_lt_rt_lb_rb[1])
        camera_lb = tuple(camera_lt_rt_lb_rb[2])
        camera_rb = tuple(camera_lt_rt_lb_rb[3])
        cv2.circle(frame_preview, camera_lt, 20, (0, 0, 255), 5)
        cv2.circle(frame_preview, camera_rt, 20, (255, 0, 0), 5)
        cv2.circle(frame_preview, camera_lb, 20, (255, 0, 0), 5)
        cv2.circle(frame_preview, camera_rb, 20, (255, 0, 0), 5)

        frame_preview = cv2.resize(frame_preview, None, fx=VIEW_SCALE, fy=VIEW_SCALE)
        cv2.imshow("camera", frame_preview)

        # 射影変換後の最も暗いRBG値と最も明るいRGB値を取得(露出調整用の情報)
        flatten_bgr = frame_perspectived.T.reshape((3, -1))
        min_bgr = flatten_bgr.min(axis=1)
        max_bgr = flatten_bgr.max(axis=1)
        print(
            f"  darkest RGB: ({min_bgr[2]:3d},{min_bgr[1]:3d},{min_bgr[0]:3d})",
            end="\n",
        )
        print(
            f"brightest RGB: ({max_bgr[2]:3d},{max_bgr[1]:3d},{max_bgr[0]:3d})",
            end="\r\033[1A",
        )

        key = cv2.waitKey(25)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    print("\n\n")
    return 0


if __name__ == "__main__":
    main()
