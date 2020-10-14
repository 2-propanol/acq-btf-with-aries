"""カメラ位置調整とキャリブレーション結果の確認"""
import cv2
import EasyPySpin
import numpy as np
from aries import Aries

import calib_cam_stage


CAM_ID: int = 0
VIEW_SCALE: float = 0.5


def main():
    cap = EasyPySpin.VideoCapture(0)
    if not cap.isOpened():
        print("Calib-Cam-Pos: Camera device error.")
        return 1

    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # assume landscape (cam_width > cam_height)
    left_border = (cam_width - cam_height) // 2
    right_border = cam_width - left_border

    max_object_size = int(cam_height / 2 ** 0.5)
    top_border = (cam_height - max_object_size) // 2
    bottom_border = cam_height - top_border

    stage = Aries()
    pts = calib_cam_stage.obj_and_img_points_from_csv("calib_cam_stage.csv")
    cam_mat = calib_cam_stage.calib_by_points(pts)

    while True:
        rot = calib_cam_stage.rot_matrix_from_pan_tilt_roll(*stage.position[0:3])

        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

        world_lt = rot @ np.array([-1, -1, 0])
        world_rt = rot @ np.array([1, -1, 0])
        world_lb = rot @ np.array([-1, 1, 0])
        world_rb = rot @ np.array([1, 1, 0])
        camera_lt = calib_cam_stage.wrap_homogeneous_dot(cam_mat, world_lt)
        camera_rt = calib_cam_stage.wrap_homogeneous_dot(cam_mat, world_rt)
        camera_lb = calib_cam_stage.wrap_homogeneous_dot(cam_mat, world_lb)
        camera_rb = calib_cam_stage.wrap_homogeneous_dot(cam_mat, world_rb)
        camera_lt = tuple(camera_lt.astype(np.int))
        camera_rt = tuple(camera_rt.astype(np.int))
        camera_lb = tuple(camera_lb.astype(np.int))
        camera_rb = tuple(camera_rb.astype(np.int))

        # 中心ガイド線
        frame = cv2.line(
            frame,
            (left_border, top_border),
            (right_border, top_border),
            (255, 255, 255),
            5,
        )
        frame = cv2.line(
            frame,
            (left_border, bottom_border),
            (right_border, bottom_border),
            (255, 255, 255),
            5,
        )

        # 最大素材サイズガイド線
        frame = cv2.line(
            frame, (left_border, 0), (right_border, cam_height), (255, 255, 255), 5
        )
        frame = cv2.line(
            frame, (right_border, 0), (left_border, cam_height), (255, 255, 255), 5
        )

        # ステージ位置から推定した画像上の四隅点(右上が赤)
        frame = cv2.circle(frame, camera_lt, 20, (0, 0, 255), 5)
        frame = cv2.circle(frame, camera_rt, 20, (255, 0, 0), 5)
        frame = cv2.circle(frame, camera_lb, 20, (255, 0, 0), 5)
        frame = cv2.circle(frame, camera_rb, 20, (255, 0, 0), 5)

        frame = cv2.resize(frame, None, fx=VIEW_SCALE, fy=VIEW_SCALE)
        cv2.imshow("press q to quit", frame)

        key = cv2.waitKey(100)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    return 0


if __name__ == "__main__":
    main()
