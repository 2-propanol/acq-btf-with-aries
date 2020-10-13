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
        lt = rot @ np.array([-1, -1, 0])
        rt = rot @ np.array([1, -1, 0])
        lb = rot @ np.array([-1, 1, 0])
        rb = rot @ np.array([1, 1, 0])
        lt = np.append(lt, 1.0)
        rt = np.append(rt, 1.0)
        lb = np.append(lb, 1.0)
        rb = np.append(rb, 1.0)

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
        frame = cv2.line(
            frame, (left_border, 0), (right_border, cam_height), (255, 255, 255), 5
        )
        frame = cv2.line(
            frame, (right_border, 0), (left_border, cam_height), (255, 255, 255), 5
        )

        frame = cv2.circle(
            frame, tuple((cam_mat @ lt)[0:2].astype(np.int)), 20, (0, 0, 255), 5
        )
        frame = cv2.circle(
            frame, tuple((cam_mat @ rt)[0:2].astype(np.int)), 20, (255, 0, 0), 5
        )
        frame = cv2.circle(
            frame, tuple((cam_mat @ lb)[0:2].astype(np.int)), 20, (255, 0, 0), 5
        )
        frame = cv2.circle(
            frame, tuple((cam_mat @ rb)[0:2].astype(np.int)), 20, (255, 0, 0), 5
        )

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
