import cv2
import EasyPySpin

CAM_ID: int = 0
VIEW_SCALE: float = 1.0


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

    while True:
        _, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

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
