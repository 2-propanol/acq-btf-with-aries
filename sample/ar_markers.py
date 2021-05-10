"""ARマーカー画像を生成する"""
import cv2
import numpy as np

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


def generate_ar_20x20(filename: str = "AR_marker_for_calib_20x20.png"):
    """解像度20x20のARマーカープレート画像を生成

    4x4(黒枠を含めて6x6)のARマーカが縦3列、横3行、計9個並ぶ
    出力例: "AR_marker_for_calib_20x20.png"
    """
    marker_plate = np.zeros((20, 20), dtype=np.uint8) + 255
    for i in range(9):
        marker = aruco.drawMarker(dictionary, i, 6)
        row = i % 3
        col = i // 3
        marker_plate[col * 7 : (col + 1) * 7 - 1, row * 7 : (row + 1) * 7 - 1] = marker
    cv2.imwrite(filename, marker_plate)


def generate_ar_40x40(filename: str = "AR_marker_for_calib_40x40.png"):
    """解像度40x40のARマーカープレート画像を生成

    4x4(黒枠を含めて6x6)のARマーカが縦5列、横5行、計25個並ぶ
    出力例: "AR_marker_for_calib_40x40.png"
    """
    marker_position = (
        # fmt: off
        ( 0,  0), ( 7,  0), (17,  0), (24,  0), (32,  0),
        ( 0,  8), ( 7, 10), (16,  7), (23, 10), (34,  9),
        ( 0, 19), ( 8, 18), (16, 16), (23, 17), (30, 16),
        ( 2, 27), ( 9, 25), (16, 23), (25, 24), (33, 23),
        ( 1, 34), ( 9, 32), (16, 34), (24, 31), (33, 33),
    )
    marker_plate = np.zeros((40, 40), dtype=np.uint8) + 255
    marker_list = [aruco.drawMarker(dictionary, i, 6) for i in range(25)]
    for position, marker in zip(marker_position, marker_list):
        row, col = position
        marker_plate[col : col + 6, row : row + 6] = marker
    cv2.imwrite(filename, marker_plate)


def test_ar_reader_with_camera(cam_id: int = 0):
    cap = cv2.VideoCapture(cam_id)
    while True:
        _, frame = cap.read()
        corners, ids, _ = aruco.detectMarkers(frame, dictionary)
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        cv2.imshow('detected markers ("q"key to quit)', frame)

        if cv2.waitKey(20) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_ar_20x20()
    generate_ar_40x40()

    if input("test aruco.detectMarkers? (this uses your camera) [y/N]: ").lower == "y":
        test_ar_reader_with_camera()
