import cv2
import EasyPySpin
from aries import Aries


def main() -> None:
    stage = Aries()
    print(stage.position)

    cap = EasyPySpin.VideoCapture(0)
    is_valid, frame = cap.read()
    if is_valid:
        cv2.imwrite("acq_minimum-test.jpg", frame)
    pass


if __name__ == "__main__":
    main()
