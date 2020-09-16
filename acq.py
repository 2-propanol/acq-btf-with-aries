from aries import Aries
import EasyPySpin
import numpy as np


def main() -> None:
    stage = Aries()
    stage.position = (0, 90, 0, 15)

    cap = EasyPySpin.VideoCapture(0)

    stage.sleep_until_stop()
    pos = list(stage.position)
    print(pos)

    frames = []

    for tv in np.linspace(-80, 80, num=17):
        pos[0] = tv
        stage.position = pos
        stage.sleep_until_stop()

        is_valid, frame = cap.read()
        if not is_valid:
            print("capture failed")
        frames.append(frame)

    np.savez_compressed("out.npz", np.array(frames))



if __name__ == "__main__":
    main()
