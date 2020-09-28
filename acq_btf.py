from itertools import product

# import cv2
import EasyPySpin

import numpy as np
import pandas as pd
from aries import Aries
from tqdm import tqdm

from transcoord import tlpltvpv_to_xyzu, xyzu_to_tlpltvpv

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


def main() -> None:
    stage = Aries()
    stage.safety_stop = False
    stage.safety_u_axis = False

    to_acq_tlpltvpv = product(THETA_PHI_PAIR_TO_ACQ, THETA_PHI_PAIR_TO_ACQ)
    to_acq_tlpltvpv = pd.DataFrame(
        [light + view for light, view in to_acq_tlpltvpv],
        columns=["tl", "pl", "tv", "pv"],
    )
    to_acq_xyzu = pd.DataFrame(
        [tlpltvpv_to_xyzu(*tlpltvpv) for tlpltvpv in to_acq_tlpltvpv.values],
        columns=["pan", "tilt", "roll", "light"],
    )
    to_acq_xyzu = to_acq_xyzu.round(4)

    df_tlpltvpv_xyzu = pd.concat([to_acq_tlpltvpv, to_acq_xyzu], axis=1)
    df_tlpltvpv_xyzu = df_tlpltvpv_xyzu.sort_values(["light", "pan", "tilt", "roll"])

    df_tlpltvpv_xyzu.to_csv("order.csv", columns=["tl", "pl", "tv", "pv"], index=False)
    scheduled_xyzu = df_tlpltvpv_xyzu[["pan", "tilt", "roll", "light"]].values

    cap = EasyPySpin.VideoCapture(0)
    frames = []
    for xyzu in tqdm(scheduled_xyzu):
        stage.position = xyzu
        stage.sleep_until_stop()

        is_valid, frame = cap.read()
        if is_valid:
            frames.append(frame)
        else:
            print("capture failed")

    np.savez_compressed("out.npz", np.array(frames))


if __name__ == "__main__":
    main()
