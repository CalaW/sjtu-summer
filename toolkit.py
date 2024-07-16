import json
import time
from glob import glob
from os.path import dirname

import cv2
import numpy as np

datasetRoot = "/Volumes/Samsung T7/2024SummerSchoolData/"


def read_label(txt_path):
    file = open(txt_path, encoding="utf-8").readlines()
    sub_id = []
    label = []
    length = len(file)
    for i in range(0, length - 1):
        label.append(file[i + 1].strip("\n").split("\t"))
        sub_id.append(label[i][0])

    return np.asarray(label).astype(np.float32), np.asarray(sub_id).astype(np.uint8)


def decode_depth_16(rgb):
    """
    Args:
        rgb: encoded depth image (w, h, 3), 8 bits

    Returns:
        depth: decoded depth image (w, h), 16 bits
    """

    assert rgb.dtype == np.uint8
    r, g, b = cv2.split(rgb)
    depth = (
        ((r.astype(np.uint16) + g.astype(np.uint16)) / 2) + (b.astype(np.uint16) // 16) * 256
    ).astype(np.uint16)
    return depth


def read_intrinsics(filename):
    """
    Args:
        filename: corresponding '.mp4' file name

    Returns:
        intrinsics: intrinsic matrix after 90 degree counterclockwise rotation
    """
    json_name = glob(dirname(filename) + "/*Param_*.json")[0]
    try:
        data = json.load(open(json_name))
        intrinsics = np.array(
            [
                [data["fy"], 0, data["height"] - data["ppy"] - 1],
                [0, data["fx"], data["ppx"]],
                [0, 0, 1],
            ]
        )
        return intrinsics

    except FileNotFoundError:
        print(f"No json file {json_name} found")


if __name__ == "__main__":
    # read labels
    label, sub_id = read_label(datasetRoot + "labels.txt")

    # take a look of a video
    deom_id = 0
    mp4 = glob(datasetRoot + f"/*/{sub_id[deom_id]:05d}/*.mp4")[0]
    left_arm_label, right_arm_label = label[deom_id][1], label[deom_id][2]
    cap = cv2.VideoCapture(mp4)

    # read corresponding intrinsic matrix
    intrinsic = read_intrinsics(mp4)

    cv2.namedWindow("Color and Depth Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color and Depth Image", 800, 600)
    while True:
        t0 = time.time()
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        color = frame[0:720, :]
        depth = frame[720:1440, :]
        # Example: create a depth colormap for visualization
        depth = decode_depth_16(depth)

        color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

        show_image = np.hstack(
            (color, cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.06), cv2.COLORMAP_JET))
        )
        cv2.imshow("Color and Depth Image", show_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
