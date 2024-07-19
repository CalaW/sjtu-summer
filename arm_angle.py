import json

import numpy as np
from scipy.signal import savgol_filter

from load_data import get_best_pred, load_data

keypoints = dict[str, list]


def get_arm_len(frames: list[list[keypoints]]) -> tuple:
    l_arm, r_arm = [], []
    for kps in frames:
        cur_kp = get_best_pred(kps)
        r_arm.append(np.linalg.norm(np.array(cur_kp[6]) - np.array(cur_kp[10])))
        l_arm.append(np.linalg.norm(np.array(cur_kp[5]) - np.array(cur_kp[9])))
    return (np.mean(r_arm), np.mean(l_arm))


def get_arm_angle(cur_kp, r_arm, l_arm):
    cur_r_arm = (cur_kp[6][1] - cur_kp[10][1]) / r_arm
    if cur_r_arm < -1:
        cur_r_arm = -1
    elif cur_r_arm > 1:
        cur_r_arm = 1
    cur_l_arm = (cur_kp[5][1] - cur_kp[9][1]) / l_arm
    if cur_l_arm < -1:
        cur_l_arm = -1
    elif cur_l_arm > 1:
        cur_l_arm = 1

    r_angle = np.pi / 2 - np.arcsin(abs(cur_r_arm))
    if cur_r_arm > 0:
        r_angle = np.pi / 2 + np.arcsin(abs(cur_r_arm))
    l_angle = np.pi / 2 - np.arcsin(abs(cur_l_arm))
    if cur_l_arm > 0:
        l_angle = np.pi / 2 + np.arcsin(abs(cur_l_arm))

    return [cur_r_arm, cur_l_arm, r_angle, l_angle]


def track_pose_2D(path, inferencer):
    all_kp = []
    cap = cv2.VideoCapture(path)

    [r_arm, l_arm] = get_arm_len(path, inferencer)
    r_ratio = []
    l_ratio = []
    r_angle = []
    l_angle = []

    i = 0
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                keypoints = get_kp(frame, inferencer)

                [cur_r_ratio, cur_l_ratio, cur_r_angle, cur_l_angle] = get_arm_angle(
                    keypoints, r_arm, l_arm
                )
                r_ratio.append(cur_r_ratio)
                l_ratio.append(cur_l_ratio)
                r_angle.append(cur_r_angle)
                l_angle.append(cur_l_angle)
            else:
                print("Failed to capture frame")
                break
        except RuntimeError as e:
            print("An error occurred", e)

        i += 1
        if i % 10 == 0:
            print(str(i) + " frame completed")

    cap.release()

    return [
        savgol_filter(r_ratio, 5, 2),
        savgol_filter(l_ratio, 5, 2),
        savgol_filter(r_angle, 5, 2),
        savgol_filter(l_angle, 5, 2),
    ]


if __name__ == "__main__":
    inferencer = MMPoseInferencer("human")
    path = "000.mp4"
    [a, b, c, d] = track_pose_2D(path, inferencer)
    # np.save('a1.npy',a)
    # np.save('a2.npy',b)
    # np.save('array1.npy',c)
    # np.save('array2.npy',d)
    plt.plot(np.degrees(c))
    plt.plot(np.degrees(d))
    plt.show()
