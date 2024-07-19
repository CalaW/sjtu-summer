import matplotlib.pyplot as plt
import numpy as np

from load_data import get_best_pred, keypoints, load_data
from lpf import LowPassFilter


def get_arm_len(frames: list[list[keypoints]]) -> tuple:
    l_arm, r_arm = [], []
    for kps in frames:
        cur_kp = get_best_pred(kps)["keypoints"]
        r_arm.append(np.linalg.norm(np.array(cur_kp[6]) - np.array(cur_kp[10])))
        l_arm.append(np.linalg.norm(np.array(cur_kp[5]) - np.array(cur_kp[9])))
    return np.mean(r_arm), np.mean(l_arm)


def calculate_arm_angles(cur_kp, r_len, l_len) -> tuple[float, float]:
    # Helper function to calculate angle based on arm difference
    def calc_angle(diff, arm_length) -> tuple[float, float]:
        arm_ratio = np.clip(diff / arm_length, -1, 1)  # Ensure the ratio is within [-1, 1]
        angle = np.pi / 2 - np.arcsin(abs(arm_ratio))
        if arm_ratio > 0:
            angle += np.pi / 2  # Adjust angle if arm_ratio is positive
        return arm_ratio, np.degrees(angle)

    # Calculate right and left arm ratios and angles
    cur_r_arm, r_angle = calc_angle(cur_kp[6][1] - cur_kp[10][1], r_len)
    cur_l_arm, l_angle = calc_angle(cur_kp[5][1] - cur_kp[9][1], l_len)

    return r_angle, l_angle


def track_arm_angles(frames: list[list[keypoints]], initial_frames: int = 5) -> tuple[list, list]:
    r_len, l_len = get_arm_len(frames[:initial_frames])
    r_angles = []
    l_angles = []

    for kps in frames:
        cur_kp = get_best_pred(kps)["keypoints"]
        cur_r_angle, cur_l_angle = calculate_arm_angles(cur_kp, r_len, l_len)
        r_angles.append(cur_r_angle)
        l_angles.append(cur_l_angle)
    return r_angles, l_angles


if __name__ == "__main__":
    for i, sample_data in load_data().items():
        sample_data = load_data()[7]
        r_angles, l_angles = track_arm_angles(sample_data)
        filtered_r_angles, filtered_l_angles = [], []
        lpf = LowPassFilter(
            sampling_frequency=30, damping_frequency=3, damping_intensity=0.5, outlier_threshold=60
        )
        for angle in l_angles:
            filtered_l_angles.append(lpf.update(angle))
        for angle in r_angles:
            filtered_r_angles.append(lpf.update(angle))
        plt.figure()
        plt.plot(r_angles)
        plt.plot(filtered_r_angles)
        # plt.savefig(f"{i}.pdf")
        plt.show()
        break
