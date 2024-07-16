import time
from glob import glob

import cv2
from mmpose.apis import MMPoseInferencer

from toolkit import dataroot, decode_depth_16, read_label

if __name__ == "__main__":
    label, sub_id = read_label(dataroot + "labels.txt")
    deom_id = 0
    mp4 = glob(dataroot + f"/*/{sub_id[deom_id]:05d}/*.mp4")[0]
    cap = cv2.VideoCapture(mp4)
    inferencer = MMPoseInferencer(pose3d="human3d")

    i = 0
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
        result_generator = inferencer(
            color, vis_out_dir=f"vis/{i:05d}", pred_out_dir=f"vis/{i:05d}"
        )
        result = next(result_generator)
        i += 1
