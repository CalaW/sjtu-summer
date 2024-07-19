import time
from pathlib import Path

import cv2
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm

from toolkit import datasetRoot, decode_depth_16, read_label

if __name__ == "__main__":
    label, sub_id = read_label(datasetRoot + "labels.txt")
    inferencer_2d = MMPoseInferencer("human")
    # inferencer_3d = MMPoseInferencer(pose3d="human3d")
    for sub_idx in tqdm([5, 29, 41, 43, 53, 55, 66, 81, 94, 104]):
        mp4 = next(Path(datasetRoot).glob(f"*/{sub_id[sub_idx]:05d}/*.mp4"))
        cap = cv2.VideoCapture(str(mp4))

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
            # dir = f"vis/{sub_id[sub_idx]}/3d/{i:05d}"
            # result_generator_3d = inferencer_3d(color, vis_out_dir=dir, pred_out_dir=dir)
            # result_3d = next(result_generator_3d)
            dir = f"vis/{sub_id[sub_idx]}/2d/{i:05d}"
            result_generator_2d = inferencer_2d(
                color,
                # vis_out_dir=dir,
                pred_out_dir=dir,
            )
            result_2d = next(result_generator_2d)
            i += 1
