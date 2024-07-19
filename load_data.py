import json
import pickle as pkl
from pathlib import Path

import numpy as np

keypoints = dict[str, list]

# Define the directory where the folders are located
root_dir = Path("vis_result")
save_file = root_dir / "data.pkl"


def load_data() -> dict[int, list[list[keypoints]]]:
    with save_file.open("rb") as f:
        data_dict = pkl.load(f)
    return data_dict


def get_best_pred(data: list[dict]) -> dict:
    return max(data, key=lambda x: np.mean(x["keypoint_scores"]))


# run data preprocessing
if __name__ == "__main__":
    data_dict: dict[int, list] = {}
    human_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for subj_dir in root_dir.glob("*/"):
        # Iterate through each directory in the root directory
        subj_id = int(subj_dir.name)
        data_dict[subj_id] = []
        for file_path in sorted(subj_dir.glob("2d/*")):
            if file_path.suffix != ".json":
                continue
            frame_id = int(file_path.stem)
            with file_path.open() as f:
                data = json.load(f)
            human_num[len(data)] += 1
            best_result = get_best_pred(data)
            data_dict[subj_id].append([best_result])

    with save_file.open("wb") as f:
        pkl.dump(data_dict, f)

    print(human_num)
