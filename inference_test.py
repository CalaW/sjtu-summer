import matplotlib.pyplot as plt
from mmpose.apis import MMPoseInferencer

img_path = (
    # "mmpose-dev-1.x/tests/data/coco/000000000785.jpg"  # replace this with your own image path
    "mmpose-dev-1.x/tests/data/h36m/S5/S5_SittingDown.54138969/S5_SittingDown.54138969_002061.jpg"  # replace this with your own image path
)

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer(pose3d="human3d")

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)

keypoints = result["predictions"][0][0]["keypoints"]
print(keypoints)

# plot keypoints in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for kp in keypoints:
    ax.scatter(kp[0], kp[1], kp[2], c="r", marker="o")

ax.set_box_aspect([1, 1, 1])

plt.show()
