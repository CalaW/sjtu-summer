import cv2
import matplotlib.pyplot as plt
from mmpose.apis import MMPoseInferencer

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise OSError("Cannot open webcam")

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer(pose3d="human3d")

while True:
    try:
        ret, frame = cap.read()
        if ret:
            result_generator = inferencer(frame, show=True)
            result = next(result_generator)

            keypoints = result["predictions"][0][0]["keypoints"]
            print(keypoints)

            # plot keypoints in 3d
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            # Add plotting code here to visualize keypoints
            for kp in keypoints:
                ax.scatter(kp[0], kp[1], kp[2], c="r", marker="o")

            ax.set_box_aspect([1, 1, 1])

            plt.show()
        else:
            print("Failed to capture frame")
            break
    except RuntimeError as e:
        print("An error occurred", e)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
