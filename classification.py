import numpy as np
import pandas as pd
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier, TimeSeriesSVC
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.sklearn import RotationForest
from sktime.transformations.panel.padder import PaddingTransformer

from arm_angle import track_arm_angles
from load_data import load_data, load_split
from lpf import LowPassFilter

data_dict, label = load_data()
split = load_split()

kp_train = {i: data_dict[i] for i in split["train"]}
kp_test = {i: data_dict[i] for i in split["test"]}
label_train = {i: label[i] for i in split["train"]}
label_test = {i: label[i] for i in split["test"]}

lpf = LowPassFilter(
    sampling_frequency=30, damping_frequency=3, damping_intensity=0.5, outlier_threshold=60
)

X_train, y_train = {}, []
X_test, y_test = {}, []
for kp, label_dict, X, y in (
    (kp_train, label_train, X_train, y_train),
    (kp_test, label_test, X_test, y_test),
):
    for i, sample_data in kp.items():
        r_angles, l_angles = track_arm_angles(sample_data)
        filtered_r_angles, filtered_l_angles = [], []
        lpf.reset()
        for angle in l_angles:
            filtered_l_angles.append(lpf.update(angle))
        lpf.reset()
        for angle in r_angles:
            filtered_r_angles.append(lpf.update(angle))
        for j, angle in enumerate(filtered_r_angles):
            X[(f"{i}L", j)] = [angle]
        for j, angle in enumerate(filtered_l_angles):
            X[(f"{i}R", j)] = [angle]
        y.append(label_dict[i]["RARM"])
        y.append(label_dict[i]["LARM"])

X_train = pd.DataFrame(X_train).T
X_test = pd.DataFrame(X_test).T
y_train = np.array(y_train)
y_test = np.array(y_test)

classifier = DummyClassifier(strategy="prior")  # 0.619
classifier = PaddingTransformer() * KNeighborsTimeSeriesClassifier()  # 0.857
classifier = TimeSeriesSVC()  # 0.761
classifier = PaddingTransformer() * TimeSeriesForestClassifier()  # 0.880
classifier = PaddingTransformer() * ResNetClassifier(n_epochs=20)  # 0.69
classifier = PaddingTransformer() * InceptionTimeClassifier(n_epochs=20, batch_size=16)  # 0.666
classifier = PaddingTransformer() * HIVECOTEV2()  # 0.762
classifier = PaddingTransformer() * ShapeletTransformClassifier(
    estimator=RotationForest(n_estimators=3),
    n_shapelet_samples=100,
    max_shapelets=10,
    batch_size=20,
)  # 0.857

classifier = PaddingTransformer() * RocketClassifier()  # 0.928

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
acc = np.mean(y_pred == y_test)
print(acc)
