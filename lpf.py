import math


class LowPassFilter:
    def __init__(
        self, sampling_frequency, damping_frequency, damping_intensity, outlier_threshold=100
    ):
        self.sampling_frequency = sampling_frequency
        self.damping_frequency = damping_frequency
        self.damping_intensity = damping_intensity
        self.outlier_threshold = outlier_threshold
        self.filtered_old_value = None
        self.old_value = None
        self._a, self._b = self.compute_internal_params()

    def compute_internal_params(self):
        a1 = math.exp(
            -1.0
            / self.sampling_frequency
            * (2.0 * math.pi * self.damping_frequency)
            / (10.0 ** (self.damping_intensity / -10.0))
        )
        b1 = 1.0 - a1
        return a1, b1

    def reset(self):
        self.filtered_old_value = None
        self.old_value = None

    def is_outlier(self, new_value):
        if self.filtered_old_value is None:
            return False  # Not an outlier if it's the first value
        difference = abs(new_value - self.filtered_old_value)
        return difference > self.outlier_threshold

    def update(self, data_in):  # -> Any:
        if self.filtered_old_value is None:
            self.filtered_old_value = data_in
            self.old_value = data_in
            return data_in
        if self.is_outlier(data_in):
            print("Outlier detected. Skipping value.")
            data_in = self.filtered_old_value
        data_out = self._b * self.old_value + self._a * self.filtered_old_value
        self.filtered_old_value = data_out
        self.old_value = data_in
        return data_out


# Example usage:
# lpf = LowPassFilter(sampling_frequency=100, damping_frequency=1, damping_intensity=0.5)
# filtered_value = lpf.update(input_value)
