from sktime.datasets import load_italy_power_demand

X_train, y_train = load_italy_power_demand(split="train", return_type="pd-multiindex")
print(X_train.to_dict(), y_train)

from sktime.registry import all_estimators

print(
    all_estimators("classifier", filter_tags={"capability:unequal_length": True}, as_dataframe=True)
)
