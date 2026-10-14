import pandas as pd

train = pd.read_csv("outputs/casting_train_features.csv")

print(train.shape)

print("\n컬럼 수:")
print(len(train.columns))

print("\n컬럼 목록:")
for col in train.columns:
    print(col)
