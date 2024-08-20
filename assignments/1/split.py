import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data
df = pd.read_csv("./../../data/external/spotify.csv")
df = df.drop_duplicates(subset='track_id', keep="first")

# suffle the data
df = df.sample(frac=1).reset_index(drop=True)

# split the data
train_size = int(0.8 * len(df))
test_size = (len(df) - train_size)//2
validate_size = len(df) - train_size - test_size

df_train, df_test, df_validate = np.split(df, [train_size, train_size + test_size])

# save the data
df_train.to_csv("./train.csv", index=False)
df_test.to_csv("./test.csv", index=False)
df_validate.to_csv("./validate.csv", index=False)
