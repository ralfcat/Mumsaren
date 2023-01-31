import pandas as pd
import numpy as np

df = pd.read_csv("dataset_true.csv")
df = df[["dealer_up", "initial_hand","actions_taken"]]
df.to_csv("dataset_true.csv", index=False)