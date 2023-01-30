import pandas as pd
import numpy as np

df = pd.read_csv("blackjack_sim.csv")
df = df[["dealer_up", "initial_hand","win"]]
df["win"] = np.where(df["win"] > 0, 1, 0)
df.to_csv("newer_dataset1337.csv", index=False)