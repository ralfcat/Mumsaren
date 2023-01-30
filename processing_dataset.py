import pandas as pd

# Load your dataset into a Pandas dataframe
df = pd.read_csv("blackjack_sim.csv")


# Specify the list of columns to keep
columns_to_keep = ["dealer_up", "initial_hand", "dealer_final", "dealer_final_value", 
                   "player_final", "player_final_value", "actions_taken", "win"]

# Keep only the specified columns in the dataframe
df = df[columns_to_keep]

# Save the processed dataframe to a new file
df.to_csv("processed_blackjack_dataset.csv", index=False)