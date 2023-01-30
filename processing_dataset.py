import pandas as pd

# Load your data into a pandas DataFrame
df = pd.read_csv("processed_blackjack_dataset.csv")

# Filter the data to only include rows where the value in the 'wins' column is greater than or equal to 0
df = df[df["win"] >= 0]

# Save the updated DataFrame to a new file or overwrite the original file
df.to_csv("processed_blackjack_dataset", index=False)