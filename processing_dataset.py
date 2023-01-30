import pandas as pd

# Load your data into a DataFrame
df = pd.read_csv("dataset_bj_removed.csv")

# Get the "player_final_value" column
dealer_final_value = df['dealer_final_value']

# Go through every element in the column
for i in range(len(dealer_final_value)):
    # Check if the value is equal to "['BJ']"
    if dealer_final_value[i] == "['BJ']":
        # Replace it with [21]
        dealer_final_value[i] = [21]

# Update the "player_final_value" column in the dataframe with the modified values
df['dealer_final_value'] = dealer_final_value
# Save the modified DataFrame to a new file
df.to_csv("dataset_bj_removed.csv", index=False)