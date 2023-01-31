import ast
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('dataset_true.csv')

# Define a function to convert the strings to lists and remove the double quotes
def convert_to_list(string):
    string = string.replace('"', '')
    return ast.literal_eval(string)

# Apply the function to the column of interest
df['initial_hand'] = df['initial_hand'].apply(convert_to_list)
df.to_csv('dataset_true2.csv', index=False)