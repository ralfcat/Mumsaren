import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset into a Pandas dataframe
df = pd.read_csv("blackjack_dataset.csv")

# Define the columns to use for training the model
columns_to_use = ['dealer_up', 'initial_hand', 'dealer_final', 'dealer_final_value', 'player_final', 'player_final_value', 'actions_taken', 'win']

# Preprocess the categorical variables
categorical_vars = ['dealer_up', 'initial_hand', 'dealer_final', 'actions_taken']
for col in categorical_vars:
    df[col] = LabelEncoder().fit_transform(df[col])

# Normalize the numerical variables
scaler = StandardScaler()
df[['dealer_final_value', 'player_final_value']] = scaler.fit_transform(df[['dealer_final_value', 'player_final_value']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[columns_to_use], df['win'], test_size=0.2, random_state=0)

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
