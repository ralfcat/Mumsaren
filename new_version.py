import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
dataset = pd.read_csv("blackjack_dataset_true.csv")

# Encode categorical variables
encoder = LabelEncoder()
dataset['dealer_up'] = encoder.fit_transform(dataset['dealer_up'])
dataset['initial_hand'] = encoder.fit_transform(dataset['initial_hand'])
dataset['actions_taken'] = encoder.fit_transform(dataset['actions_taken'])

# Split the dataset into training and testing sets
X = dataset.iloc[:, :2].values
Y = dataset.iloc[:, 2].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# One-hot encode the target variables
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Define the neural network model
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Evaluate the model
score = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (score[1]*100))

# Get the optimal action given a player's hand and dealer's upcard
def get_action(player_hand, dealer_upcard):
    player_hand = encoder.transform([player_hand])
    dealer_upcard = encoder.transform([dealer_upcard])
    X = np.array([[dealer_upcard, player_hand]])
    action_index = np.argmax(model.predict(X))
    action = encoder.inverse_transform([action_index])
    return action[0]

# Test the model
player_hand = input("Enter player's hand: ")
dealer_upcard = input("Enter dealer's upcard: ")
print("Optimal action:", get_action(player_hand, dealer_upcard))