import tensorflow as tf
import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv("newer_dataset1337.csv")

# Splitting the dataset into inputs and targets
inputs = dataset[['dealer_up','initial_hand','actions_taken']]
targets = dataset['win']

# Building the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(inputs, targets, epochs=100, batch_size=10)

# Function to map prediction to decision

def get_decision(prediction):
    if prediction > 0:
        return "hit"
    elif prediction == 0:
        return "stand"
    elif prediction == -1:
        return "split"
    else:
        return "double"

def get_input_hand(prompt):
    hand = input(prompt).split()
    try:
        hand = [int(x) for x in hand]
        return np.array(hand, dtype=np.int32)
    except:
        print("Invalid input, please enter a valid hand")
        return get_input_hand(prompt)

# Getting player's hand and dealer's upcard from the console
player_hands = [get_input_hand("Enter player's hand: ")]
dealer_upcard = get_input_hand("Enter dealer's upcard: ")

# Loop to keep making predictions until the decision is not "hit" or "split"
while True:
    # Loop over each hand
    for i, hand in enumerate(player_hands):
        # Making a prediction
        prediction = model.predict(np.array([[hand, dealer_upcard]]))[0][0]

        # Mapping the prediction to a decision
        decision = get_decision(prediction)
        print("Hand {}: Decision - {}".format(i + 1, decision))

        if decision != "hit" and decision != "split":
            break
        elif decision == "hit":
            player_hands[i] = get_input_hand("Enter player's hand: ")
            if player_hands[i] == "bust":
                break
        else:
            player_hands.append(get_input_hand("Enter player's hand for split hand: "))

    # Break the loop if all hands have a decision other than "hit" or "split"

    # Break the loop if all hands have a decision other than "hit" or "split"
    if all([decision != "hit" and decision != "split" for decision in [get_decision(model.predict([[hand, dealer_upcard]])[0][0]) for hand in player_hands]]):
        break

#Not working :(