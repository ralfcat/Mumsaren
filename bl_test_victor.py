import tensorflow as tf
import numpy as np

# Preprocessing the dataset
dataset = [[10, 1, 1], [9, 9, 0], [5, 8, -2], [1, 10, 1], [8, 8, 0], [10, 10, -2], [7, 7, 0], [1, 10, 1], [5, 5, 0], [10, 5, 1]]

# Splitting the dataset into inputs and targets
inputs = [data[:2] for data in dataset]
targets = [data[2] for data in dataset]

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

        else:
            player_hands.append(get_input_hand("Enter player's hand for split hand: "))

    # Break the loop if all hands have a decision other than "hit" or "split"

    # Break the loop if all hands have a decision other than "hit" or "split"
    if all([decision != "hit" and decision != "split" for decision in [get_decision(model.predict([[hand, dealer_upcard]])[0][0]) for hand in player_hands]]):
        break
