from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import pandas as pd

def strToList(str):
    """
    Input is a list of string type with integers, ex: "[1, 2, 3]"
    Output is said list as a list
    """
    templist = str[1:-1].replace(",", "").split()
    finallist = np.zeros(len(templist))
    for i, card in enumerate(templist):
        finallist[i] = int(card)
    return finallist

dataset = pd.read_csv("processed_blackjack_dataset.csv").to_numpy()[:10, :]

Dealer = dataset[:, 0]
Hand = dataset[:, 1]
Y = dataset[:2, 6:7]

X = np.zeros((dataset.shape[0], 6))
X [:, 0] = Dealer
for i in range(dataset.shape[0]):
    Hand_current = strToList(Hand[i])
    print(Hand_current)

# X = np.array([[dealer, card1, card2, card3, card4, card5]])
# Y = np.array([[action]])

print(Y)

# define the architecture of the model
model = Sequential()
model.add(Dense(16, input_dim=6, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))

# compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model on the dataset
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# model.save("trained_model")
