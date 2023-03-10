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


def YToArray(elem):
    elem = str(elem)
    elem = elem.split("], [")[0]
    letters = ''.join(filter(str.isalnum, elem))
    if len(letters) == 0:
        return np.array([[0, 0, 0, 0, 0, 1]])
    letter = letters[0]
    if letter == "S":
        return np.array([[1, 0, 0, 0, 0, 0]])
    if letter == "H":
        return np.array([[0, 1, 0, 0, 0, 0]])
    if letter == "D":
        return np.array([[0, 0, 1, 0, 0, 0]])
    if letter == "P":
        return np.array([[0, 0, 0, 1, 0, 0]])
    if letter == "N":
        return np.array([[0, 0, 0, 0, 1, 0]])


def ArrayToY(elem):
    """
    Converts the output of the network to the desired action.
    """
    small = 1
    if np.linalg.norm(elem - np.array([[1, 0, 0, 0, 0, 0]])) <= small:
        return "Stand"
    elif np.linalg.norm(elem - np.array([[0, 1, 0, 0, 0, 0]])) <= small:
        return "Hit"
    elif np.linalg.norm(elem - np.array([[0, 0, 1, 0, 0, 0]])) <= small:
        return "Double Down"
    elif np.linalg.norm(elem - np.array([[0, 0, 0, 1, 0, 0]])) <= small:
        return "Split"
    elif np.linalg.norm(elem - np.array([[0, 0, 0, 0, 1, 0]])) <= small:
        return "No Insurance"
    elif np.linalg.norm(elem - np.array([[0, 0, 0, 0, 0, 1]])) <= small:
        return "Black Jack"
    

dataset = pd.read_csv("processed_blackjack_dataset.csv").to_numpy()

Dealer = dataset[:, 0]      #Shape: (n_datapoints, ) type: int 
Hand = dataset[:, 1]        #Shape: (n_datapoints, ) type: string of the form '[2, 3]'
Event = dataset[:, 6]       #Shape: (n_datapoints, ) type: string of the form '[['P', 'H'], ['H', 'S']]'

# Create X and Y such that:
# X = np.array([[dealer, card1, card2, card3, card4, card5]])
# Y = np.array([[action]])
X = np.zeros((dataset.shape[0], 6))
X [:, 0] = Dealer
for i in range(dataset.shape[0]):
    Hand_current = np.array([strToList(Hand[i])])
    X[i, 1:3] = Hand_current

Y = np.zeros((Event.shape[0], 6))
for i in range(Event.shape[0]):
    Y[i] = YToArray(Event[i])



X_train = X
Y_train = Y

# define the architecture of the model
model = Sequential()
model.add(Dense(32, input_dim=6, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model on the dataset
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# model.save("trained_model")

while True:
    inp = input("state ")
    ls = inp.split()
    while len(ls) < 6:
        ls.append(0)
    ls = [int(elem) for elem in ls]
    arr = np.array([ls])
    print(ArrayToY(model(arr)))

