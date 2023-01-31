from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import pandas as pd

dataset = pd.read_csv("processed_blackjack_dataset.csv").to_numpy()

Dealer = dataset[:, 0]
Hand = dataset[:, 1]
Y = dataset[:2, 6:7]

X = np.zeros(dataset.shape[0], )
for i in range(dataset.shape[0]):


# X = np.array([[dealer, c1, c2, c3, c4, c5]])
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

card1_array = np.zeros(Xtest)
for hand in X:

Xtest = X[:2, 1]
print(Xtest)
Xtest = Xtest.replace(",", "")
print(Xtest[1:-1].split())