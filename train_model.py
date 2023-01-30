from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import pandas

array = pandas.read_csv("processed_blackjack_dataset.csv").to_numpy()

X = array[:, :2]
Y = array[:2, 6:7]

print(Y)

# define the architecture of the model
model = Sequential()
#model.add(Input(shape=(3,)))
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

# compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model on the dataset
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# model.save("trained_model")