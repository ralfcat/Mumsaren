from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import numpy as np

# define the architecture of the model
model = Sequential()
#model.add(Input(shape=(3,)))
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model on the dataset
#model.fit(X_train, y_train, epochs=10, batch_size=32)

a = np.array([[1,1,1,1,1,1,1,1]])
print(model.weights)
print(model(a))

model.save("trained_model")