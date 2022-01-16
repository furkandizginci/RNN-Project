import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN, Dense, Activation

num_words = 15000 
(xtrain, ytrain), (xtest, ytest) = imdb.load_data(num_words = num_words)

word_index = imdb.get_word_index()

print(xtrain[0])

maxlen = 130
xtrain = pad_sequences(xtrain, maxlen = maxlen)
xtest = pad_sequences(xtest, maxlen = maxlen)

model = Sequential()

model.add(Embedding(num_words, 32, input_length = len(xtrain[0])))

model.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = True, activation = "relu"))
model.add(SimpleRNN(16))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

history = model.fit(xtrain, ytrain, 
                    validation_data = (xtest, ytest), 
                    epochs = 6,
                    batch_size = 128,
                    verbose = 1)

score = model.evaluate(xtest, ytest)
print("Accuracy: %", score[1]*100)

plt.figure()
plt.plot(history.history["accuracy"], label ="Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label ="Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
