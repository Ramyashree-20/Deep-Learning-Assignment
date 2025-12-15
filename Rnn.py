import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

text = "The beautiful girl whom I met last time is very intelligent also"

chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

seq_length = 5
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[c] for c in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

X_one_hot = tf.one_hot(X, len(chars))
y_one_hot = tf.one_hot(y, len(chars))

text_len = 50

model = Sequential()
model.add(SimpleRNN(
    32,                                # CHANGED: fewer RNN units
    activation='tanh',                 # CHANGED: better for RNN
    input_shape=(seq_length, len(chars))
))
model.add(Dropout(0.2))                # ADDED: prevent overfitting
model.add(Dense(len(chars), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_one_hot, y_one_hot, epochs=50)   # CHANGED: fewer epochs

start_seq = "The handsome boy whom I met "
generated_text = start_seq

# Text generation
for _ in range(text_len):
    last_seq = generated_text[-seq_length:]
    x = np.array([[char_to_index[c] for c in last_seq]])
    x_one_hot = tf.one_hot(x, len(chars))
    
    prediction = model.predict(x_one_hot, verbose=0)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    
    generated_text += next_char

print("Generated Text:")
print(generated_text)
