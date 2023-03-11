# 1 - Fusão de características de Deep Learning com características artificiais
# 2 - Adição de carecterísticas de Deep Learning no BioAutoML

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Generate a synthetic dataset of DNA sequences and binary labels
seq_len = 50
num_seqs = 1000
X = np.random.choice(['A', 'C', 'G', 'T'], size=(num_seqs, seq_len))
y = np.random.randint(2, size=num_seqs)

# Encode the sequences using one-hot encoding and pad them to a fixed length
max_len = 100
def one_hot_encode(seq):
    seq_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return [seq_dict[c] for c in seq]

X = np.array([one_hot_encode(seq) for seq in X])
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

# Split the dataset into training and validation sets
train_size = int(0.8 * num_seqs)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, 4)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Extract features using the LSTM layer
lstm_layer = model.layers[0]
lstm_model = tf.keras.Model(inputs=lstm_layer.input, outputs=lstm_layer.output)
features = lstm_model.predict(X)

print(features.shape)
