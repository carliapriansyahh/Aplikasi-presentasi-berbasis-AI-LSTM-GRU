import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Definisi kelas Attention
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dense = Dense(1, activation='tanh')

    def call(self, inputs):
        attention = self.dense(inputs)
        attention = tf.nn.softmax(attention, axis=1)
        context = attention * inputs
        context = tf.reduce_sum(context, axis=1)
        return context

# Fungsi untuk ekstraksi MFCC
def extract_mfcc(file_path, n_mfcc=13, max_len=87):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# Fungsi untuk memproses folder data
def process_folder(folder_path, label, n_mfcc=13, max_len=87):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, max_len=max_len)
            data.append(mfcc)
            labels.append(label)
    return np.array(data), np.array(labels)

# Path data baru
right_augmented_path = '/Users/carliapriansyahh/Downloads/pythonProject/voice/augmented_right'
left_augmented_path = '/Users/carliapriansyahh/Downloads/pythonProject/voice/augmented_left'

# Memuat data augmented
right_data, right_labels = process_folder(right_augmented_path, label=1)
left_data, left_labels = process_folder(left_augmented_path, label=0)

# Menggabungkan data augmented
X_new = np.concatenate((right_data, left_data), axis=0)
y_new = np.concatenate((right_labels, left_labels), axis=0)

# Membagi data baru menjadi train dan test
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=SEED)

# Memuat model dari epoch 25
model_path = '/Users/carliapriansyahh/Downloads/pythonProject/HIBAH/best_model_epoch_25.keras'
model = load_model(model_path, custom_objects={"Attention": Attention})

# Menyusun kembali callbacks untuk fine-tuning
checkpoint = ModelCheckpoint(
    "fine_tuned_best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Fine-tune model dengan data baru (20 epoch saja)
history = model.fit(
    X_train_new,
    y_train_new,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_new, y_test_new),
    callbacks=[checkpoint, early_stopping]
)

# Evaluasi model setelah fine-tuning
evaluation = model.evaluate(X_test_new, y_test_new)
print(f"Fine-Tuned Model Accuracy: {evaluation[1]:.4f}")

# Menyimpan model terbaik setelah fine-tuning
model.save("Best CheckPoint LSTM-GRU.keras")