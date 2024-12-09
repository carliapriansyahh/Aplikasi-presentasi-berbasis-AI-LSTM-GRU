import numpy as np
import librosa
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Layer, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def extract_mfcc(file_path, n_mfcc=13, max_len=87):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

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

right_path = '/Users/carliapriansyahh/Downloads/pythonProject/voice/speech_commands_v0.01/right'
left_path = '/Users/carliapriansyahh/Downloads/pythonProject/voice/speech_commands_v0.01/left'

right_data, right_labels = process_folder(right_path, label=1)
left_data, left_labels = process_folder(left_path, label=0)

X = np.concatenate((right_data, left_data), axis=0)
y = np.concatenate((right_labels, left_labels), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

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

def train_and_evaluate_model(epochs, model_save_path):
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(inputs)

    gru_out = GRU(64, return_sequences=True)(lstm_out)

    context = Attention()(gru_out)

    output = Dense(1, activation='sigmoid')(context)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

    y_pred_prob = model.predict(X_test)

    y_pred = (y_pred_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix (Epochs: {epochs})')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fpr_dict[epochs] = fpr
    tpr_dict[epochs] = tpr
    roc_auc_dict[epochs] = roc_auc

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Epochs: {epochs}")
    print(f"AUC: {roc_auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]}")

epochs_list = [5, 10, 15, 20, 25, 30]
for epochs in epochs_list:
    model_save_path = f"best_model_epoch_{epochs}.keras"
    train_and_evaluate_model(epochs, model_save_path)

plt.figure()

colors = ['blue', 'green', 'yellow', 'red', 'orange', 'purple']
for i, epochs in enumerate(epochs_list):
    plt.plot(fpr_dict[epochs], tpr_dict[epochs], color=colors[i], lw=2,
             label=f'ROC Epoch {epochs} (area = {roc_auc_dict[epochs]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Epochs')
plt.legend(loc="lower right")
plt.show()