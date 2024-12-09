import os
import librosa
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Fungsi untuk mengekstrak MFCC dari file audio
def extract_mfcc(file_path, n_mfcc=13, max_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# Fungsi untuk memproses semua file audio dalam folder
def process_folder(folder_path, label, n_mfcc=13, max_len=100):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, max_len=max_len)
            data.append(mfcc)
            labels.append(label)
    return np.array(data), np.array(labels)

# Path ke data
right_path = '/Users/carliapriansyahh/Downloads/pythonProject/speech_commands_v0.01/right'
left_path = '/Users/carliapriansyahh/Downloads/pythonProject/speech_commands_v0.01/left'

# Load data
right_data, right_labels = process_folder(right_path, label=1)
left_data, left_labels = process_folder(left_path, label=0)

# Gabungkan data
X = np.concatenate((right_data, left_data), axis=0)
y = np.concatenate((right_labels, left_labels), axis=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Melatih model HMM
hmm_model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
hmm_model.fit(np.vstack(X_train))

# Prediksi
y_pred = []
for sample in X_test:
    log_likelihood = hmm_model.score(sample)
    y_pred.append(1 if log_likelihood > 0 else 0)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"HMM Accuracy: {accuracy:.2f}")
print(f"HMM Precision: {precision:.2f}")
print(f"HMM Recall: {recall:.2f}")