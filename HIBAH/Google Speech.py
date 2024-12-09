import os
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Path ke data audio
right_path = '/Users/carliapriansyahh/Downloads/pythonProject/voice/augmented_right'
left_path = '/Users/carliapriansyahh/Downloads/pythonProject/voice/augmented_left'

# Fungsi untuk mendapatkan semua path file audio
def get_audio_file_paths(folder_path):
    file_paths = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_paths.append(os.path.join(folder_path, file_name))
    return file_paths

# Load file dari folder
right_audio_files = get_audio_file_paths(right_path)
left_audio_files = get_audio_file_paths(left_path)

# Gabungkan file 'right' dan 'left' untuk pelatihan dan pengujian
audio_files = right_audio_files + left_audio_files
labels = [1] * len(right_audio_files) + [0] * len(left_audio_files)

# Bagi dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(audio_files, labels, test_size=0.3, random_state=42)

# Inisialisasi recognizer
recognizer = sr.Recognizer()

# Fungsi untuk mengenali ucapan menggunakan Google Speech Recognition API
def recognize_speech(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return 1 if "right" in text.lower() else 0
        except sr.UnknownValueError:
            return -1  # Jika pengenalan gagal
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return -1

# Prediksi dan evaluasi
y_pred = [recognize_speech(file_path) for file_path in X_test]

# Hanya gunakan prediksi yang valid (tidak -1)
valid_predictions = [(p, y) for p, y in zip(y_pred, y_test) if p != -1]
if valid_predictions:
    y_pred_filtered, y_test_filtered = zip(*valid_predictions)

    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
    precision = precision_score(y_test_filtered, y_pred_filtered)
    recall = recall_score(y_test_filtered, y_pred_filtered)

    # Cetak hasil
    print(f"Google API Model Accuracy: {accuracy:.2f}")
    print(f"Google API Model Precision: {precision:.2f}")
    print(f"Google API Model Recall: {recall:.2f}")
else:
    print("Tidak ada prediksi valid untuk dievaluasi.")