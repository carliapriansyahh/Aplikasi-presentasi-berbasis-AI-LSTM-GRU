from pydub import AudioSegment
import os


# Function to split audio into 1-second chunks and save as .wav
def split_audio_to_wav(input_file, output_dir):
    # Load the audio file
    audio = AudioSegment.from_file(input_file, format="m4a")

    # Duration of each chunk in milliseconds (1 second = 1000 ms)
    chunk_length_ms = 1000

    # Total length of the audio in milliseconds
    total_length_ms = len(audio)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over the audio, split into chunks of 1 second
    for i in range(0, total_length_ms, chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file_name = os.path.join(output_dir, f"chunk_{i // 1000}.wav")
        chunk.export(chunk_file_name, format="wav")
        print(f"Saved {chunk_file_name}")


# Example usage
input_file = "/Users/carliapriansyahh/Downloads/pythonProject/Voice Test Real Word/Jalan Dr. Muwardi 1 No. 3 14.m4a"  # Replace with your file path
output_dir = "/Users/carliapriansyahh/Downloads/pythonProject/Voice Test Real Word"  # Replace with your desired output directory
split_audio_to_wav(input_file, output_dir)