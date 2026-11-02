from pathlib import Path

import numpy as np
import librosa
from PIL import Image
from scipy.signal import find_peaks

import math



# Define the folders
base_folder = Path(
    "/Users/jacksonnovotny/Desktop/Machine Learning Final Project/"
    "Spectogram Generators"
)

input_folder = base_folder / "birdSoundsClean"
output_folder = base_folder / "Converted Spectograms"

# Create the output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Audio file extensions to process
audio_extensions = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif"
}


# Consistent audio and spectrogram settings
sample_rate = 22050
n_fft = 2048
hop_length = 512
n_mels = 128
clip_seconds = 37
ignore_under = 9
target_samples = sample_rate * clip_seconds 
min_samples = sample_rate * ignore_under

# Find every audio file in the input folder
audio_files = [
    file for file in input_folder.rglob("*")
    if file.is_file() and file.suffix.lower() in audio_extensions
]

print(f"Found {len(audio_files)} audio files.")

successful = 0
failed = 0

# Get the max audio length for normalizing the data
max_duration = 0.0
min_duration = 65
total_length = 0.0

for index, filename in enumerate(audio_files, start=1):
    slices = 0
    try:
        print(f"[{index}/{len(audio_files)}] Processing: {filename.name}")

        # https://librosa.org/doc/latest/auto_tutorials/01-intro/01-load.html
        y, sr = librosa.load(filename, sr=sample_rate, mono=True)

        amplitude_envelope = np.abs(y)
        peaks, _ = find_peaks(amplitude_envelope, height=0.3, distance=sr*0.5)
        
        peak_times = peaks / sr

        # How long we want each spectogram time slice to be
        time_slice = 2

        if len(y) == 0:
            print(f"Skipping empty file: {filename.name}")
            failed += 1
            continue

        for peak in peak_times:
            # Makes one spectogram 1.5 seconds before the peak
            if (peak - 1.5 > 0 and peak + 0.5 < (len(y) / sr)):
                slices += 1

                idx_start = int((peak - 1.5 )* sr)
                idx_end = int((peak + 0.5) * sr)
                y_slice = y[idx_start:idx_end]   

                mel = librosa.feature.melspectrogram(
                    y=y_slice, sr=sample_rate, n_fft=n_fft,
                    hop_length=hop_length, n_mels=n_mels
                )

                spectrogram_db = librosa.power_to_db(mel, ref=np.max)
                spectrogram_db = np.clip(spectrogram_db, -80, 0)

                spectrogram_image = ((spectrogram_db + 80) / 80 * 255).astype(np.uint8)
                spectrogram_image = np.flipud(spectrogram_image)

                image = Image.fromarray(spectrogram_image)
                image.save(output_folder / f"{filename.stem}_spectrogram_slice{slices}.png")
                successful += 1
            # Makes one spectogram 0.5 seconds before the peak
            if (peak - 0.5 > 0 and peak + 1.5 < (len(y) / sr)):
                slices += 1

                idx_start = int((peak - 0.5)* sr)
                idx_end = int((peak + 1.5) * sr)
                y_slice = y[idx_start:idx_end]   

                mel = librosa.feature.melspectrogram(
                    y=y_slice, sr=sample_rate, n_fft=n_fft,
                    hop_length=hop_length, n_mels=n_mels
                )

                spectrogram_db = librosa.power_to_db(mel, ref=np.max)
                spectrogram_db = np.clip(spectrogram_db, -80, 0)

                spectrogram_image = ((spectrogram_db + 80) / 80 * 255).astype(np.uint8)
                spectrogram_image = np.flipud(spectrogram_image)

                image = Image.fromarray(spectrogram_image)
                image.save(output_folder / f"{filename.stem}_spectrogram_slice{slices}.png")
                successful += 1

    except Exception as error:
        print(f"Error processing {filename.name}: {error}")
        failed += 1

print("\nProcessing complete!")
print(f"Successfully converted: {successful}")
print(f"Failed: {failed}")
print(f"Output folder: {output_folder}")
