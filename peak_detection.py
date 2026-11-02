import numpy as np
import librosa
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

audio_path = "/Users/jacksonnovotny/Desktop/Machine Learning Final Project/Spectogram Generators/birdSoundsClean/Acorn Woodpecker 01 Calls US-CA.mp3"
y, sr = librosa.load(audio_path, sr=None)

amplitude_envelope = np.abs(y)

peaks, _ = find_peaks(amplitude_envelope, height=0.3, distance=sr*0.5)
print(peaks)


peak_times = peaks / sr

print(f"Detected {len(peaks)} peaks.")
print("Peak timestamps (seconds):", peak_times)

time = np.linspace(0, len(y) / sr, num=len(y))
plt.figure(figsize=(10, 4))
plt.plot(time, amplitude_envelope, label="Audio Signal Envelope")
plt.plot(time[peaks], amplitude_envelope[peaks], "x", color="red", label="Detected Peaks")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
