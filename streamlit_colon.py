import streamlit as st
import librosa
import numpy as np
import joblib
import os
import sounddevice as sd
import wavio
import matplotlib.pyplot as plt
import gdown
# Load the trained model
MODEL_PATH = "./best_rf_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1xtdK73bVV2XOx9iXcVN2xKbwy82QeqNQ"  # Replace with your actual file ID

# Load the trained model only once
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model into memory (this happens once when the app starts)
model = joblib.load(MODEL_PATH)

# Function to extract features
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)

    # Pitch Features
    pitch_values = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean, pitch_std, pitch_range = np.mean(pitch_values), np.std(pitch_values), np.ptp(pitch_values)

    # Intensity Features
    rms_energy = librosa.feature.rms(y=y).flatten()
    intensity_mean, intensity_std, intensity_range = np.mean(rms_energy), np.std(rms_energy), np.ptp(rms_energy)

    # Speech Rate
    duration = librosa.get_duration(y=y, sr=sr)
    peaks = librosa.effects.split(y, top_db=30)
    speech_rate = len(peaks) / duration if duration > 0 else 0

    # Spectral Features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # MFCC Features (First 13 Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Final Feature Vector
    feature_vector = np.hstack([
        pitch_mean, pitch_std, pitch_range,
        intensity_mean, intensity_std, intensity_range,
        speech_rate, spectral_centroid, spectral_rolloff, zcr,
        mfccs_mean
    ])

    return feature_vector

# Function to predict emotion
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.array(features).reshape(1, -1)  # Ensure correct shape
    prediction = model.predict(features)[0]
    return prediction

# Streamlit UI
st.title("🎭 Real-Time Voice Emotion Detection")

# File Upload
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    emotion = predict_emotion(file_path)
    st.subheader(f"🎭 Predicted Emotion: **{emotion}**")

# Recording Audio
if st.button("🎙️ Record & Predict"):
    duration = 3  # Recording duration in seconds
    sr = 22050  # Sampling rate
    st.info("🎤 Recording... Speak now!")
    
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Recording complete!")

    # Save recorded audio
    file_path = "recorded_audio.wav"
    wavio.write(file_path, audio, sr, sampwidth=2)

    # Display waveform
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(audio, color='blue')
    ax.set_title("Recorded Audio Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Predict emotion
    emotion = predict_emotion(file_path)
    st.subheader(f"🎭 Predicted Emotion: **{emotion}**")
