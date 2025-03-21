import librosa
import numpy as np
import joblib
import streamlit as st
from io import BytesIO
import os
import gdown

# Constants
MODEL_PATH = "./best_rf_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1xtdK73bVV2XOx9iXcVN2xKbwy82QeqNQ"  # Replace with your actual file ID

# Load the trained model only once
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model into memory (this happens once when the app starts)
model = joblib.load(MODEL_PATH)

def extract_features(audio, sr):
    """Extract features from audio."""
    pitch_values = librosa.yin(audio, fmin=50, fmax=300)
    pitch_mean, pitch_std, pitch_range = np.mean(pitch_values), np.std(pitch_values), np.ptp(pitch_values)
    
    rms_energy = librosa.feature.rms(y=audio).flatten()
    intensity_mean, intensity_std, intensity_range = np.mean(rms_energy), np.std(rms_energy), np.ptp(rms_energy)
    
    duration = librosa.get_duration(y=audio, sr=sr)
    peaks = librosa.effects.split(audio, top_db=30)
    speech_rate = len(peaks) / duration if duration > 0 else 0
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    feature_vector = np.hstack([ 
        pitch_mean, pitch_std, pitch_range, 
        intensity_mean, intensity_std, intensity_range, 
        speech_rate, spectral_centroid, spectral_rolloff, zcr, 
        mfccs_mean 
    ])
    return feature_vector

def predict_emotion(audio, sr):
    """Predict emotion from extracted features."""
    features = extract_features(audio, sr)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

# Streamlit app
st.title("Emotion Prediction from Audio")

# Upload file section
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    try:
        audio_bytes = uploaded_file.read()
        audio_buffer = BytesIO(audio_bytes)

        # Load audio
        audio, sr = librosa.load(audio_buffer, sr=22050)

        # Display the audio file and message
        st.audio(uploaded_file, format="audio/wav")
        st.write("Processing your audio...")

        # Run prediction and display results
        mood = predict_emotion(audio, sr)
        st.write(f"Predicted Mood: {mood}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
