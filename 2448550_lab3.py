"""
Streamlit Speech-to-Text App (With Feedback and Error Handling)

Features:
 - Record from microphone or upload audio
 - Transcribe with Whisper, Wav2Vec2, and Google SpeechRecognition
 - Provide stage-wise feedback and user-friendly error messages
"""

import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment

# Whisper
import whisper

# Wav2Vec2
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# SpeechRecognition (Google Web Speech API)
import speech_recognition as sr

# Streamlit mic_recorder
from streamlit_mic_recorder import mic_recorder

# --- Helper Functions ---
def ensure_wav_mono_16k(src_path: str) -> str:
    """
    Converts any audio file (wav, webm, ogg, mp3, etc.) to mono 16kHz WAV.
    """
    try:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(out.name, format="wav")
        return out.name
    except Exception:
        data, sr_orig = sf.read(src_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr_orig != 16000:
            data = librosa.resample(data.astype(float), orig_sr=sr_orig, target_sr=16000)
            sr_orig = 16000
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(out.name, data, sr_orig)
        return out.name


def transcribe_whisper(audio_path: str, model_size: str = "base") -> str:
    model = whisper.load_model(model_size)
    res = model.transcribe(audio_path)
    text = res.get("text", "").strip()
    if not text:
        raise ValueError("Could not understand audio. Please speak more clearly.")
    return text


def transcribe_wav2vec(audio_path: str, model_name: str = "facebook/wav2vec2-base-960h") -> str:
    proc = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    audio_input, sr = sf.read(audio_path)
    if audio_input.ndim > 1:
        audio_input = np.mean(audio_input, axis=1)
    if sr != 16000:
        audio_input = librosa.resample(audio_input.astype(float), orig_sr=sr, target_sr=16000)
        sr = 16000
    inputs = proc(audio_input, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = proc.batch_decode(predicted_ids)[0].strip().lower()
    if not transcription:
        raise ValueError("Could not understand audio. Please speak more clearly.")
    return transcription


def transcribe_speechrec(audio_path: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        if not text:
            raise ValueError("Could not understand audio. Please speak more clearly.")
        return text
    except sr.UnknownValueError:
        raise ValueError("Could not understand audio. Please try speaking more clearly.")
    except sr.RequestError:
        raise ConnectionError("Google API unavailable. Please check your internet connection.")


# --- Streamlit App ---
st.title("🎙️ Speech-to-Text App")

st.write("Upload an audio file or record live. The transcription will be displayed for Whisper, Wav2Vec2, and Google SpeechRecognition.")

# Feedback before recording
st.info("Speak something...")

# File Upload
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "flac", "ogg", "mp3", "webm"])

# Microphone Recording
audio = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹ Stop Recording", key="recorder")

source_path = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        source_path = tmp.name
elif audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio["bytes"])
        source_path = tmp.name

if source_path:
    audio_path = ensure_wav_mono_16k(source_path)
    st.audio(audio_path, format="audio/wav")

    if st.button("Transcribe"):
        try:
            st.info("Recognizing with Whisper...")
            whisper_text = transcribe_whisper(audio_path)
            st.success("Speech successfully converted to text with Whisper!")

            st.info("Recognizing with Wav2Vec2...")
            wav2vec_text = transcribe_wav2vec(audio_path)
            st.success("Speech successfully converted to text with Wav2Vec2!")

            st.info("Recognizing with Google SpeechRecognition...")
            google_text = transcribe_speechrec(audio_path)
            st.success("Speech successfully converted to text with Google API!")

            # Display results
            st.subheader("Transcription Results")
            st.write(f"**Whisper:** {whisper_text}")
            st.write(f"**Wav2Vec2:** {wav2vec_text}")
            st.write(f"**Google API:** {google_text}")

        except ValueError as ve:
            st.error(f"Recognition Error: {ve}")
        except ConnectionError as ce:
            st.error(f"Connection Error: {ce}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")
