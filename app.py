import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# --- Check if cv2 is imported ---
if 'cv2' not in globals():
    st.error("Error: OpenCV (cv2) library not loaded. Please ensure it's installed and imported correctly.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mood-Based Music Recommender",
    page_icon="🎵",
    layout="centered"
)

# --- SPOTIFY API AUTHENTICATION ---
# Use environment variables loaded from Colab secrets
try:
    CLIENT_ID = os.getenv("9506dcd6a5844830a79100829ff6dad2")
    CLIENT_SECRET = os.getenv("aa7f4a9e0aa1412f8dd04d5aaf9dbe8a")

    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("Spotify credentials not found. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in Colab secrets.")
        sp = None
    else:
        auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        sp = spotipy.Spotify(auth_manager=auth_manager)
except Exception as e:
    st.error(f"Error during Spotify authentication: {e}")
    sp = None

# --- HELPER FUNCTIONS ---
def get_recommendations(mood):
    """Generates music recommendations based on mood using Spotify API."""
    if not sp:
        return []

    # Advanced mapping of moods to Spotify audio features
    mood_features = {
        'happy':    {'seed_genres': ['happy', 'pop', 'dance'], 'target_valence': 0.9, 'target_energy': 0.8},
        'sad':      {'seed_genres': ['sad', 'acoustic', 'rainy-day'], 'target_valence': 0.1, 'target_energy': 0.2},
        'angry':    {'seed_genres': ['rock', 'metal', 'industrial'], 'target_valence': 0.2, 'target_energy': 0.9},
        'neutral':  {'seed_genres': ['chill', 'ambient', 'lo-fi'], 'target_valence': 0.5, 'target_energy': 0.5},
        'surprise': {'seed_genres': ['electronic', 'synth-pop'], 'target_valence': 0.8, 'target_energy': 0.7},
        'fear':     {'seed_genres': ['dark-ambient', 'soundtracks'], 'target_valence': 0.1, 'target_energy': 0.3},
        'disgust':  {'seed_genres': ['punk', 'grindcore'], 'target_valence': 0.1, 'target_energy': 0.8},
    }

    # Get parameters for the detected mood, default to neutral if not found
    params = mood_features.get(mood.lower(), mood_features['neutral'])

    try:
        recommendations = sp.recommendations(
            seed_genres=params['seed_genres'],
            limit=10,
            target_valence=params['target_valence'],
            target_energy=params['target_energy']
        )
        return recommendations['tracks']
    except Exception as e:
        st.error(f"Could not fetch recommendations: {e}")
        return []

# --- INITIALIZE SESSION STATE ---
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'detected_mood' not in st.session_state:
    st.session_state.detected_mood = None
if 'playlist' not in st.session_state:
    st.session_state.playlist = []

# --- STREAMLIT APP LAYOUT ---
st.title("🎵 Mood-Based Music Recommender")
st.write("Let AI find the perfect soundtrack for your current mood. Just capture your face!")

# Use st.camera_input for a cleaner, more integrated experience
captured_image_buffer = st.camera_input("Take a picture to begin...")

if captured_image_buffer:
    # Convert the image buffer to an OpenCV image
    bytes_data = captured_image_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Store the image in session state to persist it
    st.session_state.captured_image = cv2_img
    # Reset previous results
    st.session_state.detected_mood = None
    st.session_state.playlist = []

if st.session_state.captured_image is not None:
    st.markdown("---")
    st.subheader("Your Captured Image")

    # Display the captured image (converting BGR from OpenCV to RGB for display)
    st.image(st.session_state.captured_image[:, :, ::-1], use_column_width=True)

    # Analyze button to trigger emotion detection
    if st.button("Analyze My Mood"):
        with st.spinner("Detecting your emotion..."):
            try:
                # Analyze the emotion using DeepFace
                result = DeepFace.analyze(st.session_state.captured_image, actions=['emotion'], enforce_detection=False)
                st.session_state.detected_mood = result[0]['dominant_emotion']
            except Exception:
                st.error("Could not detect a face. Please try a different picture.")
                st.session_state.detected_mood = None

# Display mood and fetch playlist if mood has been detected
if st.session_state.detected_mood:
    st.success(f"Detected Mood: **{st.session_state.detected_mood.capitalize()}**")

    with st.spinner(f"Curating a playlist for a '{st.session_state.detected_mood}' mood..."):
        st.session_state.playlist = get_recommendations(st.session_state.detected_mood)

# Display the playlist if it has been generated
if st.session_state.playlist:
    st.markdown("---")
    st.subheader("Here's Your Custom Playlist")

    for i, track in enumerate(st.session_state.playlist):
        name = track['name']
        artist = track['artists'][0]['name']
        url = track['external_urls']['spotify']
        album_img = track['album']['images'] and track['album']['images'][0]['url'] or None


        col1, col2 = st.columns([1, 4])
        with col1:
            if album_img:
                st.image(album_img, width=80)
        with col2:
            st.markdown(f"**{i+1}. [{name}]({url})**")
            st.write(f"_{artist}_")
