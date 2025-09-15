# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Mood Playlist Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HELPER FUNCTIONS ---

def get_spotify_client():
    """Initializes and returns a Spotify client."""
    try:
        client_id = st.secrets["9506dcd6a5844830a79100829ff6dad2"]
        client_secret = st.secrets["aa7f4a9e0aa1412f8dd04d5aaf9dbe8a"]
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception:
        st.error("Could not connect to Spotify. Please check your API credentials in secrets.toml.")
        return None

def analyze_emotion(image):
    """Analyzes the emotion from an image and returns the results."""
    try:
        # DeepFace.analyze returns a list of dictionaries, one for each face
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return analysis[0]  # Return analysis of the first (and likely only) face
    except Exception as e:
        return None

def plot_emotion_distribution(emotion_data):
    """Creates and displays a bar chart of emotion probabilities."""
    emotions = emotion_data['emotion']
    df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Percentage'])
    df = df.sort_values(by='Percentage', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Percentage', y='Emotion', data=df, palette='viridis', orient='h', ax=ax)
    
    ax.set_title('Emotion Analysis Breakdown', fontsize=16)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_ylabel('')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    st.pyplot(fig)

def get_playlist(spotify_client, mood):
    """Generates a playlist based on mood using specific audio features."""
    # Advanced mapping of moods to Spotify audio features
    mood_audio_features = {
        'happy':    {'target_valence': 0.9, 'target_energy': 0.8, 'seed_genres': ['happy', 'pop', 'dance']},
        'sad':      {'target_valence': 0.1, 'target_energy': 0.2, 'seed_genres': ['sad', 'acoustic', 'rainy-day']},
        'angry':    {'target_valence': 0.2, 'target_energy': 0.9, 'seed_genres': ['rock', 'metal', 'industrial']},
        'neutral':  {'target_valence': 0.5, 'target_energy': 0.5, 'seed_genres': ['chill', 'ambient', 'lo-fi']},
        'surprise': {'target_valence': 0.8, 'target_energy': 0.7, 'seed_genres': ['electronic', 'synth-pop']},
        'fear':     {'target_valence': 0.1, 'target_energy': 0.3, 'seed_genres': ['dark-ambient', 'soundtracks']},
        'disgust':  {'target_valence': 0.1, 'target_energy': 0.8, 'seed_genres': ['punk', 'grindcore']},
    }
    
    params = mood_audio_features.get(mood, mood_audio_features['neutral']) # Default to neutral
    
    try:
        results = spotify_client.recommendations(
            seed_genres=params['seed_genres'],
            limit=20,
            target_valence=params['target_valence'],
            target_energy=params['target_energy']
        )
        
        playlist = []
        for track in results['tracks']:
            playlist.append({
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "url": track['external_urls']['spotify'],
                "album_art": track['album']['images'][0]['url'],
                "embed_url": f"https://open.spotify.com/embed/track/{track['id']}"
            })
        return playlist
    except Exception as e:
        st.warning(f"Could not generate playlist: {e}")
        return []

# --- MAIN APP LAYOUT ---

# Initialize Spotify client
sp = get_spotify_client()

# Sidebar
with st.sidebar:
    st.header("About This App")
    st.write("""
    This **AI-powered application** brings your emotions to life through music. 
    1.  **Capture Your Moment:** Take a picture using your device's camera.
    2.  **AI Emotion Analysis:** A deep learning model analyzes your facial expression to determine your dominant mood.
    3.  **Custom Playlist:** The app curates a unique Spotify playlist tailored to the detected emotion using specific audio features.
    """)
    st.subheader("Technologies Used")
    st.info("Python | Streamlit | OpenCV | DeepFace (CNN) | Spotify API | Pandas | Matplotlib")

# Main content
st.title("🎵 AI Mood Playlist Generator")
st.markdown("---")

st.header("Step 1: Capture Your Mood")
img_file_buffer = st.camera_input("Take a picture to analyze your emotion", key="camera")

if img_file_buffer:
    # Convert buffer to a usable OpenCV image
    bytes_data = img_file_buffer.getvalue()
    cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.markdown("---")
    st.header("Step 2: AI Analysis Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(cv_image, channels="BGR", caption="Your Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing your emotion..."):
            emotion_analysis = analyze_emotion(cv_image)

        if emotion_analysis:
            dominant_emotion = emotion_analysis['dominant_emotion']
            st.success(f"**Dominant Mood Detected: {dominant_emotion.capitalize()}**")
            
            # Display the emotion distribution plot
            plot_emotion_distribution(emotion_analysis)
            
            st.markdown("---")
            st.header("Step 3: Your Personalized Playlist")
            
            if sp:
                with st.spinner(f"Curating a '{dominant_emotion}' playlist..."):
                    playlist = get_playlist(sp, dominant_emotion)
                
                if playlist:
                    # Display top track with embedded player
                    top_track = playlist[0]
                    st.subheader(f"Top Recommendation: {top_track['name']} by {top_track['artist']}")
                    st.components.v1.iframe(top_track['embed_url'], height=80)
                    
                    # Display the rest of the playlist
                    st.subheader("More Recommendations")
                    for i, track in enumerate(playlist[1:], 1):
                        st.write(f"{i}. **{track['name']}** by {track['artist']}")
                else:
                    st.warning("Couldn't generate a playlist. Try again.")
        else:
            st.error("Could not detect a face in the image. Please try taking another picture.")
