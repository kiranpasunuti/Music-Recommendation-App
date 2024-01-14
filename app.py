
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
import requests

# Function to create YouTube API outputs
def create_youtube_api_outputs(playlist_name, df):
    # Your existing code for YouTube API integration

# Function to display recommendations
def display_recommendations(youtube_playlist):
    num_recommendations = 10

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'].tolist() + youtube_playlist['combined_features'].tolist())
    cosine_sim = cosine_similarity(tfidf_matrix[-len(youtube_playlist):], tfidf_matrix[:-len(youtube_playlist)])

    # Spotify-like interface
    st.header("Top Recommendations:")

    for i in range(len(youtube_playlist)):
        track_indices = np.argsort(cosine_sim[i])[::-1][:num_recommendations]
        unique_tracks = set()

        for index, track_index in enumerate(track_indices):
            if track_index < len(df):
                track = df.iloc[track_index]
                if track['track_name'] not in unique_tracks:
                    youtube_url = youtube_playlist['url'].iloc[i] if i < len(youtube_playlist) else "No YouTube URL available"
                    image_url = get_spotify_image_url(track['track_name'], track['artists'])

                    # Display recommendations with Spotify-like interface
                    st.image(image_url, caption=f"Track: {track['track_name']}")
                    st.write(f"Popularity: {track['popularity']}")
                    st.write(f"Artist: {track['artists']}")
                    st.write(f"Genre: {track['track_genre']}")
                    st.write(f"YouTube URL: [{youtube_url}]({youtube_url})")

                    unique_tracks.add(track['track_name'])

# Streamlit app
df = pd.read_excel('Book1.xlsx')
api_key = 'AIzaSyCqchacThqojFg_9QjVu3tpADud46os85M'

st.title("Music Recommendation App")

# Get user input for the playlist name
playlist_name = st.text_input("Enter the playlist name:")

# Get recommendations only if the playlist name is provided
if playlist_name:
    youtube_playlist = create_youtube_api_outputs(playlist_name, df)

    if youtube_playlist.empty:
        st.warning("No matching YouTube videos found for the given playlist.")
    else:
        display_recommendations(youtube_playlist)
