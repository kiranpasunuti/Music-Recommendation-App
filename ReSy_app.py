!pip install scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build


# In[33]:


def create_youtube_api_outputs(playlist_name, df):
    youtube = build('youtube', 'v3', developerKey=api_key)
    search_response = youtube.search().list(
        q=playlist_name,
        part='id,snippet',
        type='video',
        maxResults=1 
    ).execute()

    video_details = []
    for item in search_response['items']:
        video_details.append({
            'title': item['snippet']['title'],
            'videoId': item['id']['videoId'],
            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })

    youtube_playlist = pd.DataFrame(video_details)

    # Create 'combined_features' column for youtube_playlist
    youtube_playlist['combined_features'] = youtube_playlist['title'].astype(str) + ' Unknown'

    # Create 'combined_features' column for df
    df['combined_features'] = df[['track_name', 'artists', 'popularity', 'track_genre']].astype(str).agg(' '.join, axis=1)
    
    df = df[~df['combined_features'].duplicated(keep='first')]  # to eliminate duplicates

    return youtube_playlist



# In[34]:


def recommendation_engine(playlist_name, df, num_recommendations=10):
    youtube_playlist = create_youtube_api_outputs(playlist_name, df)
    
    if youtube_playlist.empty:
        print("No matching YouTube videos found for the given playlist.")
        return
    df['combined_features'] = df[['track_name', 'artists', 'popularity', 'track_genre']].astype(str).agg(' '.join, axis=1)
    youtube_playlist['combined_features'] = youtube_playlist['title'].astype(str) + ' Unknown'  
    
    df = df[~df['combined_features'].duplicated(keep='first')]  #to eliminate duplicates
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'].tolist() + youtube_playlist['combined_features'].tolist())
    cosine_sim = cosine_similarity(tfidf_matrix[-len(youtube_playlist):], tfidf_matrix[:-len(youtube_playlist)])

    for i in range(len(youtube_playlist)):
        track_indices = np.argsort(cosine_sim[i])[::-1][:num_recommendations]
        unique_tracks = set()

        print(f"\nTop recommendations for {youtube_playlist['title'].iloc[i]}:\n")
        for index, track_index in enumerate(track_indices):

            if track_index < len(df):                       #  #Verifing if the track_index is within the bounds of the DataFrame df.
                track = df.iloc[track_index]                ##Retrieves the track information from the DataFrame using the index.

                if track['track_name'] not in unique_tracks:
                    youtube_url = youtube_playlist['url'].iloc[i] if i < len(youtube_playlist) else "No YouTube URL available"
                    image_url = get_spotify_image_url(track['track_name'], track['artists'])
                    print(f"Track: {track['track_name']}\nPopularity: {track['popularity']}\nArtist: {track['artists']}\nGenre: {track['track_genre']}\nYouTube URL: {youtube_url}\nImage URL: {image_url}\n")
                    unique_tracks.add(track['track_name'])

            else:
                print(f"Index {track_index} is out of bounds for the DataFrame.")
                
           

import requests
def get_spotify_image_url(track_name, artist_name):
    SPOTIFY_CLIENT_ID = '1eadc3aa756843018ebe4ab96af1f149'
    SPOTIFY_CLIENT_SECRET = '2c6a61f2ebd74874b005cad8083abf67'
    access_token = get_spotify_access_token(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    search_url = f'https://api.spotify.com/v1/search?q={track_name} {artist_name}&type=track'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(search_url, headers=headers)
    result = response.json()
    
    if 'tracks' in result and 'items' in result['tracks'] and result['tracks']['items']:
        return result['tracks']['items'][0]['album']['images'][0]['url']    # extracted the image URL from the spotify API response
    else:
        return "No image available"


def get_spotify_access_token(client_id, client_secret):
    token_url = 'https://accounts.spotify.com/api/token'
    auth = (client_id, client_secret)
    data = {'grant_type': 'client_credentials'}
    response = requests.post(token_url, auth=auth, data=data)
    result = response.json()
    return result.get('access_token', '')
    # In[35]:


df = pd.read_excel('Book1.xlsx')
api_key = 'AIzaSyCqchacThqojFg_9QjVu3tpADud46os85M'


# In[36]:


# Define your Streamlit app
def recommendation_app():
    st.title("Music Recommendation App")

    # Get user input for the playlist name
    playlist_name = st.text_input("Enter the playlist name:")

    # Get recommendations only if the playlist name is provided
    if playlist_name:
        youtube_playlist = create_youtube_api_outputs(playlist_name, df)

        if youtube_playlist.empty:
            st.warning("No matching YouTube videos found for the given playlist.")
        else:
            st.header("Top Recommendations:")
            display_recommendations(youtube_playlist)


# In[37]:


# Helper function to display recommendations
def display_recommendations(youtube_playlist):
    num_recommendations = 10  # You can adjust the number of recommendations to display

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'].tolist() + youtube_playlist['combined_features'].tolist())
    cosine_sim = cosine_similarity(tfidf_matrix[-len(youtube_playlist):], tfidf_matrix[:-len(youtube_playlist)])

   

    # Display recommendations in a horizontal layout
    for i in range(len(youtube_playlist)):
        track_indices = np.argsort(cosine_sim[i])[::-1][:num_recommendations]
        unique_tracks = set()

        # Create rows with four recommendations each
        for index, track_index in enumerate(track_indices):
            if track_index < len(df):
                track = df.iloc[track_index]
                if track['track_name'] not in unique_tracks:
                    youtube_url = youtube_playlist['url'].iloc[i] if i < len(youtube_playlist) else "No YouTube URL available"
                    image_url = get_spotify_image_url(track['track_name'], track['artists'])

                    # Display recommendations horizontally with four per line
                    #col_num = index % recommendations_per_line

                    # Add a container div with custom CSS to create a horizontal layout
                    st.markdown(
                        f"""
                        <div style="display: inline-block; margin: 10px;">
                            <img src="{image_url}" width="200">
                            <p>Track: {track['track_name']}</p>
                            <p>Popularity: {track['popularity']}</p>
                            <p>Artist: {track['artists']}</p>
                            <p>Genre: {track['track_genre']}</p>
                            Youtube URL: <a href="{youtube_url}" target="_blank">{youtube_url}</a> 



                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    unique_tracks.add(track['track_name'])

                    







# In[38]:


# Run the Streamlit app
if __name__ == '__main__':
    recommendation_app()


# In[ ]:




