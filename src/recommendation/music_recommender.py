import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ⚠️ TEMPORARY (DO NOT PUSH TO GITHUB)
client_id = "Enter your client id"
client_secret = "Enter your client Secret"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

def recommend_music(emotion):
    query_map = {
        "happy": "happy upbeat",
        "sad": "sad acoustic",
        "angry": "rock intense",
        "neutral": "chill lo-fi",
        "surprise": "party",
        "fear": "calm relaxing",
        "disgust": "focus instrumental"
    }

    results = sp.search(q=query_map[emotion], limit=10, type='track')

    songs = []
    for item in results['tracks']['items']:
        name = item['name']
        artist = item['artists'][0]['name']
        url = item['external_urls']['spotify']
        image = item['album']['images'][0]['url'] if item['album']['images'] else ""

        songs.append({
            "name": name,
            "artist": artist,
            "url": url,
            "image": image
        })

    return songs