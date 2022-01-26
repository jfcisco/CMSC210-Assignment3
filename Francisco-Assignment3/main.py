import pandas as pd
from numpy import mat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

print("Welcome to the Song Recommender Program!")
print("Please wait for the program to load.")

# Import songs dataset from the CSV file
print("Importing songs dataset...")
songs = pd.read_csv("songs.csv", low_memory=False)
# Dataset used: Billboard Hot 100 Songs from 1999 to 2019
# The include .csv file only contains a subset of features from the original dataset.
# I chose to include only the features required for this assignment
# Source: https://www.kaggle.com/danield2255/data-on-songs-from-billboard-19992019

# Copy original title, singer, and composer into a separate columns
# for retrieval and printing later on
songs['orig_title'] = songs['title'].fillna('Not Available')
songs['orig_singer'] = songs['singer'].fillna('Not Available')
songs['orig_composer'] = songs['composer'].fillna('Not Available')

# Pre-process and clean data
features = ['title', 'singer', 'composer', 'genre', 'lyrics']

for feature in features:
    songs[feature] = songs[feature].fillna('')

# Prepare soup feature by combining all features
def clean_data(x):
    return str.lower(x.replace(" ", "").replace("\n", ""))

for feature in features:
    songs[feature] = songs[feature].apply(clean_data)

def create_soup(song):
    return (song['title'] + ' ' + song['singer'] + ' ' + song['composer'] + ' ' 
    + song['genre'] + ' ' + song['lyrics'])

songs['soup'] = songs.apply(create_soup, axis=1)

# Prepare TF-IDF matrix from the song dataset
tfidf = TfidfVectorizer(stop_words='english')
print("Fitting TF-IDF Matrix...")
tfidf_matrix = tfidf.fit_transform(songs['soup'])

print("Computing cosine similarity matrix...")
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
csim_shape = cosine_sim.shape

# Make a reverse mapping of dataset indices and song names
songs = songs.reset_index()
indices = pd.Series(songs.index, index=songs['title']).drop_duplicates()  

def find_song(songs, title, singer=None, composer=None):
    # Create a vector that help look for the song that matches the given title, singer, composer
    match_vector = songs['orig_title'].str.contains(title, na=False, case=False)

    if singer is not None:
        match_vector = match_vector & songs['orig_singer'].str.contains(singer, na=False, case=False)

    if composer is not None:
        match_vector = match_vector & songs['orig_composer'].str.contains(composer, na=False, case=False)

    return songs[match_vector]

# Recommender function that outputs the most similar songs to the given song
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the 10 most similar songs
    sim_scores = sim_scores[1:11]
    similar_song_indices = [i[0] for i in sim_scores]

    return songs[['orig_title', 'orig_singer', 'orig_composer']].iloc[similar_song_indices]

# Interact with the user to get song title
title = input("Enter a song title: ")
singer = input("Enter the singer (optional): ")
composer = input("Enter the composer (optional): ")

matching_song = find_song(songs, title, 
    singer if singer != "" else None, 
    composer if composer != "" else None)

number_of_matching_songs = len(matching_song)
if number_of_matching_songs == 1:
    print("Found a matching song!")
    user_song = matching_song.iloc[0]
    print(f"Title: {user_song['orig_title']}") 
    print(f"Singer: {user_song['orig_singer']}")
    print(f"Composer: {user_song['orig_composer']}")

    recommendations = get_recommendations(matching_song['title'].values[0])
    print(f"Here are some songs similar to it:")
    print(recommendations)
elif number_of_matching_songs == 0:
    print("No matching songs found!")
else:
    print("Too many matching songs!")
    print("Please be more specific")

input("Press any key to exit...")