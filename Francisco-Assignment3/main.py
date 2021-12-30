import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Import songs dataset from the CSV file
print("Importing songs dataset...")
songs = pd.read_csv("songs.csv", low_memory=False)

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

# Recommender function that outputs the most similar songs to the given song
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the 10 most similar songs
    sim_scores = sim_scores[1:11]
    similar_song_indices = [i[0] for i in sim_scores]

    return songs[['title', 'singer']].iloc[similar_song_indices]

# print("Songs similar to oldtownroad")
# similar_to_old_town_road = get_recommendations("oldtownroad")
# print(similar_to_old_town_road)

# print("Songs similar to senorita")
# senorita = get_recommendations("senorita")
# print(senorita)