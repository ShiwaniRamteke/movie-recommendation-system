# movie_recommendation.py
# ------------------------------------------
# Movie Recommendation System using TF-IDF & Cosine Similarity
# ------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv("movies.csv")

# Check if required columns exist
if 'title' not in movies.columns or 'description' not in movies.columns:
    raise ValueError("CSV file must contain 'title' and 'description' columns.")

# Fill any missing descriptions with an empty string
movies['description'] = movies['description'].fillna('')

# Convert the text data into TF-IDF feature vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['description'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def recommend_movies(title, num_recommendations=5):
    # Convert to lowercase for case-insensitive search
    title = title.lower()
    movie_indices = movies[movies['title'].str.lower() == title].index

    if movie_indices.empty:
        print("❌ Movie not found. Please check the name and try again.")
        return

    idx = movie_indices[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    print(f"\n🎬 Movies similar to '{title.title()}':\n")
    for i, movie in enumerate(movies['title'].iloc[movie_indices], start=1):
        print(f"{i}. {movie}")

# Main program
if __name__ == "__main__":
    print("🎥 Welcome to the Movie Recommendation System!")
    while True:
        movie_name = input("\nEnter a movie title (or type 'exit' to quit): ").strip()
        if movie_name.lower() == 'exit':
            print("👋 Thanks for using the system. Goodbye!")
            break
        recommend_movies(movie_name)
