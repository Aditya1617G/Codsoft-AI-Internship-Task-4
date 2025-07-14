import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

# Sample movie dataset
movies = pd.DataFrame({
    "title": [
        "The Matrix", "John Wick", "Toy Story", "Finding Nemo",
        "Inception", "The Godfather", "Avengers: Endgame",
        "Interstellar", "Shrek", "The Dark Knight"
    ],
    "genre": [
        "Action Sci-Fi", "Action Thriller", "Animation Comedy",
        "Animation Adventure", "Action Sci-Fi", "Crime Drama",
        "Action Sci-Fi Superhero", "Sci-Fi Drama", "Animation Comedy",
        "Action Crime Thriller"
    ]
})

# Preprocess: lowercase titles
movies["title_lower"] = movies["title"].str.strip().str.lower()

# TF-IDF vectorization on genres
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
genre_matrix = vectorizer.fit_transform(movies["genre"])
similarity = cosine_similarity(genre_matrix)

# Recommendation function with fuzzy matching
def recommend(movie_title, movies, similarity_matrix, top_n=3, min_score=70):
    movie_title = movie_title.strip().lower()

    # Fuzzy match title
    match_result = process.extractOne(
        movie_title,
        movies["title_lower"],
        scorer=fuzz.token_sort_ratio
    )

    if match_result is None or match_result[1] < min_score:
        print(f"\nNo good match found for '{movie_title}'.")
        print("Try one of these movie titles:")
        for title in movies["title"].sample(5).values:
            print(" -", title)
        return None

    matched_title, score, idx = match_result
    actual_title = movies.iloc[idx]["title"]
    print(f"\nClosest match: '{actual_title}' (Confidence: {score}%)")

    # Get similarity scores and recommend
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movies.iloc[i[0]]["title"] for i in sim_scores[1:top_n + 1]]

    return actual_title, top_movies

# Get user input
user_input = input("Enter a movie title: ").strip()

if not user_input:
    print("No input provided. Please try again.")
else:
    result = recommend(user_input, movies, similarity)

    if result:
        matched_title, recommendations = result
        print(f"\nBecause you watched '{matched_title}', you might also like:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\nNo recommendations found.")

# Safe exit
try:
    input("\nPress Enter to exit...")
except EOFError:
    pass
