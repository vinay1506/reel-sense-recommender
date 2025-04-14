
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

class ContentBasedRecommender:
    def __init__(self):
        """Initialize the content-based recommender system"""
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
    
    def fit(self, movies_df, plot_descriptions=None):
        """
        Fit the content-based recommender with movie data
        
        Parameters:
        movies_df (DataFrame): Movies data with 'movieId', 'title', and 'genres' columns
        plot_descriptions (dict, optional): Dictionary mapping movieId to plot descriptions
        
        Returns:
        self: The fitted recommender
        """
        self.movies_df = movies_df.copy()
        
        # Create a feature soup: combine genres and descriptions if available
        print("Creating feature soup for content-based filtering...")
        
        # Convert genres from list to string
        self.movies_df['genres_str'] = self.movies_df['genres'].apply(lambda x: ' '.join(x))
        
        if plot_descriptions:
            # Add plot descriptions if available
            self.movies_df['description'] = self.movies_df['movieId'].map(
                lambda x: plot_descriptions.get(x, '')
            )
            self.movies_df['features'] = self.movies_df['genres_str'] + ' ' + self.movies_df['description']
        else:
            # Use only genres if no descriptions
            self.movies_df['features'] = self.movies_df['genres_str']
        
        # Create TF-IDF matrix
        print("Computing TF-IDF matrix...")
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['features'])
        
        # Compute cosine similarity matrix
        print("Computing cosine similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Mapping of movie titles to indices for quick lookup
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
        
        print("Content-based recommender fitted successfully!")
        return self
    
    def recommend(self, title, top_n=10):
        """
        Recommend similar movies based on content similarity
        
        Parameters:
        title (str): Title of the movie to get recommendations for
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Top N movie recommendations
        """
        # Get the index of the movie that matches the title
        if title not in self.indices:
            print(f"Movie '{title}' not found in the dataset.")
            similar_titles = self.movies_df['title'][self.movies_df['title'].str.contains(title, case=False)]
            if not similar_titles.empty:
                print(f"Did you mean one of these?: {', '.join(similar_titles.head().tolist())}")
            return pd.DataFrame()
        
        idx = self.indices[title]
        
        # Get the pairwise similarity scores with all movies
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top N most similar movies (excluding the input movie)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return top N similar movies with similarity scores
        recommendations = self.movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations[['title', 'genres', 'similarity_score']]
    
    def visualize_similarity_heatmap(self, titles, figsize=(10, 8)):
        """
        Create a heatmap of similarity between selected movies
        
        Parameters:
        titles (list): List of movie titles to include in the heatmap
        figsize (tuple): Figure size
        
        Returns:
        matplotlib.figure.Figure: The heatmap figure
        """
        # Filter for movies that exist in our dataset
        valid_titles = [title for title in titles if title in self.indices]
        
        if not valid_titles:
            print("None of the provided titles were found in the dataset.")
            return None
        
        # Get indices for the valid titles
        indices = [self.indices[title] for title in valid_titles]
        
        # Create a subset of the similarity matrix
        similarity_subset = self.cosine_sim[np.ix_(indices, indices)]
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            similarity_subset, 
            annot=True, 
            cmap="YlGnBu", 
            xticklabels=valid_titles, 
            yticklabels=valid_titles
        )
        plt.title('Content Similarity Between Selected Movies')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('results/visualizations/movie_similarity_heatmap.png')
        return plt.gcf()
    
    def get_feature_importance(self, title):
        """
        Get the most important features (terms) for a given movie
        
        Parameters:
        title (str): Movie title
        
        Returns:
        dict: Dictionary of feature importance scores
        """
        if title not in self.indices:
            print(f"Movie '{title}' not found in the dataset.")
            return {}
        
        idx = self.indices[title]
        movie_vector = self.tfidf_matrix[idx].toarray()[0]
        
        # Get the TF-IDF vocabulary
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit_transform(self.movies_df['features'])
        
        # Map feature indices to feature names
        feature_names = {i:word for word, i in tfidf.vocabulary_.items()}
        
        # Create a dictionary of feature importance
        importance = {}
        for i, score in enumerate(movie_vector):
            if score > 0:
                importance[feature_names.get(i, f'feature_{i}')] = score
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    # Example usage
    ratings_df, movies_df = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    cleaned_ratings, cleaned_movies = preprocess_data(ratings_df, movies_df)
    
    # Initialize and fit the recommender
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(cleaned_movies)
    
    # Get recommendations for a movie
    test_movie = cleaned_movies['title'].iloc[0]
    print(f"\nGetting recommendations for movie: {test_movie}")
    recommendations = content_recommender.recommend(test_movie, top_n=5)
    print(recommendations)
    
    # Visualize similarity between a few movies
    sample_movies = cleaned_movies['title'].sample(5).tolist()
    print(f"\nVisualizing similarity between: {', '.join(sample_movies)}")
    content_recommender.visualize_similarity_heatmap(sample_movies)
