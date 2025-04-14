
import pandas as pd
import numpy as np
import os
import pickle
from data_preprocessing import load_data, preprocess_data
from content_based_filtering import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from evaluation import evaluate_rmse, precision_at_k
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine"""
        self.content_recommender = None
        self.collaborative_recommender = None
        self.movies_df = None
        self.ratings_df = None
        self.model_path = "models"
        os.makedirs(self.model_path, exist_ok=True)
    
    def load_data(self, ratings_path, movies_path):
        """Load movie and rating data"""
        self.ratings_df, self.movies_df = load_data(ratings_path, movies_path)
        self.ratings_df, self.movies_df = preprocess_data(self.ratings_df, self.movies_df)
        return self.ratings_df, self.movies_df
    
    def train_models(self, n_factors=50, collaborative_method='svd'):
        """Train recommendation models"""
        if self.movies_df is None or self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Training content-based filtering model...")
        self.content_recommender = ContentBasedRecommender()
        self.content_recommender.fit(self.movies_df)
        
        print(f"Training collaborative filtering model using {collaborative_method}...")
        self.collaborative_recommender = CollaborativeFilteringRecommender(method=collaborative_method)
        self.collaborative_recommender.fit(self.ratings_df, self.movies_df, n_factors=n_factors)
        
        print("Models trained successfully!")
        return self
    
    def save_models(self):
        """Save trained models to disk"""
        if self.content_recommender is None or self.collaborative_recommender is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print("Saving models to disk...")
        with open(f"{self.model_path}/content_recommender.pkl", "wb") as f:
            pickle.dump(self.content_recommender, f)
        
        with open(f"{self.model_path}/collaborative_recommender.pkl", "wb") as f:
            pickle.dump(self.collaborative_recommender, f)
        
        print(f"Models saved to {self.model_path}!")
        return self
    
    def load_models(self):
        """Load trained models from disk"""
        print("Loading models from disk...")
        
        try:
            with open(f"{self.model_path}/content_recommender.pkl", "rb") as f:
                self.content_recommender = pickle.load(f)
            
            with open(f"{self.model_path}/collaborative_recommender.pkl", "rb") as f:
                self.collaborative_recommender = pickle.load(f)
            
            print("Models loaded successfully!")
        except FileNotFoundError:
            print("Model files not found. Train and save models first.")
            return None
        
        return self
    
    def get_movie_recommendations(self, movie_title, top_n=10):
        """
        Get movie recommendations based on a given movie title
        
        Parameters:
        movie_title (str): Title of the movie to base recommendations on
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Recommended movies
        """
        if self.content_recommender is None:
            raise ValueError("Content-based recommender not trained or loaded.")
        
        # Get content-based recommendations
        content_recommendations = self.content_recommender.recommend(movie_title, top_n=top_n)
        
        if content_recommendations.empty:
            print(f"Movie '{movie_title}' not found or no recommendations available.")
            return None
        
        return content_recommendations
    
    def get_user_recommendations(self, user_id, top_n=10):
        """
        Get user recommendations based on a given user ID
        
        Parameters:
        user_id (int): User ID to get recommendations for
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Recommended movies
        """
        if self.collaborative_recommender is None:
            raise ValueError("Collaborative recommender not trained or loaded.")
        
        # Check if user exists in the dataset
        if user_id not in self.collaborative_recommender.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset.")
            return None
        
        # Get collaborative filtering recommendations
        cf_recommendations = self.collaborative_recommender.recommend_for_user(user_id, top_n=top_n)
        
        if cf_recommendations.empty:
            print(f"No recommendations available for user {user_id}.")
            return None
        
        return cf_recommendations
    
    def get_hybrid_recommendations(self, user_id, movie_title=None, content_weight=0.3, top_n=10):
        """
        Get hybrid recommendations combining collaborative and content-based filtering
        
        Parameters:
        user_id (int): User ID to get recommendations for
        movie_title (str, optional): Movie title to enhance recommendations
        content_weight (float): Weight for content-based recommendations (0-1)
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Recommended movies
        """
        if self.content_recommender is None or self.collaborative_recommender is None:
            raise ValueError("Both recommenders must be trained or loaded.")
        
        # Check if user exists
        if user_id not in self.collaborative_recommender.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset.")
            return None
        
        # Get collaborative filtering scores (predicted ratings)
        cf_recommendations = self.collaborative_recommender.recommend_for_user(user_id, top_n=top_n*2)
        
        if cf_recommendations.empty:
            print(f"No collaborative recommendations available for user {user_id}.")
            if movie_title:
                # Fall back to content-based only
                return self.get_movie_recommendations(movie_title, top_n)
            return None
        
        # Create a dictionary of movie_id -> predicted_rating
        cf_scores = dict(zip(cf_recommendations['movieId'], cf_recommendations['predicted_rating']))
        
        # Normalize CF scores to [0,1] range
        max_cf_score = max(cf_scores.values())
        min_cf_score = min(cf_scores.values())
        range_cf = max_cf_score - min_cf_score
        
        if range_cf > 0:
            for movie_id in cf_scores:
                cf_scores[movie_id] = (cf_scores[movie_id] - min_cf_score) / range_cf
        
        # If a movie title is provided, get content-based recommendations
        if movie_title:
            content_recommendations = self.content_recommender.recommend(movie_title, top_n=top_n*2)
            
            if not content_recommendations.empty:
                # Get movie IDs from recommendations
                movie_ids = self.movies_df[self.movies_df['title'].isin(content_recommendations['title'])]['movieId']
                
                # Create a dictionary of movie_id -> similarity_score
                content_scores = dict(zip(
                    movie_ids, 
                    content_recommendations['similarity_score']
                ))
                
                # Combine the scores
                combined_scores = {}
                all_movie_ids = set(cf_scores.keys()) | set(content_scores.keys())
                
                for movie_id in all_movie_ids:
                    cf_score = cf_scores.get(movie_id, 0)
                    content_score = content_scores.get(movie_id, 0)
                    
                    # Weighted combination
                    combined_scores[movie_id] = (1 - content_weight) * cf_score + content_weight * content_score
                
                # Sort movies by combined score
                sorted_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                top_movie_ids = [movie_id for movie_id, _ in sorted_movies[:top_n]]
                
                # Get movie details
                hybrid_recommendations = self.movies_df[self.movies_df['movieId'].isin(top_movie_ids)].copy()
                
                # Add scores to the results
                hybrid_recommendations['score'] = hybrid_recommendations['movieId'].apply(
                    lambda x: combined_scores.get(x, 0)
                )
                
                return hybrid_recommendations.sort_values('score', ascending=False)
        
        # If no movie title or content recommendations, return collaborative filtering results
        return cf_recommendations.head(top_n)
    
    def visualize_movie_ratings_by_genre(self):
        """Visualize average movie ratings by genre"""
        if self.movies_df is None or self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a movie_id to average rating mapping
        movie_ratings = self.ratings_df.groupby('movieId')['rating'].mean()
        
        # Create a genre to ratings mapping
        genre_ratings = {}
        
        for _, movie in self.movies_df.iterrows():
            movie_id = movie['movieId']
            if movie_id in movie_ratings.index:
                avg_rating = movie_ratings[movie_id]
                
                for genre in movie['genres']:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                    
                    genre_ratings[genre].append(avg_rating)
        
        # Calculate average rating per genre
        genre_avg_ratings = {
            genre: np.mean(ratings) for genre, ratings in genre_ratings.items() 
            if len(ratings) >= 10  # Only include genres with enough movies
        }
        
        # Sort genres by average rating
        sorted_genres = sorted(genre_avg_ratings.items(), key=lambda x: x[1], reverse=True)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x=[g[0] for g in sorted_genres], 
            y=[g[1] for g in sorted_genres],
            palette="viridis"
        )
        plt.title('Average Movie Ratings by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        os.makedirs('results/visualizations', exist_ok=True)
        plt.savefig('results/visualizations/genre_ratings.png')
        return plt.gcf()
    
    def visualize_rating_distribution_by_year(self):
        """Visualize rating distribution by year"""
        if self.movies_df is None or self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Merge ratings with movies to get year information
        merged_df = pd.merge(self.ratings_df, self.movies_df[['movieId', 'year']], on='movieId')
        
        # Filter out rows with missing years
        merged_df = merged_df.dropna(subset=['year'])
        
        # Convert year to numeric and filter reasonable range
        merged_df['year'] = pd.to_numeric(merged_df['year'], errors='coerce')
        merged_df = merged_df[(merged_df['year'] >= 1950) & (merged_df['year'] <= 2023)]
        
        # Group by year and calculate statistics
        year_stats = merged_df.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
        
        # Only include years with enough ratings
        year_stats = year_stats[year_stats['count'] >= 50]
        
        # Create visualization
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # Plot average rating by year
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Rating', color=color)
        ax1.plot(year_stats['year'], year_stats['mean'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for count
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Number of Ratings', color=color)
        ax2.bar(year_stats['year'], year_stats['count'], color=color, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Average Movie Ratings by Year')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        os.makedirs('results/visualizations', exist_ok=True)
        plt.savefig('results/visualizations/ratings_by_year.png')
        return fig
    
    def visualize_user_activity(self):
        """Visualize user activity distribution"""
        if self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Count ratings per user
        user_activity = self.ratings_df.groupby('userId').size().reset_index(name='rating_count')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot histogram of user activity
        sns.histplot(user_activity['rating_count'], bins=50, kde=True)
        plt.title('Distribution of User Activity')
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Number of Users')
        plt.xscale('log')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs('results/visualizations', exist_ok=True)
        plt.savefig('results/visualizations/user_activity.png')
        return plt.gcf()
    
    def get_popular_movies(self, n=10, min_ratings=100):
        """
        Get popular movies based on number of ratings and average rating
        
        Parameters:
        n (int): Number of movies to return
        min_ratings (int): Minimum number of ratings required
        
        Returns:
        DataFrame: Popular movies
        """
        if self.movies_df is None or self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Group by movieId and calculate stats
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        # Flatten the column names
        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Filter movies with enough ratings
        movie_stats = movie_stats[movie_stats['rating_count'] >= min_ratings]
        
        # Sort by average rating
        popular_movies = movie_stats.sort_values(['avg_rating', 'rating_count'], ascending=False).head(n)
        
        # Merge with movie details
        popular_movies = popular_movies.merge(self.movies_df[['movieId', 'title', 'genres']], on='movieId')
        
        return popular_movies

if __name__ == "__main__":
    # Example usage
    engine = RecommendationEngine()
    
    # Load data
    print("\nLoading MovieLens data...")
    ratings_df, movies_df = engine.load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    
    # Train models
    print("\nTraining recommendation models...")
    engine.train_models(n_factors=50)
    
    # Save models
    print("\nSaving models to disk...")
    engine.save_models()
    
    # Get recommendations for a movie
    test_movie = "Toy Story"
    print(f"\nGetting recommendations for movie: {test_movie}")
    movie_recommendations = engine.get_movie_recommendations(test_movie, top_n=5)
    if movie_recommendations is not None:
        print(movie_recommendations[['title', 'similarity_score']])
    
    # Get recommendations for a user
    test_user = ratings_df['userId'].iloc[0]
    print(f"\nGetting recommendations for user: {test_user}")
    user_recommendations = engine.get_user_recommendations(test_user, top_n=5)
    if user_recommendations is not None:
        print(user_recommendations[['title', 'predicted_rating']])
    
    # Get hybrid recommendations
    print(f"\nGetting hybrid recommendations for user {test_user} and movie {test_movie}")
    hybrid_recommendations = engine.get_hybrid_recommendations(test_user, test_movie, top_n=5)
    if hybrid_recommendations is not None:
        print(hybrid_recommendations[['title', 'score']])
    
    # Visualize movie ratings by genre
    print("\nVisualizing movie ratings by genre...")
    engine.visualize_movie_ratings_by_genre()
    
    # Visualize rating distribution by year
    print("\nVisualizing rating distribution by year...")
    engine.visualize_rating_distribution_by_year()
    
    # Visualize user activity
    print("\nVisualizing user activity...")
    engine.visualize_user_activity()
    
    print("\nRecommendation engine demo completed!")
