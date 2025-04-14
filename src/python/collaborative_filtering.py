
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt

class CollaborativeFilteringRecommender:
    def __init__(self, method='svd'):
        """
        Initialize the collaborative filtering recommender system
        
        Parameters:
        method (str): Method to use for collaborative filtering ('svd' or 'cosine')
        """
        self.method = method
        self.user_item_matrix = None
        self.user_item_matrix_normalized = None
        self.movie_features = None
        self.user_features = None
        self.sigma = None
        self.predicted_ratings = None
        self.movies_df = None
        self.ratings_df = None
        self.mean_user_rating = None
        self.movie_id_mapping = None
        self.movie_to_idx = None
        self.idx_to_movie = None
        
    def fit(self, ratings_df, movies_df, n_factors=50):
        """
        Fit the collaborative filtering model
        
        Parameters:
        ratings_df (DataFrame): Ratings data with userId, movieId, rating columns
        movies_df (DataFrame): Movies data with movieId, title columns
        n_factors (int): Number of latent factors for SVD
        
        Returns:
        self: The fitted recommender
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        print("Creating user-item matrix...")
        # Create the user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Create mapping between movie IDs and matrix indices
        self.movie_id_mapping = {
            idx: movie_id for idx, movie_id in enumerate(self.user_item_matrix.columns)
        }
        self.movie_to_idx = {
            movie_id: idx for idx, movie_id in self.movie_id_mapping.items()
        }
        self.idx_to_movie = {
            idx: movie_id for movie_id, idx in self.movie_to_idx.items()
        }
        
        if self.method == 'svd':
            # Normalize by subtracting the mean rating for each user
            print("Normalizing ratings...")
            self.user_item_matrix_normalized = self.user_item_matrix.copy()
            self.mean_user_rating = self.user_item_matrix.mean(axis=1)
            
            for user_idx, user_id in enumerate(self.user_item_matrix.index):
                self.user_item_matrix_normalized.loc[user_id] -= self.mean_user_rating[user_id]
            
            # Get sparse matrix
            matrix = self.user_item_matrix_normalized.to_numpy()
            
            # Perform SVD
            print(f"Performing SVD with {n_factors} factors...")
            U, sigma, Vt = svds(matrix, k=n_factors)
            
            # Convert sigma to diagonal matrix
            sigma = np.diag(sigma)
            
            # Save the decomposition results
            self.user_features = U
            self.sigma = sigma
            self.movie_features = Vt
            
            # Reconstruct the ratings matrix
            print("Reconstructing ratings matrix...")
            predicted_ratings = np.dot(np.dot(U, sigma), Vt) + np.array(self.mean_user_rating)[:, np.newaxis]
            self.predicted_ratings = pd.DataFrame(
                predicted_ratings, 
                index=self.user_item_matrix.index, 
                columns=self.user_item_matrix.columns
            )
            
        elif self.method == 'cosine':
            # Calculate item-item similarity using cosine similarity
            print("Computing item-item cosine similarity...")
            self.item_similarity = cosine_similarity(self.user_item_matrix.T)
            self.item_similarity = pd.DataFrame(
                self.item_similarity,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        
        print("Collaborative filtering model fitted successfully!")
        return self
    
    def recommend_for_user(self, user_id, top_n=10, exclude_rated=True):
        """
        Recommend movies for a specific user
        
        Parameters:
        user_id (int): User ID
        top_n (int): Number of recommendations to return
        exclude_rated (bool): Whether to exclude already rated movies
        
        Returns:
        DataFrame: Top N movie recommendations for the user
        """
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset.")
            return pd.DataFrame()
        
        if self.method == 'svd':
            # Get predicted ratings for the user
            user_ratings = self.predicted_ratings.loc[user_id]
        
        elif self.method == 'cosine':
            # Get the user's ratings
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # Create weighted sum of similar items for unrated items
            user_ratings_filled = user_ratings.copy()
            
            for item in self.user_item_matrix.columns:
                if user_ratings[item] == 0:  # If item is unrated
                    # Get similarity scores for this item with all other items
                    sim_scores = self.item_similarity[item]
                    
                    # Get user's ratings for items similar to this one
                    rated_items_sim = user_ratings * (sim_scores > 0)
                    
                    # Calculate weighted average rating
                    if (sim_scores > 0).sum() > 0 and rated_items_sim.sum() > 0:
                        user_ratings_filled[item] = (rated_items_sim * sim_scores).sum() / sim_scores[rated_items_sim > 0].sum()
            
            user_ratings = user_ratings_filled
            
        # Get the user's rated movies if we want to exclude them
        if exclude_rated:
            rated_movies = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].tolist()
            # Filter out already rated movies
            user_ratings = user_ratings[~user_ratings.index.isin(rated_movies)]
        
        # Sort and get top N recommendations
        top_movie_indices = user_ratings.sort_values(ascending=False).head(top_n).index
        
        # Map movie IDs back to movie details
        recommended_movies = self.movies_df[self.movies_df['movieId'].isin(top_movie_indices)].copy()
        
        # Add predicted ratings to the result
        recommended_movies['predicted_rating'] = recommended_movies['movieId'].apply(
            lambda x: user_ratings[x] if x in user_ratings.index else None
        )
        
        return recommended_movies.sort_values('predicted_rating', ascending=False)
    
    def recommend_similar_movies(self, movie_id, top_n=10):
        """
        Recommend similar movies based on collaborative filtering
        
        Parameters:
        movie_id (int): Movie ID to find similar movies for
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Top N similar movies
        """
        if self.method != 'cosine':
            print("This method is only available for cosine similarity-based collaborative filtering.")
            return pd.DataFrame()
        
        if movie_id not in self.item_similarity.index:
            print(f"Movie {movie_id} not found in the dataset.")
            return pd.DataFrame()
        
        # Get similarity scores for this movie with all other movies
        movie_similarities = self.item_similarity[movie_id]
        
        # Sort and get top N similar movies
        similar_movie_ids = movie_similarities.sort_values(ascending=False).head(top_n+1).index.tolist()
        
        # Remove the input movie itself (it will be the most similar to itself)
        if movie_id in similar_movie_ids:
            similar_movie_ids.remove(movie_id)
        else:
            similar_movie_ids = similar_movie_ids[:-1]
        
        # Get movie details
        similar_movies = self.movies_df[self.movies_df['movieId'].isin(similar_movie_ids)].copy()
        
        # Add similarity scores
        similar_movies['similarity_score'] = similar_movies['movieId'].apply(
            lambda x: movie_similarities[x]
        )
        
        return similar_movies.sort_values('similarity_score', ascending=False)
    
    def evaluate(self, test_ratings):
        """
        Evaluate the recommender system using RMSE
        
        Parameters:
        test_ratings (DataFrame): Test ratings with userId, movieId, rating columns
        
        Returns:
        float: Root Mean Squared Error
        """
        test_user_item_pairs = list(zip(test_ratings['userId'], test_ratings['movieId']))
        actual_ratings = test_ratings['rating'].values
        
        # Get predicted ratings for test pairs
        predicted_ratings = []
        
        for user_id, movie_id in test_user_item_pairs:
            if user_id in self.predicted_ratings.index and movie_id in self.predicted_ratings.columns:
                predicted_ratings.append(self.predicted_ratings.loc[user_id, movie_id])
            else:
                # If user or movie not in training data, use the global mean
                predicted_ratings.append(self.ratings_df['rating'].mean())
        
        # Calculate RMSE
        rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        return rmse
    
    def precision_at_k(self, test_ratings, k=10, threshold=3.5):
        """
        Calculate Precision@K for the recommender system
        
        Parameters:
        test_ratings (DataFrame): Test ratings with userId, movieId, rating columns
        k (int): Number of recommendations to consider
        threshold (float): Rating threshold for considering an item as relevant
        
        Returns:
        float: Precision@K
        """
        # Group test ratings by user
        user_test_ratings = test_ratings.groupby('userId')
        
        precisions = []
        
        for user_id, user_ratings in user_test_ratings:
            if user_id not in self.user_item_matrix.index:
                continue
                
            # Get recommended items for this user
            recommendations = self.recommend_for_user(user_id, top_n=k, exclude_rated=False)
            
            if recommendations.empty:
                continue
                
            # Get the recommended movie ids
            recommended_movies = recommendations['movieId'].tolist()
            
            # Get the relevant items from the test set (items with rating >= threshold)
            relevant_items = user_ratings[user_ratings['rating'] >= threshold]['movieId'].tolist()
            
            # Calculate precision
            if not recommended_movies:
                precision = 0
            else:
                precision = len(set(relevant_items) & set(recommended_movies)) / len(recommended_movies)
                
            precisions.append(precision)
        
        # Return average precision
        return np.mean(precisions) if precisions else 0
        
    def visualize_user_preferences(self, user_id, top_n=10):
        """
        Visualize a user's preferences and recommendations
        
        Parameters:
        user_id (int): User ID
        top_n (int): Number of top items to show
        
        Returns:
        matplotlib.figure.Figure: The visualization figure
        """
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset.")
            return None
        
        # Get the user's actual ratings
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        top_rated = user_ratings.sort_values('rating', ascending=False).head(top_n)
        
        # Get movie titles for top rated movies
        top_rated = top_rated.merge(self.movies_df[['movieId', 'title']], on='movieId')
        
        # Get the user's predicted ratings for unwatched movies
        recommendations = self.recommend_for_user(user_id, top_n=top_n)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot top rated movies
        ax1.barh(top_rated['title'], top_rated['rating'], color='skyblue')
        ax1.set_title(f'Top {top_n} Rated Movies by User {user_id}')
        ax1.set_xlabel('Rating')
        ax1.set_xlim(0, 5.5)
        ax1.invert_yaxis()
        
        # Plot recommended movies
        ax2.barh(recommendations['title'], recommendations['predicted_rating'], color='lightgreen')
        ax2.set_title(f'Top {top_n} Recommended Movies for User {user_id}')
        ax2.set_xlabel('Predicted Rating')
        ax2.set_xlim(0, 5.5)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'results/visualizations/user_{user_id}_preferences.png')
        
        return fig

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    from sklearn.model_selection import train_test_split
    
    # Example usage
    ratings_df, movies_df = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    cleaned_ratings, cleaned_movies = preprocess_data(ratings_df, movies_df)
    
    # Split data for evaluation
    train_ratings, test_ratings = train_test_split(cleaned_ratings, test_size=0.2, random_state=42)
    
    # Initialize and fit the recommender
    print("\nFitting SVD model...")
    cf_recommender_svd = CollaborativeFilteringRecommender(method='svd')
    cf_recommender_svd.fit(train_ratings, cleaned_movies, n_factors=50)
    
    # Evaluate the model
    rmse = cf_recommender_svd.evaluate(test_ratings)
    print(f"RMSE: {rmse:.4f}")
    
    precision = cf_recommender_svd.precision_at_k(test_ratings, k=10, threshold=3.5)
    print(f"Precision@10: {precision:.4f}")
    
    # Get recommendations for a user
    test_user = cleaned_ratings['userId'].iloc[0]
    print(f"\nGetting recommendations for user {test_user}:")
    recommendations = cf_recommender_svd.recommend_for_user(test_user, top_n=5)
    print(recommendations[['title', 'predicted_rating']])
    
    # Optional: Fit cosine similarity model
    print("\nFitting cosine similarity model...")
    cf_recommender_cosine = CollaborativeFilteringRecommender(method='cosine')
    cf_recommender_cosine.fit(train_ratings, cleaned_movies)
    
    # Get similar movies for a movie
    test_movie = cleaned_ratings['movieId'].iloc[0]
    movie_title = cleaned_movies[cleaned_movies['movieId'] == test_movie]['title'].iloc[0]
    print(f"\nFinding similar movies to {movie_title} (ID: {test_movie}):")
    similar_movies = cf_recommender_cosine.recommend_similar_movies(test_movie, top_n=5)
    print(similar_movies[['title', 'similarity_score']])
    
    # Visualize user preferences
    cf_recommender_svd.visualize_user_preferences(test_user)
