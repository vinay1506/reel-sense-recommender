
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(ratings_path, movies_path):
    """
    Load the MovieLens dataset
    
    Parameters:
    ratings_path (str): Path to the ratings CSV file
    movies_path (str): Path to the movies CSV file
    
    Returns:
    tuple: (ratings_df, movies_df) - Pandas DataFrames containing the data
    """
    print("Loading data...")
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    print("Data loaded successfully!")
    return ratings_df, movies_df

def preprocess_data(ratings_df, movies_df):
    """
    Preprocess the MovieLens dataset
    
    Parameters:
    ratings_df (DataFrame): Ratings data
    movies_df (DataFrame): Movies data
    
    Returns:
    tuple: (cleaned_ratings, cleaned_movies) - Cleaned DataFrames
    """
    print("Preprocessing data...")
    
    # Check for missing values
    print(f"Missing values in ratings: {ratings_df.isnull().sum().sum()}")
    print(f"Missing values in movies: {movies_df.isnull().sum().sum()}")
    
    # Drop missing values if any
    cleaned_ratings = ratings_df.dropna()
    cleaned_movies = movies_df.dropna()
    
    # Create a year column from title
    cleaned_movies['year'] = cleaned_movies['title'].str.extract('\\((\\d{4})\\)')
    cleaned_movies['title'] = cleaned_movies['title'].str.replace('\\(\\d{4}\\)', '').str.strip()
    
    # Convert genres from string to list
    cleaned_movies['genres'] = cleaned_movies['genres'].str.split('|')
    
    print("Data preprocessing completed!")
    return cleaned_ratings, cleaned_movies

def create_user_item_matrix(ratings_df):
    """
    Create a user-item matrix from ratings data
    
    Parameters:
    ratings_df (DataFrame): Ratings data
    
    Returns:
    DataFrame: User-item matrix
    """
    user_item_matrix = ratings_df.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    return user_item_matrix

if __name__ == "__main__":
    # Example usage
    ratings_df, movies_df = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    cleaned_ratings, cleaned_movies = preprocess_data(ratings_df, movies_df)
    user_item_matrix = create_user_item_matrix(cleaned_ratings)
    print("User-item matrix shape:", user_item_matrix.shape)
