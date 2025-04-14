
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(ratings_df, movies_df):
    """
    Perform exploratory data analysis on the MovieLens dataset
    
    Parameters:
    ratings_df (DataFrame): Ratings data
    movies_df (DataFrame): Movies data
    
    Returns:
    dict: Dictionary with EDA results and visualizations
    """
    print("Performing exploratory data analysis...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results/visualizations', exist_ok=True)
    
    eda_results = {}
    
    # Basic statistics
    eda_results['ratings_stats'] = ratings_df.describe()
    eda_results['movies_stats'] = movies_df.describe(include='all')
    
    # Number of ratings per user
    user_ratings_count = ratings_df.groupby('userId').size()
    eda_results['avg_ratings_per_user'] = user_ratings_count.mean()
    eda_results['median_ratings_per_user'] = user_ratings_count.median()
    
    # Number of ratings per movie
    movie_ratings_count = ratings_df.groupby('movieId').size()
    eda_results['avg_ratings_per_movie'] = movie_ratings_count.mean()
    eda_results['median_ratings_per_movie'] = movie_ratings_count.median()
    
    # Rating distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_df['rating'], bins=10, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('results/visualizations/rating_distribution.png')
    
    # Top 10 most rated movies
    top_rated_movies = movie_ratings_count.sort_values(ascending=False).head(10)
    top_rated_movies_df = movies_df[movies_df['movieId'].isin(top_rated_movies.index)]
    eda_results['top_rated_movies'] = top_rated_movies_df
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_rated_movies.values, y=top_rated_movies_df['title'])
    plt.title('Top 10 Movies by Number of Ratings')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Movie Title')
    plt.tight_layout()
    plt.savefig('results/visualizations/top10_rated_movies.png')
    
    # Genre distribution
    genre_counts = {}
    for genres in movies_df['genres']:
        for genre in genres:
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1
    
    genre_df = pd.DataFrame({
        'Genre': list(genre_counts.keys()),
        'Count': list(genre_counts.values())
    }).sort_values('Count', ascending=False)
    
    eda_results['genre_distribution'] = genre_df
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Count', y='Genre', data=genre_df)
    plt.title('Distribution of Movie Genres')
    plt.tight_layout()
    plt.savefig('results/visualizations/genre_distribution.png')
    
    # Rating heatmap by year and genre
    if 'year' in movies_df.columns:
        # Create a year-genre matrix
        year_genre_rating = {}
        
        for idx, movie in movies_df.iterrows():
            movie_id = movie['movieId']
            year = movie['year']
            genres = movie['genres']
            
            if year is not None and pd.notna(year):
                ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']
                avg_rating = ratings.mean() if not ratings.empty else np.nan
                
                for genre in genres:
                    if (year, genre) not in year_genre_rating:
                        year_genre_rating[(year, genre)] = []
                    
                    year_genre_rating[(year, genre)].append(avg_rating)
        
        # Calculate average ratings per year and genre
        year_genre_avg = {}
        for key, ratings in year_genre_rating.items():
            ratings = [r for r in ratings if not pd.isna(r)]
            if ratings:
                year_genre_avg[key] = np.mean(ratings)
        
        # Convert to DataFrame for heatmap
        years = sorted(list({y for y, g in year_genre_avg.keys()}))
        genres = sorted(list({g for y, g in year_genre_avg.keys()}))
        
        heatmap_data = np.zeros((len(years), len(genres)))
        
        for i, year in enumerate(years):
            for j, genre in enumerate(genres):
                if (year, genre) in year_genre_avg:
                    heatmap_data[i, j] = year_genre_avg[(year, genre)]
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            heatmap_data,
            xticklabels=genres,
            yticklabels=years,
            cmap="YlGnBu",
            annot=False
        )
        plt.title('Average Rating by Year and Genre')
        plt.xlabel('Genre')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.savefig('results/visualizations/year_genre_heatmap.png')
    
    print("EDA completed!")
    return eda_results

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    # Example usage
    ratings_df, movies_df = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    cleaned_ratings, cleaned_movies = preprocess_data(ratings_df, movies_df)
    eda_results = perform_eda(cleaned_ratings, cleaned_movies)
    
    # Print some key findings
    print("\nKey EDA Findings:")
    print(f"Average ratings per user: {eda_results['avg_ratings_per_user']:.2f}")
    print(f"Average ratings per movie: {eda_results['avg_ratings_per_movie']:.2f}")
    print("\nTop rated movies:")
    print(eda_results['top_rated_movies']['title'].head())
