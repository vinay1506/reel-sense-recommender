
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from math import sqrt

def evaluate_rmse(true_ratings, predicted_ratings):
    """
    Calculate Root Mean Square Error
    
    Parameters:
    true_ratings (array-like): True ratings
    predicted_ratings (array-like): Predicted ratings
    
    Returns:
    float: RMSE value
    """
    return sqrt(mean_squared_error(true_ratings, predicted_ratings))

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Precision@K
    
    Parameters:
    recommended_items (list): List of recommended item IDs
    relevant_items (list): List of relevant (ground truth) item IDs
    k (int): Number of recommendations to consider
    
    Returns:
    float: Precision@K value
    """
    # Ensure we only consider k recommendations
    recommended_items = recommended_items[:k]
    
    # Calculate number of relevant items in the recommendations
    num_relevant = len(set(recommended_items) & set(relevant_items))
    
    # Calculate precision
    return num_relevant / len(recommended_items) if recommended_items else 0

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Recall@K
    
    Parameters:
    recommended_items (list): List of recommended item IDs
    relevant_items (list): List of relevant (ground truth) item IDs
    k (int): Number of recommendations to consider
    
    Returns:
    float: Recall@K value
    """
    # Ensure we only consider k recommendations
    recommended_items = recommended_items[:k]
    
    # Calculate number of relevant items in the recommendations
    num_relevant = len(set(recommended_items) & set(relevant_items))
    
    # Calculate recall
    return num_relevant / len(relevant_items) if relevant_items else 0

def mean_average_precision(recommendations_dict, ground_truth_dict):
    """
    Calculate Mean Average Precision
    
    Parameters:
    recommendations_dict (dict): Dictionary mapping user IDs to lists of recommended item IDs
    ground_truth_dict (dict): Dictionary mapping user IDs to lists of relevant item IDs
    
    Returns:
    float: MAP value
    """
    average_precisions = []
    
    for user_id in recommendations_dict:
        if user_id not in ground_truth_dict:
            continue
            
        recommended_items = recommendations_dict[user_id]
        relevant_items = ground_truth_dict[user_id]
        
        if not relevant_items:
            continue
        
        # Calculate cumulative sum of precision at each position
        # where a relevant item was found
        precision_sum = 0
        num_relevant_found = 0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                num_relevant_found += 1
                precision_sum += num_relevant_found / (i + 1)
        
        # Calculate average precision for this user
        ap = precision_sum / len(relevant_items) if relevant_items else 0
        average_precisions.append(ap)
    
    # Return mean of average precisions
    return np.mean(average_precisions) if average_precisions else 0

def normalized_discounted_cumulative_gain(recommended_items, item_relevance, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)
    
    Parameters:
    recommended_items (list): List of recommended item IDs
    item_relevance (dict): Dictionary mapping item IDs to relevance scores
    k (int): Number of recommendations to consider
    
    Returns:
    float: NDCG value
    """
    # Ensure we only consider k recommendations
    recommended_items = recommended_items[:k]
    
    # Calculate DCG
    dcg = 0
    for i, item in enumerate(recommended_items):
        rel = item_relevance.get(item, 0)
        dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because i is 0-indexed
    
    # Calculate ideal DCG
    ideal_items = sorted(item_relevance.items(), key=lambda x: x[1], reverse=True)
    ideal_items = [item[0] for item in ideal_items][:k]
    
    idcg = 0
    for i, item in enumerate(ideal_items):
        rel = item_relevance.get(item, 0)
        idcg += (2 ** rel - 1) / np.log2(i + 2)
    
    # Calculate NDCG
    return dcg / idcg if idcg > 0 else 0

def plot_evaluation_metrics(metrics_dict, title="Recommendation System Evaluation", figsize=(12, 8)):
    """
    Plot evaluation metrics
    
    Parameters:
    metrics_dict (dict): Dictionary mapping metric names to values
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The plot figure
    """
    plt.figure(figsize=figsize)
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create bar plot
    bars = plt.bar(metrics, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', 
            va='bottom'
        )
    
    plt.title(title)
    plt.ylim(0, 1.1)  # Most metrics are in [0,1] range
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/visualizations/evaluation_metrics.png')
    return plt.gcf()

def compare_recommenders(recommenders_dict, metric_func, test_data, **metric_kwargs):
    """
    Compare multiple recommenders based on a metric
    
    Parameters:
    recommenders_dict (dict): Dictionary mapping recommender names to recommender objects
    metric_func (function): Function to calculate the metric
    test_data (DataFrame): Test dataset
    metric_kwargs (dict): Additional arguments for the metric function
    
    Returns:
    DataFrame: Comparison results
    """
    results = {}
    
    for name, recommender in recommenders_dict.items():
        metric_value = metric_func(recommender, test_data, **metric_kwargs)
        results[name] = metric_value
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Recommender': list(results.keys()),
        'Metric Value': list(results.values())
    })
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Recommender', y='Metric Value', data=results_df)
    plt.title(f'Comparison of Recommenders by {metric_func.__name__}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/visualizations/recommender_comparison_{metric_func.__name__}.png')
    
    return results_df

def plot_recommendation_coverage(recommenders_dict, movies_df, n_recommendations=10):
    """
    Plot the coverage of different recommenders
    
    Parameters:
    recommenders_dict (dict): Dictionary mapping recommender names to recommender objects
    movies_df (DataFrame): Movies data
    n_recommendations (int): Number of recommendations per user
    
    Returns:
    matplotlib.figure.Figure: The plot figure
    """
    coverage_results = {}
    total_movies = len(movies_df)
    
    for name, recommender in recommenders_dict.items():
        # Get unique movies that appear in recommendations
        recommended_movies = set()
        
        for user_id in recommender.user_item_matrix.index:
            try:
                recommendations = recommender.recommend_for_user(user_id, top_n=n_recommendations)
                recommended_movies.update(recommendations['movieId'].tolist())
            except Exception as e:
                print(f"Error getting recommendations for user {user_id} with {name}: {e}")
        
        # Calculate coverage
        coverage = len(recommended_movies) / total_movies
        coverage_results[name] = coverage
    
    # Create and plot results
    plt.figure(figsize=(10, 6))
    plt.bar(list(coverage_results.keys()), list(coverage_results.values()), color='lightgreen')
    plt.title(f'Recommendation Coverage (Top {n_recommendations} Recommendations)')
    plt.ylabel('Coverage (% of total movies)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, (name, coverage) in enumerate(coverage_results.items()):
        plt.text(i, coverage + 0.02, f'{coverage:.1%}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/recommendation_coverage.png')
    
    return plt.gcf()

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    from content_based_filtering import ContentBasedRecommender
    from collaborative_filtering import CollaborativeFilteringRecommender
    from sklearn.model_selection import train_test_split
    
    # Load and preprocess data
    ratings_df, movies_df = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    cleaned_ratings, cleaned_movies = preprocess_data(ratings_df, movies_df)
    
    # Split data for evaluation
    train_ratings, test_ratings = train_test_split(cleaned_ratings, test_size=0.2, random_state=42)
    
    # Fit recommenders
    print("Fitting recommenders for evaluation...")
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(cleaned_movies)
    
    svd_recommender = CollaborativeFilteringRecommender(method='svd')
    svd_recommender.fit(train_ratings, cleaned_movies)
    
    cosine_recommender = CollaborativeFilteringRecommender(method='cosine')
    cosine_recommender.fit(train_ratings, cleaned_movies)
    
    # Evaluate RMSE for SVD recommender
    print("\nEvaluating SVD recommender...")
    rmse = svd_recommender.evaluate(test_ratings)
    print(f"RMSE: {rmse:.4f}")
    
    # Calculate Precision@K for different recommenders
    test_users = test_ratings['userId'].unique()[:10]  # Use a subset for faster testing
    recommenders = {
        'Content-Based': content_recommender,
        'SVD': svd_recommender,
        'Cosine Similarity': cosine_recommender
    }
    
    # Example evaluation
    metrics = {
        'RMSE (SVD)': rmse,
        'Precision@10 (SVD)': svd_recommender.precision_at_k(test_ratings, k=10),
        'Precision@10 (Cosine)': cosine_recommender.precision_at_k(test_ratings, k=10)
    }
    
    # Plot metrics
    print("\nPlotting evaluation metrics...")
    plot_evaluation_metrics(metrics)
    
    # Plot recommendation coverage
    print("\nCalculating recommendation coverage...")
    plot_recommendation_coverage(recommenders, cleaned_movies)
    
    print("Evaluation completed!")
