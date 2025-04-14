
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from recommendation_engine import RecommendationEngine

# Set page configuration
st.set_page_config(
    page_title="ReelSense Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0A1428;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #0A1428;
    }
    h1, h2, h3 {
        color: #E50914;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
    }
    .stButton>button:hover {
        background-color: #B2070E;
        color: white;
    }
    .movie-card {
        background-color: #192841;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .movie-title {
        color: #FFFFFF;
        font-weight: bold;
    }
    .movie-details {
        color: #AAAAAA;
    }
    .rating-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .rating-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .rating-low {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Movie-based"

@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

def format_genres(genres_list):
    """Format genres list for display."""
    if isinstance(genres_list, list):
        return ", ".join(genres_list)
    return genres_list

def display_movie_card(movie, score_col=None):
    """Display a movie in a styled card format."""
    with st.container():
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-details">
                <strong>Genres:</strong> {format_genres(movie['genres'])}
            </div>
            {f'<div><strong>Score:</strong> <span class="rating-high">{movie[score_col]:.2f}</span></div>' if score_col and score_col in movie else ''}
        </div>
        """, unsafe_allow_html=True)

def main():
    # App title
    st.title("🎬 ReelSense: Movie Recommendation System")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Data loading section in sidebar
    st.sidebar.header("Data Source")
    
    data_option = st.sidebar.selectbox(
        "Select data source",
        ["MovieLens 100K", "MovieLens 1M", "Custom Dataset"]
    )
    
    data_path = None
    
    if data_option == "MovieLens 100K":
        data_path = "ml-100k"
        ratings_file = "u.data"
        movies_file = "u.item"
        st.sidebar.info("Using MovieLens 100K dataset with 100,000 ratings from 943 users on 1,682 movies.")
    elif data_option == "MovieLens 1M":
        data_path = "ml-1m"
        ratings_file = "ratings.dat"
        movies_file = "movies.dat"
        st.sidebar.info("Using MovieLens 1M dataset with 1 million ratings from 6,040 users on 3,900 movies.")
    else:
        ratings_path = st.sidebar.text_input("Path to ratings file (CSV format)")
        movies_path = st.sidebar.text_input("Path to movies file (CSV format)")
    
    # Load data button
    if st.sidebar.button("Load Data"):
        try:
            with st.spinner("Loading data..."):
                st.session_state.engine = RecommendationEngine()
                
                if data_option == "Custom Dataset":
                    if not ratings_path or not movies_path:
                        st.error("Please provide paths to both ratings and movies files.")
                        return
                    ratings_df, movies_df = st.session_state.engine.load_data(ratings_path, movies_path)
                else:
                    # Use sample data paths for demo
                    # In a real app, you would need to download the datasets
                    ratings_df, movies_df = st.session_state.engine.load_data(
                        'ml-latest-small/ratings.csv', 
                        'ml-latest-small/movies.csv'
                    )
                
                st.session_state.data_loaded = True
                st.success(f"Data loaded successfully: {len(ratings_df)} ratings and {len(movies_df)} movies.")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # Model training section in sidebar (only show if data is loaded)
    if st.session_state.data_loaded:
        st.sidebar.header("Model Training")
        
        n_factors = st.sidebar.slider("Number of latent factors for SVD", min_value=20, max_value=200, value=50, step=10)
        method = st.sidebar.selectbox("Collaborative filtering method", ["svd", "cosine"])
        
        if st.sidebar.button("Train Models"):
            try:
                with st.spinner("Training recommendation models..."):
                    st.session_state.engine.train_models(n_factors=n_factors, collaborative_method=method)
                    st.session_state.models_trained = True
                    st.success("Models trained successfully!")
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
        
        # Option to save/load models
        st.sidebar.header("Save/Load Models")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Save Models") and st.session_state.models_trained:
                try:
                    with st.spinner("Saving models..."):
                        st.session_state.engine.save_models()
                        st.success("Models saved successfully!")
                except Exception as e:
                    st.error(f"Error saving models: {str(e)}")
        
        with col2:
            if st.button("Load Models"):
                try:
                    with st.spinner("Loading models..."):
                        st.session_state.engine = RecommendationEngine()
                        engine = st.session_state.engine.load_models()
                        
                        if engine:
                            st.session_state.models_trained = True
                            st.success("Models loaded successfully!")
                        else:
                            st.warning("Could not load models. Train and save them first.")
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
    
    # Main area
    if not st.session_state.data_loaded:
        # Show welcome message if data is not loaded
        st.write("""
        ## Welcome to ReelSense Movie Recommender!
        
        This application uses machine learning to recommend movies based on your preferences.
        It implements both content-based and collaborative filtering approaches.
        
        ### To get started:
        
        1. Select a dataset in the sidebar
        2. Click "Load Data" to load the movie and rating data
        3. Train the recommendation models
        4. Use the tabs below to get recommendations
        
        ### Features:
        
        - Content-based recommendations using movie genres and TF-IDF
        - Collaborative filtering using matrix factorization (SVD)
        - Hybrid recommendations combining both approaches
        - Visualizations to understand movie trends and user preferences
        """)
        
        # Show sample image
        st.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80", 
                 caption="Movie Recommendations")
    
    elif st.session_state.models_trained:
        # Tabs for different recommendation types
        tab1, tab2, tab3, tab4 = st.tabs([
            "Movie-based", "User-based", "Hybrid", "Visualizations"
        ])
        
        with tab1:
            st.header("Get Movie Recommendations")
            st.write("Find movies similar to your favorites based on content (genres, themes, etc.)")
            
            # Movie search input
            movie_title = st.text_input("Enter a movie title", "Toy Story")
            top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=5)
            
            if st.button("Get Content-Based Recommendations"):
                try:
                    with st.spinner("Finding similar movies..."):
                        recommendations = st.session_state.engine.get_movie_recommendations(movie_title, top_n=top_n)
                        
                        if recommendations is not None:
                            st.write(f"### Top {top_n} Movies Similar to '{movie_title}'")
                            
                            for _, movie in recommendations.iterrows():
                                display_movie_card(movie, 'similarity_score')
                            
                            # Download results option
                            st.download_button(
                                label="Download Recommendations as CSV",
                                data=convert_df_to_csv(recommendations),
                                file_name=f"{movie_title.replace(' ', '_')}_recommendations.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning(f"No recommendations found for '{movie_title}'. Please check the movie title.")
                except Exception as e:
                    st.error(f"Error getting recommendations: {str(e)}")
        
        with tab2:
            st.header("Get User Recommendations")
            st.write("Get personalized recommendations based on user rating patterns")
            
            # User ID input
            user_id = st.number_input("Enter User ID", min_value=1, value=1, step=1)
            top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=5, key="user_top_n")
            
            if st.button("Get Collaborative Filtering Recommendations"):
                try:
                    with st.spinner("Finding personalized recommendations..."):
                        recommendations = st.session_state.engine.get_user_recommendations(user_id, top_n=top_n)
                        
                        if recommendations is not None:
                            st.write(f"### Top {top_n} Recommendations for User {user_id}")
                            
                            for _, movie in recommendations.iterrows():
                                display_movie_card(movie, 'predicted_rating')
                            
                            # Download results option
                            st.download_button(
                                label="Download Recommendations as CSV",
                                data=convert_df_to_csv(recommendations),
                                file_name=f"user_{user_id}_recommendations.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning(f"No recommendations found for user {user_id}. Please check the user ID.")
                except Exception as e:
                    st.error(f"Error getting recommendations: {str(e)}")
        
        with tab3:
            st.header("Hybrid Recommendations")
            st.write("Get the best of both worlds - recommendations based on both content and collaborative filtering")
            
            # Inputs for hybrid recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                user_id = st.number_input("Enter User ID", min_value=1, value=1, step=1, key="hybrid_user_id")
            
            with col2:
                movie_title = st.text_input("Enter a movie title (optional)", "", key="hybrid_movie")
            
            top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=5, key="hybrid_top_n")
            content_weight = st.slider("Content-based weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            
            if st.button("Get Hybrid Recommendations"):
                try:
                    with st.spinner("Finding hybrid recommendations..."):
                        movie_param = movie_title if movie_title else None
                        recommendations = st.session_state.engine.get_hybrid_recommendations(
                            user_id, movie_title=movie_param, content_weight=content_weight, top_n=top_n
                        )
                        
                        if recommendations is not None:
                            st.write(f"### Top {top_n} Hybrid Recommendations for User {user_id}")
                            
                            for _, movie in recommendations.iterrows():
                                display_movie_card(movie, 'score' if 'score' in recommendations.columns else 'predicted_rating')
                            
                            # Download results option
                            st.download_button(
                                label="Download Recommendations as CSV",
                                data=convert_df_to_csv(recommendations),
                                file_name=f"hybrid_recommendations.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning(f"No recommendations found. Please check your inputs.")
                except Exception as e:
                    st.error(f"Error getting recommendations: {str(e)}")
        
        with tab4:
            st.header("Visualizations")
            st.write("Explore movie trends and user preferences")
            
            viz_option = st.selectbox(
                "Select visualization",
                ["Movie Ratings by Genre", "Ratings by Year", "User Activity", "Popular Movies"]
            )
            
            if viz_option == "Movie Ratings by Genre":
                try:
                    with st.spinner("Generating visualization..."):
                        fig = st.session_state.engine.visualize_movie_ratings_by_genre()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
            
            elif viz_option == "Ratings by Year":
                try:
                    with st.spinner("Generating visualization..."):
                        fig = st.session_state.engine.visualize_rating_distribution_by_year()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
            
            elif viz_option == "User Activity":
                try:
                    with st.spinner("Generating visualization..."):
                        fig = st.session_state.engine.visualize_user_activity()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
            
            elif viz_option == "Popular Movies":
                min_ratings = st.slider("Minimum number of ratings", min_value=10, max_value=500, value=100, step=10)
                num_movies = st.slider("Number of movies to show", min_value=5, max_value=25, value=10, step=5)
                
                try:
                    with st.spinner("Finding popular movies..."):
                        popular_movies = st.session_state.engine.get_popular_movies(n=num_movies, min_ratings=min_ratings)
                        
                        if not popular_movies.empty:
                            st.write(f"### Top {num_movies} Popular Movies (minimum {min_ratings} ratings)")
                            
                            for _, movie in popular_movies.iterrows():
                                with st.container():
                                    st.markdown(f"""
                                    <div class="movie-card">
                                        <div class="movie-title">{movie['title']}</div>
                                        <div class="movie-details">
                                            <strong>Genres:</strong> {format_genres(movie['genres'])}
                                        </div>
                                        <div>
                                            <strong>Average Rating:</strong> <span class="rating-high">{movie['avg_rating']:.2f}</span>
                                            <strong>Number of Ratings:</strong> {movie['rating_count']}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("No popular movies found with the selected criteria.")
                except Exception as e:
                    st.error(f"Error finding popular movies: {str(e)}")
    
    else:
        # Show message if data is loaded but models are not trained
        st.info("Data loaded! Please train the recommendation models in the sidebar to continue.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **ReelSense Recommender System**
    
    A machine learning-based movie recommendation system using 
    content-based and collaborative filtering approaches.
    """)

if __name__ == "__main__":
    main()
