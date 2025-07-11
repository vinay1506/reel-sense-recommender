<h1 align="center">🎬 ReelSense - Movie Recommendation System</h1>

<p align="center">
  A hybrid movie recommendation system built with <strong>Python</strong>, <strong>Django REST API</strong>, and optionally <strong>React</strong>.  
  Supports both <strong>Collaborative Filtering</strong> and <strong>Content-Based Filtering</strong> using the MovieLens dataset.
</p>

<p align="center">
  <a href="https://www.djangoproject.com/"><img src="https://img.shields.io/badge/Backend-Django-green?style=flat-square"></a>
  <a href="https://reactjs.org/"><img src="https://img.shields.io/badge/Frontend-React-blue?style=flat-square"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Machine%20Learning-Python-yellow?style=flat-square"></a>
</p>

---

<h2>🚀 Features</h2>

<ul>
  <li>🎯 Collaborative Filtering using cosine similarity</li>
  <li>🧠 Content-Based Filtering using movie metadata</li>
  <li>📊 MovieLens dataset support (100k+ ratings)</li>
  <li>🔗 RESTful API built with Django and DRF</li>
  <li>🌐 Optional React frontend for modern UI</li>
</ul>


<h4>2. Content-Based Filtering</h4>
Uses movie genres, titles, tags

TF-IDF vectorization + cosine similarity

Recommends movies with similar content


<h2>⚙️ Installation & Setup</h2> <h3>📦 Backend (Django)</h3>
bash
Copy
Edit
# 1. Clone the repository
git clone https://github.com/vinay1506/reel-sense-recommender.git
cd reel-sense-recommender/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Django server
python manage.py runserver
<h4>🔗 API Endpoints</h4>
GET /api/recommend/?user_id=10 → Collaborative filtering for user ID

GET /api/recommend-by-title/?title=The Matrix → Content-based filtering

<h3>🌐 Frontend (React - Optional)</h3>
bash
Copy
Edit
# Navigate to frontend folder
cd ../src

# Install dependencies
npm install

# Start React development server
npm run dev
You can fetch data from the Django API and display movie recommendations in a rich, styled UI.

<h2>🧠 Dataset</h2>
This project uses the MovieLens 100k dataset, which contains:

100,836 ratings

9742 movies

610 users

Stored under dataset/.

<h2>📈 Recommendation Techniques</h2> <h4>1. Collaborative Filtering</h4>
Builds a user-item interaction matrix

Computes cosine similarity between users

Predicts movies based on neighbors’ preferences

<h4>2. Content-Based Filtering</h4>
Uses movie genres, titles, tags

TF-IDF vectorization + cosine similarity

Recommends movies with similar content

<h2>📂 Project Structure</h2>

```bash
vinay1506-reel-sense-recommender/
├── backend/                    # Django backend
│   ├── recommender/
│   │   ├── views.py           # API views
│   │   ├── recommendation_engine.py
│   │   ├── collaborative_filtering.py
│   │   ├── content_based_filtering.py
│   │   └── ...
│   └── ...
├── src/                        # Optional React frontend (Vite + TypeScript)
│   └── ...
└── dataset/
    ├── ratings.csv
    ├── movies.csv
    └── ...
