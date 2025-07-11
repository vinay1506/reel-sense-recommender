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

---

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
