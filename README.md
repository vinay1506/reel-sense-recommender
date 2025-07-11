<h1 align="center">🎬 ReelSense - Movie Recommendation System</h1>

<p align="center">
  A hybrid movie recommendation system built with <strong>Python</strong>, <strong>Django REST API</strong>, and <strong>Machine Learning</strong>.  
  Supports both <strong>Collaborative Filtering</strong> and <strong>Content-Based Filtering</strong> using the MovieLens dataset.
</p>

<p align="center">
  <a href="https://www.djangoproject.com/"><img src="https://img.shields.io/badge/Backend-Django-green?style=flat-square"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/ML-Python-yellow?style=flat-square"></a>
  <a href="https://github.com/vinay1506"><img src="https://img.shields.io/github/stars/vinay1506?style=social"></a>
</p>

---

<h2>🚀 Features</h2>

<ul>
  <li>🎯 Collaborative Filtering using cosine similarity</li>
  <li>🧠 Content-Based Filtering using TF-IDF vectorization</li>
  <li>📊 MovieLens dataset support (100k+ ratings)</li>
  <li>🔗 RESTful API built with Django + Django REST Framework</li>
  <li>🌐 Optional React frontend for full-stack integration</li>
</ul>

---

<h2>📂 Project Structure</h2>

<pre>
vinay1506-reel-sense-recommender/
├── backend/                    # Django backend
│   ├── recommender/
│   │   ├── views.py           # API views
│   │   ├── recommendation_engine.py
│   │   ├── collaborative_filtering.py
│   │   ├── content_based_filtering.py
│   │   ├── data_preprocessing.py
│   │   └── ...
│   └── ...
├── dataset/
│   ├── ratings.csv
│   ├── movies.csv
│   └── ...
├── src/                        # (Optional) React frontend
│   └── ...
</pre>

---

<h2>🛠 Technologies Used</h2>

<ul>
  <li><strong>Python</strong> - Pandas, NumPy, Scikit-Learn</li>
  <li><strong>Django + Django REST Framework</strong> - REST API backend</li>
  <li><strong>MovieLens Dataset</strong> - Real-world ratings and tags</li>
  <li><strong>React</strong> (optional) - TypeScript + Tailwind CSS</li>
</ul>

---

<h2>⚙️ Installation & Setup</h2>

<h3>📦 Backend (Django)</h3>

<pre>
# 1. Clone the repository
git clone https://github.com/vinay1506/reel-sense-recommender.git
cd reel-sense-recommender/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Django development server
python manage.py runserver
</pre>

<h4>🌐 API Endpoints</h4>

<table>
  <tr>
    <th>Endpoint</th>
    <th>Method</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>/api/recommend/?user_id=10</code></td>
    <td>GET</td>
    <td>Get recommendations based on user ID</td>
  </tr>
  <tr>
    <td><code>/api/recommend-by-title/?title=Inception</code></td>
    <td>GET</td>
    <td>Get similar movies based on a movie title</td>
  </tr>
</table>

---

<h3>🌐 Frontend (React - Optional)</h3>

<pre>
# 1. Navigate to frontend directory
cd ../src

# 2. Install dependencies
npm install

# 3. Start React development server
npm run dev
</pre>

React app runs at: <code>http://localhost:5173</code><br>
Django API runs at: <code>http://127.0.0.1:8000</code>

---

<h2>🧠 Recommendation Techniques</h2>

<h4>📌 Collaborative Filtering</h4>
<ul>
  <li>Builds a user-item matrix from ratings</li>
  <li>Uses cosine similarity to compare user preferences</li>
  <li>Predicts movies based on neighbors’ viewing history</li>
</ul>

<h4>📌 Content-Based Filtering</h4>
<ul>
  <li>Uses genres, titles, and tags as metadata</li>
  <li>Applies TF-IDF vectorization to feature vectors</li>
  <li>Recommends similar movies based on metadata similarity</li>
</ul>

---

<h2>🧪 Dataset</h2>

Source: <a href="https://grouplens.org/datasets/movielens/">MovieLens 100K Dataset</a>

<ul>
  <li>🎥 9,742 Movies</li>
  <li>👤 610 Users</li>
  <li>🌟 100,836 Ratings</li>
  <li>🏷️ 3,683 Tags</li>
</ul>

---

<h2>📈 Example Usage</h2>

<pre>
GET http://127.0.0.1:8000/api/recommend/?user_id=10

Response:
{
  "recommendations": [
    "The Matrix (1999)",
    "Interstellar (2014)",
    "Inception (2010)"
  ]
}
</pre>

---

<h2>📄 License</h2>

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

<h2>💬 Contact</h2>

<ul>
  <li><strong>Name:</strong> Vinay Kakumanu</li>
  <li><strong>Email:</strong> <a href="mailto:vinaykakumnu1506@gmail.com">vinaykakumnu1506@gmail.com</a></li>
  <li><strong>LinkedIn:</strong> <a href="https://linkedin.com/in/vinay-kakumanu-25407027b">vinay-kakumanu-25407027b</a></li>
  <li><strong>GitHub:</strong> <a href="https://github.com/vinay1506">vinay1506</a></li>
</ul>

---

<p align="center"><b>⭐ If you like this project, give it a star on GitHub!</b></p>
