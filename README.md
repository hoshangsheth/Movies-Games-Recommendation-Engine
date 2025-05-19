# 🎬 Movies & Games 🎮 Recommendation Engine

A hybrid recommendation system that lets you search for movies and games — then delivers personalized, relevant suggestions based on your input. Powered by cosine similarity on over **30,000+ TF-IDF vectorized features** and enhanced with smart fuzzy search for seamless, intuitive matching.

---

### 🎯 Try It Live!  
No installations, no fuss — just click and dive into your next favorite movie or game right from your browser:

[![Streamlit App](https://img.shields.io/badge/Streamlit-LiveApp-green?logo=streamlit)](https://movies-games-recommendation-engine.streamlit.app/)

---

## 🚀 Features
- **Dual Recommendation:** Switch effortlessly between movies and games recommendations
- **Fuzzy Search:** Handles typos and approximate inputs gracefully using fuzzy string matching
- **Rich Data:** Fetches posters, ratings, and relevant streaming or purchase links via TMDB and RAWG APIs
- **Advanced Similarity:** Recommendations computed using cosine similarity on TF-IDF vectorized features — over **30,000+ unique tokens** analyzed for maximum relevance
- **Efficient & Scalable:** Optimized data processing with NumPy and Pandas ensures fast responses even on large datasets

---

## 🔧 Tech Stack & Techniques
- **Python** — Core programming language powering the logic
- **Streamlit** — Interactive web app framework for easy deployment
- **TMDB API & RAWG API** — Source of rich metadata for movies and games
- **FuzzyWuzzy** — Implements fuzzy string matching for smart search capability
- **TF-IDF Vectorization** — Converts textual metadata into high-dimensional numeric vectors representing importance of terms
- **Cosine Similarity** — Calculates similarity between TF-IDF vectors to find closest matches
- **NumPy & Pandas** — High-performance data manipulation and computation libraries

---

## 📈 Why This Project?  
Blending NLP techniques like TF-IDF with classic ML similarity measures and real-time API data fetching, this project showcases how to build scalable, intelligent recommendation engines. Whether you’re a movie buff or a gamer, this engine serves up personalized content that truly fits your taste—powered by over 30,000 features crunching behind the scenes.

---
