# 🎬 Movies & Games 🎮 Recommendation Engine

A hybrid recommendation system that lets you search for movies and games — then delivers personalized, relevant suggestions based on your input. Powered by cosine similarity on over **30,000+ TF-IDF vectorized features** and enhanced with smart fuzzy search for seamless, intuitive matching.

---

### 🎯 Try It Live!  
No installations, no fuss — just click and dive into your next favorite movie or game right from your browser:

[![Streamlit App](https://img.shields.io/badge/Streamlit-LiveApp-green?logo=streamlit)](https://movies-games-recommendation-engine.streamlit.app/)

---

## 📌 Features

🔍 **Smart Search with Aliases & Fuzzy Matching**  
🧠 **Cosine Similarity-Based Recommendations**  
🎞️ **Movies**: Posters, trailers, cast pictures, descriptions, genres, watch links, and ratings  
🕹️ **Games**: Store links, developer/publisher, tags, ESRB ratings, website, and screenshots  
🎨 **Futuristic UI**: Custom CSS with animated transitions, modern sidebar, and responsive layout  
🧭 **Intuitive Navigation**: Sidebar menu with pages for Home, Recommend Movies, Recommend Games, and Contact Me  
📩 **Google Sheets Integration** for the contact form

---

## 🧠 Recommendation Logic

- **Data Cleaning & Preprocessing**: Titles cleaned using regex
- **Fuzzy Search**: Implemented via `rapidfuzz` to handle partial and alias-based searches
- **Similarity Computation**: Precomputed Cosine Similarity Matrix (Pickle + Numpy)
- **Smart Aliasing**: Robust dictionaries for common abbreviations (e.g., "ZNMD" → *Zindagi Na Milegi Dobara*)
- **Metadata Enhancement**: Enriched recommendations with trailers, store links, cast, ratings, screenshots, etc.

---

## 🛠️ Tech Stack

| Tool        | Use                                                                 |
|-------------|----------------------------------------------------------------------|
| Python      | Core logic                                                           |
| Streamlit   | Frontend & app deployment                                            |
| Pandas/Numpy| Data wrangling and similarity matrices                               |
| RapidFuzz   | Fuzzy string matching                                                |
| Pickle/NPY  | Serialized cosine similarity matrices                                |
| Google Sheets | Contact form backend via `gspread`                                 |
| CSS         | Custom styling & animations                                          |

---

## 📈 Why This Project?  
Blending NLP techniques like TF-IDF with classic ML similarity measures and real-time API data fetching, this project showcases how to build scalable, intelligent recommendation engines. Whether you’re a movie buff or a gamer, this engine serves up personalized content that truly fits your taste—powered by over 30,000 features crunching behind the scenes.

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/hoshangsheth/Movies-Games-Recommendation-Engine.git
cd Movies-Games-Recommendation-Engine

# Create virtual environment & install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
