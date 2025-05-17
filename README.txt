# 🎬🎮 Movies and Games Recommendation Engine

A powerful hybrid recommendation system built using Python and Streamlit that suggests **movies** and **video games** based on user input.  
It leverages content-based filtering, smart fuzzy string matching, and cosine similarity to return relevant results in a sleek, interactive UI.

---

## 🌐 Try it Live

🎯 **Experience it in action — no install required!**  
Click below to try out the full web app on Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/HuggingFace-LiveApp-blue?logo=huggingface)](https://huggingface.co/spaces/Hoshang08/Movies-Games-Recommendation-Engine)

---

## 📌 Project Overview

This multi-phase project includes two custom-built recommendation engines for **movies** and **games**, powered by cosine similarity and metadata vectorization.  
The app features a polished **multi-page Streamlit interface**, allowing seamless switching between recommendation sections and a feedback form.

---

## 🚀 Features

- **🎥 Movie Recommendation Engine**  
  - Pulls metadata via TMDB API  
  - Uses vectorized descriptions + cosine similarity  
  - Smart fuzzy search handles typos and close matches  
  - Displays posters, ratings, and streaming links

- **🎮 Game Recommendation Engine**  
  - Uses RAWG API (or Steam) for data  
  - Processes tags, genres, platforms, and descriptions  
  - Recommends similar games using cosine similarity  
  - Shows game covers, ratings, and purchase/store links

- **🖥️ Streamlit Multi-Page App**  
  - Dedicated pages for Movies, Games, and Contact  
  - Sidebar navigation with a clean UI  
  - Contact Me form saves user messages to Google Sheets

---

## 🔁 Workflow

### 1️⃣ Movie Recommendation Engine  
- TMDB API → Metadata  
- Preprocess & vectorize text  
- Cosine similarity for content filtering  
- Fuzzy string match for smart search  
- Display results in 4-column grid with posters

### 2️⃣ Game Recommendation Engine  
- RAWG API → Metadata  
- Data cleanup & feature extraction  
- Vectorization + cosine similarity  
- Fuzzy match for flexible input  
- Display results with images, ratings, and store links

### 3️⃣ Streamlit Integration  
- Multi-page structure: Home, Movies, Games, Contact  
- Recommendation logic linked per page  
- Form input → Google Sheets backend (via Google API)

### 4️⃣ Deployment  
- Code cleanup and documentation  
- `requirements.txt` for dependency management  
- Deployed on Hugging Face Spaces (with `gdown` for large file fetching)

### 5️⃣ Future Enhancements  
- Add collaborative filtering or hybrid models  
- Integrate user authentication and preferences  
- Accept user ratings to dynamically fine-tune suggestions

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. pip install -r requirements.txt

3. streamlit run app.py
