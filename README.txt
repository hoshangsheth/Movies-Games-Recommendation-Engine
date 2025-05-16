# Movies and Games Recommendation Engine

A hybrid recommendation system built with Python and Streamlit that suggests movies and games based on user input. The project leverages content-based filtering techniques, fuzzy string matching for smart search, and presents results in an interactive web app.

---

## Project Overview

This project is developed in multiple phases, focusing on building two main recommendation engines: one for movies and another for games. Both engines use textual metadata and cosine similarity to provide personalized recommendations. The final product is integrated into a multi-page Streamlit app with a user-friendly interface.

---

## Features

- **Movie Recommendation Engine**  
  Fetches movie metadata from TMDB API, preprocesses data, vectorizes descriptions, and uses cosine similarity to recommend similar movies.  
  Includes smart search with fuzzy matching to handle typos.

- **Game Recommendation Engine**  
  Fetches game data from RAWG API (or Steam API), cleans and preprocesses the data, vectorizes textual features, and recommends similar games.  
  Also includes fuzzy string matching for robust search functionality.

- **Multi-page Streamlit App**  
  User can switch between Movies and Games recommendation pages.  
  Displays posters, ratings, and streaming/purchase links in a neat grid layout.  
  Includes an optional Contact Me form to collect user feedback.

---

## Workflow

### Step 1: Movie Recommendation Engine  
- Fetch movie metadata from TMDB API  
- Clean and preprocess data into a structured format  
- Vectorize movie descriptions using TF-IDF or Count Vectorizer  
- Calculate cosine similarity for recommendations  
- Implement fuzzy matching for smart search  
- Display recommendations in Streamlit with movie posters and ratings  

### Step 2: Game Recommendation Engine  
- Fetch game metadata from RAWG API or Steam API  
- Clean and preprocess game data  
- Vectorize game descriptions and metadata  
- Calculate cosine similarity for game recommendations  
- Use fuzzy matching for approximate game name search  
- Display results with game covers, ratings, and purchase links  

### Step 3: Integration & Streamlit Multi-Page Setup  
- Multi-page app navigation: Home, Movies, Games, Contact Me  
- Search and display recommendations on respective pages  
- Optional contact form to collect user input and store in Google Sheets  

### Step 4: Deployment  
- Clean and document code  
- Prepare requirements.txt  
- Deploy on Streamlit Cloud linked with GitHub repository  

### Step 5: Post-Deployment Enhancements (Future Work)  
- Integrate collaborative filtering or hybrid recommendation models  
- Add user login and preferences  
- Enable user ratings to refine recommendations  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
