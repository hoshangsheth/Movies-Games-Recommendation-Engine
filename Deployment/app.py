import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import re
from rapidfuzz import process, fuzz
from datetime import datetime
from streamlit_option_menu import option_menu
from PIL import Image
import os
import gdown
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# ---------------------------- Set Streamlit page configuration ----------------------------
st.set_page_config(page_title="Movie - Game Recommendation Engine", layout="wide")

# Load the files:

# This gets the absolute path of the current script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_resource
def load_movies():
    file_path = os.path.join(BASE_DIR, "movies_recommended.pkl")
    return pickle.load(open(file_path, 'rb'))

@st.cache_resource
def load_movies_matrix():
    file_path = os.path.join(BASE_DIR, "cosine_sim.pkl")
    return pickle.load(open(file_path, 'rb'))

@st.cache_resource
def load_games():
    file_path = os.path.join(BASE_DIR, "games_recommended.pkl")
    return pickle.load(open(file_path, 'rb'))

@st.cache_resource
def load_games_matrix():
    file_path = os.path.join(BASE_DIR, "cosine_sim.npy")
    return np.load(file_path)

# Load all
movies = load_movies()
movies_matrix = load_movies_matrix()
games = load_games()
games_matrix = load_games_matrix()

# Aliases:
movie_aliases = {
    "znmd": "Zindagi Na Milegi Dobara",
    "dch": "Dil Chahta Hai",
    "3idiots": "3 Idiots",
    "k3g": "Kabhi Khushi Kabhie Gham",
    "lagaan": "Lagaan",
    "dhoom": "Dhoom 3",
    "tzp": "Taare Zameen Par",
    "bb": "Bajrangi Bhaijaan",
    "dangal": "Dangal",
    "aaa": "Andaz Apna Apna",
    "barfi": "Barfi!",
    "mi": "Mission Impossible",
    "tdk": "The Dark Knight",
    "inception": "Inception",
    "lotr": "The Lord of the Rings",
    "matrix": "The Matrix",
    "endgame": "Avengers: Endgame",
    "forrest": "Forrest Gump",
    "interstellar": "Interstellar",
    "jp": "Jurassic Park",
    "potc": "Pirates of the Caribbean",
    "singham": "Singham Again",
    "ddlj": "Dilwale Dulhania Le Jayenge",
    "rrr": "RRR",
    "kgf": "KGF Chapter 1",
    "kgf2": "KGF Chapter 2",
    "koi mil gaya": "Koi... Mil Gaya",
    "kmg": "Koi... Mil Gaya",
    "krish": "Krrish",
    "bahubali": "Baahubali: The Beginning",
    "bahubali2": "Baahubali 2: The Conclusion",
    "bb2": "Bhool Bhulaiyaa 2",
    "qsqt": "Qayamat Se Qayamat Tak",
    "gadar": "Gadar: Ek Prem Katha",
    "gadar2": "Gadar 2",
    "sholay": "Sholay",
    "mnik": "My Name is Khan",
    "swades": "Swades",
    "kites": "Kites",
    "dostana": "Dostana",
    "chak de": "Chak De! India",
    "md": "Mohabbatein",
    "rnpm": "Rab Ne Bana Di Jodi",
    "ktkg": "Kuch Tum Kaho Kuch Hum Kahein",
    "kkn": "Kabir Khan",
    "tmk": "Tees Maar Khan",
    "angry birds": "The Angry Birds Movie",
    "angry birds 2": "The Angry Birds Movie 2",
    "nemo": "Finding Nemo"
}

alias_dict = {
    # GTA
    'gta5': 'grand theft auto v',
    'gta 5': 'grand theft auto v',
    'gta v': 'grand theft auto v',
    'gta': 'grand theft auto',
    'gta 4': 'grand theft auto iv',
    'gta4': 'grand theft auto iv',

    # Witcher
    'witcher 3': 'the witcher 3 wild hunt',
    'tw3': 'the witcher 3 wild hunt',

    # BatMan
    'batman': 'batman arkham knight',

    # Uncharted
    'uncharted': 'uncharted drakes fortune',
    'uncharted 4': 'uncharted 4 a thiefs end',
    'uncharted lost': 'uncharted lost legacy',

    # Test Drive
    'test drive': 'test drive unlimited',

    # Forza
    'forza': 'forza horizon 5',

    # Need for Speed
    'nfs': 'need for speed',
    'nfs heat': 'need for speed heat',
    'nfs unbound': 'need for speed unbound',

    # Red Dead Redemption
    'rdr': 'red dead redemption',
    'rdr2': 'red dead redemption 2',
    'red dead 2': 'red dead redemption 2',
    'red dead': 'red dead redemption 2',

    # Zelda
    'botw': 'the legend of zelda breath of the wild',
    'zelda botw': 'the legend of zelda breath of the wild',

    # Elden Ring
    'elden': 'elden ring',
    'elden ring': 'elden ring',

    # God Of War
    'gow': 'god of war',
    'god of war 4': 'god of war',
    'god of war': 'god of war',

    # MineCraft
    'minecraft': 'minecraft',

    # Fortnite
    'fortnite': 'fortnite',

    # Call Of Duty
    'cod': 'call of duty',
    'call of duty': 'call of duty',

    # Horizon
    'hzd': 'horizon zero dawn',
    'horizon': 'horizon zero dawn',

    # Spider-Man
    'spiderman': 'marvels spider man',
    'spider man': 'marvels spider man',
    'marvel spiderman': 'marvels spider man',

    # CyberPunk
    'cyberpunk': 'cyberpunk 2077',
    'cyberpunk 2077': 'cyberpunk 2077',

    # Assassin's Creed
    'ac valhalla': 'assassins creed valhalla',
    'assassins creed valhalla': 'assassins creed valhalla',
    'acv': 'assassins creed valhalla',
    'ac': 'assassins creed',
    'ac2': 'assassins creed 2',

    # Resident Evil
    're8': 'resident evil village',
    'resident evil 8': 'resident evil village',
    'village': 'resident evil village',

    # Last Of Us
    'tlou': 'the last of us',
    'tlou2': 'the last of us part ii',
    'last of us': 'the last of us',
    'last of us 2': 'the last of us part ii',

    # Dragon Ball
    'dragon ball': 'dragon ball z',
    'dragon z': 'dragon ball fighterz',
    'dbz': 'dragon ball z',
    'dbz budokai': 'dragonal ball budokai tenkaichi',
    'dbz sparking zero': 'dragon ball sparking zero',

    # Harry Potter
    'hogwarts': 'hogwarts legacy',

    # Sekiro
    'sekiro': 'sekiro shadows die twice',

    # Call of Duty specific entries
    "cod mw": "call of duty modern warfare",
    "cod mw2": "call of duty modern warfare 2",
    "cod bo": "call of duty black ops",
    "cod bo2": "call of duty black ops ii",

    # PUBG variations
    "pubg": "playerunknowns battlegrounds",
    "bgmi": "battlegrounds mobile india",

    # Apex Legends
    "apex": "apex legends",

    # Valorant
    "valo": "valorant",
    "valorant": "valorant",

    # CS:GO
    "csgo": "counter strike global offensive",
    "cs 2": "counter strike 2",

    # League of Legends
    "lol": "league of legends",

    # DOTA 2
    "dota": "dota 2",

    # FIFA
    "fifa": "fifa 23",

    # PES (Pro Evolution Soccer)
    "pes": "efootball pes 2021",
    "efootball": "efootball 2023",

    # Elder Scrolls
    "skyrim": "the elder scrolls v skyrim",

    # Diablo
    "diablo 4": "diablo iv",

    # Far Cry
    "fc5": "far cry 5",
    "fc6": "far cry 6",

    # Hitman
    "hitman 3": "hitman 3",

    # Mass Effect
    "me": "mass effect",
    "me2": "mass effect 2",

    # Bioshock
    "bioshock": "bioshock infinite",

    # Doom
    "doom": "doom eternal",

    # Star Wars
    "jfo": "star wars jedi fallen order",
    "survivor": "star wars jedi survivor",

    # Borderlands
    "bl3": "borderlands 3",

    # Avengers
    "avengers": "Marvel's Avengers"
}

# ------------------------ Recommendation Functions -----------------------

       # ------------------------ Movies -----------------------
def recommend_movies(user_input, top_n=10):
    try:
        # Validate user input
        if not isinstance(user_input, str) or not user_input.lower().strip():
            raise ValueError("User input must not be empty. Please add a movie to get recommendations.")
        
        # Clean user input
        user_input_clean = re.sub(r'[^a-zA-Z0-9\s]', '', user_input.lower().strip())

        # Use alias if available
        if user_input_clean in movie_aliases:
            user_input_clean = movie_aliases[user_input_clean]

        # Handle case where no fuzzy match is found
        match_result = process.extractOne(user_input_clean, movies['title_clean'].to_list(), scorer=fuzz.ratio)
        if match_result is None:
            raise ValueError(f"Movie {user_input} is not updated in the data. It will be added in future update of application.")
        
        best_match = match_result[0]
        print(f"Best Match is: {best_match}")

        # Get index of the best match
        idx = movies[movies['title_clean'] == best_match].index[0]

        # Ensure that idx is valid
        if idx < 0 or idx >= len(movies):
            raise IndexError(f"Index {idx} is out of range.")
        
        # Calculate similarity scores directly from the precomputed cosine_sim matrix
        sim_scores = list(enumerate(movies_matrix[idx]))      # Use the precomputed cosine similarity matrix

        # Sort the similarity scores (excluding the movie itself)
        similar_movies_idx = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        # Prepare results
        results = []
        for i, _ in similar_movies_idx:
            movie_data = movies.loc[i]

            # Check if all necessary fields exist
            if any(field not in movie_data for field in ['title', 'top_cast','cast_profile_path', 'description', 'genres', 'languages', 'rating', 'poster_path', 'release_date', 'watch_link']):
                continue

            # Get trailer info if available:
            video_key = movie_data.get('video_key')
            trailer_url = f"https://www.youtube.com/watch?v={video_key}" if pd.notna(video_key) else None

            results.append({
                'Title': movies.loc[i,'title'],
                'Top Cast': movies.loc[i, 'top_cast'],
                'Cast Picture': movies.loc[i, 'cast_profile_path'],
                'Description': movies.loc[i,'description'],
                'Genre': movies.loc[i,'genres'],
                'Language': movies.loc[i,'languages'],
                'Release Date': movies.loc[i, 'release_date'],
                'Rating': movies.loc[i,'rating'],
                'Poster': movies.loc[i,'poster_path'],
                'Stream': movies.loc[i, 'watch_link'],
                'Trailer': trailer_url
            })


        return results
    
    except ValueError as ve:
        return {'Error': str(ve)}
    
    except IndexError as ie:
        return {'Error': f'Index error: {str(ie)}'}
    
    except Exception as e:
        return {'Error': f'An unexpected error occurred: {str(e)}'}
    
        # ------------------------ Games -----------------------
def recommend_games(user_input, top_n=10):
    try:
        # Validate user input
        if not isinstance(user_input, str) or not user_input.lower().strip():
            raise ValueError("User input must not be empty. Please add a game to get recommendations.")
        
        # Clean user input
        user_input_clean = re.sub(r'[^a-zA-Z0-9\s]', '', user_input.lower().strip())

        # Use alias if available
        if user_input_clean in alias_dict:
            user_input_clean = alias_dict[user_input_clean]

        # Handle case where no fuzzy match is found
        match_result = process.extractOne(user_input_clean, games['title_clean'].to_list(), scorer=fuzz.ratio)
        if match_result is None:
            raise ValueError(f"Movie {user_input} is not updated in the data. It will be added in future update of application.")
        
        best_match = match_result[0]
        print(f"Best Match is: {best_match}")

        # Get index of the best match
        idx = games[games['title_clean'] == best_match].index[0]

        # Ensure that idx is valid
        if idx < 0 or idx >= len(games):
            raise IndexError(f"Index {idx} is out of range.")
        
        # Calculate similarity scores directly from the precomputed cosine_sim matrix
        sim_scores = list(enumerate(games_matrix[idx]))      # Use the precomputed cosine similarity matrix

        # Sort the similarity scores (excluding the movie itself)
        similar_games_idx = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        # Prepare results
        results = []
        for i, _ in similar_games_idx:
            game_data = games.loc[i]
            # Safely split store names and domains into lists, or use empty lists
            store_names = game_data['store_name'].split(', ')
            store_domains = game_data['store_domain'].split(', ')

            # Zip only if valid and lengths match
            store_display = ', '.join([
                    f"{name} : https://{domain}" for name, domain in zip(store_names, store_domains)
                    ])
                
            # Check if all necessary fields exist
            if any(field not in game_data for field in ['title', 'description_clean','genres', 'release_date', 'rating', 'tags', 'developers', 'publishers', 'esrb_rating', 'background_image_url','website']):
                continue

            results.append({
                'Title': games.loc[i,'title'],
                'Description': games.loc[i, 'description_clean'],
                'Genre': games.loc[i, 'genres'],
                'Release Date': games.loc[i,'release_date'],
                'Rating': games.loc[i,'rating'],
                'Platforms': games.loc[i, 'platforms'],
                'Stores': store_display,
                'Tags': games.loc[i,'tags'],
                'Developer': games.loc[i, 'developers'],
                'Publisher': games.loc[i,'publishers'],
                'ESRB_Rating': games.loc[i,'esrb_rating'],
                'Poster': games.loc[i, 'background_image_url'],
                'Website': games.loc[i, 'website'],
                'Screenshots': games.loc[i, 'screenshots']
            })


        return results
    
    except ValueError as ve:
        return {'Error': str(ve)}
    
    except IndexError as ie:
        return {'Error': f'Index error: {str(ie)}'}
    
    except Exception as e:
        return {'Error': f'An unexpected error occurred: {str(e)}'}
    
# -------------------------------- Streamlit UI --------------------------------

# ---------- Custom CSS ----------
import streamlit as st
from streamlit_option_menu import option_menu

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #0F0F1C;
            padding-top: 20px;
        }

        .menu-container {
            background: linear-gradient(135deg, #00F260, #0575E6);
            padding: 15px;
            border-radius: 10px;
        }

        .stButton>button {
            border-radius: 8px;
            background-color: #8A2BE2;
            color: white;
            padding: 10px 24px;
            transition: 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #BA55D3;
            transform: scale(1.05);
        }

        h1, h2, h3 {
            color: white;
        }

        .header-container {
            background: #12122b;
            border-radius: 15px;
            padding: 10px 15px;
            text-align: left;
            color: white;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 12px;
            border: 2px solid #2f2f4f;
        }

        .header-container img {
            width: 50px;
            height: 50px;
        }

        .icon-text {
            font-size: 18px;
        }

        .css-1v0mbdj.ef3psqc12 {
            padding: 0px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Icon URLs ----------
home_icon = "https://cdn-icons-png.flaticon.com/512/1946/1946433.png"
movies_icon = "https://cdn-icons-png.flaticon.com/512/3959/3959330.png"
games_icon = "https://cdn-icons-png.flaticon.com/512/727/727399.png"
contact_icon = "https://cdn-icons-png.flaticon.com/512/732/732200.png"

# ---------- Header ----------
with st.sidebar:
    st.markdown(f"""
        <div class="header-container">
            <img src="{movies_icon}" />
            <span class="icon-text">Movies & Games<br>Recommendation Engine</span>
        </div>
    """, unsafe_allow_html=True)

    # ---------- Sidebar Menu ----------
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Recommend Movies", "Recommend Games", "Contact Me"],
        icons=["house", "film", "controller", "envelope"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0F0F1C"},
            "icon": {"color": "#FFD700", "font-size": "20px"},
            "nav-link": {
                "color": "white",
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px",
                "--hover-color": "#4B0082"
            },
            "nav-link-selected": {
                "background-color": "#8A2BE2",
                "color": "white",
                "font-weight": "bold"
            },
        }
    )

    st.markdown(
    """
    <hr style="border-color: #4B0082; margin-top: 20px; margin-bottom: 5px;">
    <p style="text-align: center; color: #ccc; font-size: 14px; margin-bottom: 8px;">
        Connect with me here
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; padding-bottom: 10px;">
        <a href="https://www.linkedin.com/in/hoshangsheth" target="_blank" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" width="28">
        </a>
        <a href="https://github.com/hoshangsheth" target="_blank" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" alt="GitHub" width="28">
        </a>
    </div>
    """,
    unsafe_allow_html=True
    )



# --------------------------- Home Page ----------------------------------
if selected == "Home":
        st.markdown(
        """
        <style>
        /* Apply background image only to main content area */
        [data-testid="stAppViewContainer"] {
            background-image: url('https://i.postimg.cc/hvkWbgX3/Chat-GPT-Image-May-16-2025-08-59-38-PM.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }

        /* Add a dark overlay using a box-shadow trick */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            box-shadow: inset 0 0 0 1000px rgba(0, 0, 0, 0.4);
            z-index: 0;
            pointer-events: none;
        }

        /* Raise content above overlay */
        [data-testid="stVerticalBlock"] {
            position: relative;
            z-index: 1;
        }

        /* Make default text light for visibility */
        .css-18e3th9, .css-1v0mbdj, .stTextInput>div>div>input {
            color: #eeeeee !important;
        }
        </style>
        """,
        unsafe_allow_html=True
        )

        st.markdown(
            """
            <h1 style="
                font-size: 4.7em;
                background: linear-gradient(to right, #6a0dad, #f5d300);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: bold;
                text-align: center;
            ">
                🎥 Movie & Game 🎮 Recommendation Engine
            </h1>
            """,
            unsafe_allow_html=True
        )

        with st.container():
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #2c003e, #f5d3001a);
                    border: 2px solid #6a0dad;
                    border-radius: 15px;
                    padding: 25px;
                    margin-top: 20px;
                    color: white;
                    font-size: 1.05rem;
                    box-shadow: 0 0 15px rgba(106, 13, 173, 0.4);
                    line-height: 1.8;
                ">
                This smart, easy-to-use platform helps you discover <b>movies</b> 🍿 and <b>games</b> 🕹️ you'll love, based on what you already enjoy.
                Whether you're into heart-pounding thrillers 💥, light-hearted comedies 🎭, or action-packed adventures ⚔️, we've got hand-picked suggestions just for you.
                </div>
                """,
                unsafe_allow_html=True
            )

        # What You Can Do Here - Two Columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #2c003e, #f5d3001a);
                    border: 2px solid #6a0dad;
                    border-radius: 15px;
                    padding: 20px;
                    margin-top: 20px;
                    color: white;
                    font-size: 1.05rem;
                    line-height: 1.7;
                    box-shadow: 0 0 10px rgba(106, 13, 173, 0.3);
                ">
                <h4 style='color:#FFD700;'>🔍︎ Find Movies You'll Love</h4>
                Just type the name of a movie you like, and we'll show you similar gems that match your taste.
                Our algorithm analyzes thousands of titles to find your perfect match.
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #2c003e, #f5d3001a);
                    border: 2px solid #6a0dad;
                    border-radius: 15px;
                    padding: 20px;
                    margin-top: 20px;
                    color: white;
                    font-size: 1.05rem;
                    line-height: 1.7;
                    box-shadow: 0 0 10px rgba(106, 13, 173, 0.3);
                ">
                <h4 style='color:#FFD700;'>모 Get Great Game Picks</h4>
                Not sure what to play next? Enter a game title, and we'll suggest exciting options across PC 💻, PS4 🎮, PS5🎮, and Xbox🎮 that match your gaming style.
                </div>
                """,
                unsafe_allow_html=True
            )

        # Why You'll Love It - Three Columns
        st.markdown("<br><h4 style='color:#FFD700;'>🕭 Why You'll Love It:</h4>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                """
                <div style="
                    background: #2c003e;
                    border: 2px solid #6a0dad;
                    border-radius: 12px;
                    padding: 15px;
                    color: white;
                    font-size: 1.02rem;
                    box-shadow: 0 0 10px rgba(255, 215, 0, 0.2);
                ">
                <b>⏱ Save Time</b><br>
                No more endless scrolling on Netflix or game stores — we save you time with personalized recommendations.
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                """
                <div style="
                    background: #2c003e;
                    border: 2px solid #6a0dad;
                    border-radius: 12px;
                    padding: 15px;
                    color: white;
                    font-size: 1.02rem;
                    box-shadow: 0 0 10px rgba(255, 215, 0, 0.2);
                ">
                <b>💭 Smart Search</b><br>
                Even if you misspell a name or only remember part of a title, we've got you covered with AI search engine.
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                """
                <div style="
                    background: #2c003e;
                    border: 2px solid #6a0dad;
                    border-radius: 12px;
                    padding: 15px;
                    color: white;
                    font-size: 1.02rem;
                    box-shadow: 0 0 10px rgba(255, 215, 0, 0.2);
                ">
                <b>🎞️ Complete Details</b><br>
                Every recommendation comes with trailers, posters, ratings, and direct links so you can dive right in.
                </div>
                """,
                unsafe_allow_html=True
            )

        # Simple and Smooth Container
        with st.container():
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #2c003e, #f5d3001a);
                    border: 2px solid #6a0dad;
                    border-radius: 15px;
                    padding: 20px;
                    margin-top: 30px;
                    color: white;
                    font-size: 1.05rem;
                    line-height: 1.8;
                    box-shadow: 0 0 12px rgba(106, 13, 173, 0.3);
                ">
                <b>✔ Simple and Smooth</b><br>
                Just use the menu on the left 🧭 to explore. It's quick, fun, and designed with your experience in mind.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("""
                        <style>
                        div.stButton > button {
                            background: linear-gradient(135deg, #6a0dad, #8b00ff);
                            color: white;
                            padding: 12px 24px;
                            font-size: 1.1rem;
                            border: 2px solid #f5d300;
                            border-radius: 12px;
                            box-shadow: 0 0 12px rgba(245, 211, 0, 0.3);
                            transition: all 0.3s ease-in-out;
                            width: 100%;
                        }
                        div.stButton > button:hover {
                            background: linear-gradient(135deg, #7a1de4, #a35bff);
                            transform: scale(1.03);
                            border-color: #ffe600;
                            cursor: pointer;
                        }
                        </style>
                        """, unsafe_allow_html=True)

# --- Page Title and Description ---     
elif selected == "Recommend Movies":
    st.markdown(
    """
    <style>
    /* Apply background image only to main content area */
    [data-testid="stAppViewContainer"] {
        background-image: url('https://user-images.githubusercontent.com/33485020/108069438-5ee79d80-7089-11eb-8264-08fdda7e0d11.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Add a dark overlay using a box-shadow trick */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        box-shadow: inset 0 0 0 1000px rgba(0, 0, 0, 0.4);
        z-index: 0;
        pointer-events: none;
    }

    /* Raise content above overlay */
    [data-testid="stVerticalBlock"] {
        position: relative;
        z-index: 1;
    }

    /* Make default text light for visibility */
    .css-18e3th9, .css-1v0mbdj, .stTextInput>div>div>input {
        color: #eeeeee !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("""
    <style>
    /* Style all buttons */
    div.stButton > button {
        background-color: #4B0082 !important;  /* Midnight Purple */
        color: white !important;
        border: none;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: white !important;
        color: black !important;
        cursor: pointer;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)


    st.title("🎬 **Lights. Camera. Action!**")
    st.markdown("""
    🎬 **Looking for your next binge-worthy flick?**
                
    **Tell us a movie you love, and we'll serve up a lineup of must-watch titles you'll vibe with.**
    **Whether you're into heart-racing thrillers, feel-good rom-coms, or mind-bending sci-fi — your next favorite film is just a click away.** 🍿🎞️
    
    **Lights down, sound up. Let's find your next obsession.** 🌟
    """)

    def render_rating_stars(rating):
            full_stars = int(rating)
            half_star = (rating - full_stars) >= 0.5

            stars = "⭐" * full_stars
            if half_star:
                stars += "✬"
            return stars

    # --- User Input ---
    user_input = st.text_input("Enter a Movie Title 🎥", placeholder="e.g. Avengers, Gladiator, Interstellar, ZNMD, Sooryavanshi...")
    
    if st.button("📽 Recommend Movies"):
        st.session_state.recommend_triggered = True
        st.session_state.user_movie_input = user_input
        st.session_state.selected_movie_index = None  # Reset dialog state

    if st.session_state.get("recommend_triggered", False):
        recommendations = recommend_movies(st.session_state.user_movie_input)

        if isinstance(recommendations, dict) and 'Error' in recommendations:
            st.error(recommendations['Error'])
        else:
            # Define the dialog function using decorator
            @st.dialog("🎬 Movie Details", width="large")
            def show_movie_details(movie):
                dcols = st.columns([1, 2])

                with dcols[0]:
                    st.image(movie['Poster'], use_container_width=True)

                with dcols[1]:
                    st.subheader(movie['Title'])
                    st.markdown(f"**Description:** {movie['Description']}")
                    st.markdown(f"**Genre:** {movie['Genre']}")
                    st.markdown(f"**Language:** {movie['Language']}")
                    st.markdown(f"**Release Date:** {movie['Release Date']}")
                    rating_num = round(float(movie['Rating']), 1)
                    stars = render_rating_stars(rating_num)
                    st.markdown(f"**Rating:** {rating_num}  {stars}")


                    st.markdown("**Top Cast:**")
                    cast_images = movie['Cast Picture'].split(',')
                    cast_names = movie['Top Cast'].split(',')
                    cast_cols = st.columns(len(cast_names))
                    for k, col in enumerate(cast_cols):
                        with col:
                            if k < len(cast_images):
                                st.image(cast_images[k], width=60)
                            st.caption(cast_names[k])

                    if movie['Stream']:
                        st.markdown(f"""
                            <a href="{movie['Stream']}" target="_blank" style="
                                display: inline-block;
                                background-color: #6C63FF;
                                color: white;
                                padding: 10px 20px;
                                border-radius: 10px;
                                text-decoration: none;
                                font-weight: bold;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                                transition: all 0.3s ease;
                            " onmouseover="this.style.backgroundColor='#4b47cc'" onmouseout="this.style.backgroundColor='#6C63FF'">
                                📺 Watch Now
                            </a>
                        """, unsafe_allow_html=True)
                    if movie['Trailer']:
                        st.video(movie['Trailer'])

            # --- Display movies grid with buttons ---
            num_cols = 4
            fixed_height = 400
            midnight_purple = "#6C63FF"  # midnight purple color
            for i in range(0, len(recommendations), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(recommendations):
                        movie = recommendations[idx]
                        with cols[j]:
                            #st.image(movie['Poster'], use_container_width=True)
                            # Custom img tag with fixed height and auto width to keep aspect ratio
                            st.markdown(f"""
                                <img src="{movie['Poster']}" 
                                    style="height: {fixed_height}px; width: auto; display: block; margin-left: auto; margin-right: auto; border-radius: 10px;" />
                            """, unsafe_allow_html=True)
                            # Title bar styled box
                            st.markdown(f"""
                                <div style="
                                    background-color: #dcdcdc;
                                    color: {midnight_purple};
                                    padding: 6px;
                                    border-radius: 8px;
                                    text-align: center;
                                    font-weight: 600;
                                    margin-top: 4px;
                                    margin-bottom: 6px;
                                    font-size: 14px;
                                    box-shadow: 0 2px 5px rgba(108, 99, 255, 0.3);
                                ">
                                    {movie['Title']}
                                </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"🛈 Details", key=f"button_{idx}"):
                                show_movie_details(movie)

elif selected == "Recommend Games":
    st.markdown(
    """
    <style>
    /* Apply background image only to main content area */
    [data-testid="stAppViewContainer"] {
        background-image: url('https://wallpapercave.com/wp/wp7816746.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Add a dark overlay using a box-shadow trick */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        box-shadow: inset 0 0 0 1000px rgba(0, 0, 0, 0.4);
        z-index: 0;
        pointer-events: none;
    }

    /* Raise content above overlay */
    [data-testid="stVerticalBlock"] {
        position: relative;
        z-index: 1;
    }

    /* Make default text light for visibility */
    .css-18e3th9, .css-1v0mbdj, .stTextInput>div>div>input {
        color: #eeeeee !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("""
    <style>
    /* Style all buttons */
    div.stButton > button {
        background-color: #4B0082 !important;  /* Midnight Purple */
        color: white !important;
        border: none;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: white !important;
        color: black !important;
        cursor: pointer;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🕹️ Ready. Set. Play!")
    st.markdown("""
                🎮 Looking for your next gaming adventure?

                Type in your favorite title — whether it's action-packed, story-rich, or multiplayer madness — and we'll drop a curated list you'll love.

                Let's level up your game list! 🕹️🔥
    """)

    user_input = st.text_input("Enter a Game Title 🎮", placeholder="e.g. Elden Ring, God of War, Spider-Man, Need for Speed...")

    if st.button("🎮 Recommend Games"):
        st.session_state.recommend_triggered_games = True
        st.session_state.user_game_input = user_input
        st.session_state.selected_game_index = None

    if st.session_state.get("recommend_triggered_games", False):
        recommendations = recommend_games(st.session_state.user_game_input)

        if isinstance(recommendations, dict) and 'Error' in recommendations:
            st.error(recommendations['Error'])
        else:
            @st.dialog("🎮 Game Details", width="large")
            def show_game_details(game):
                dcols = st.columns([1, 2])

                with dcols[0]:
                    st.image(game['Poster'], use_container_width=True)

                with dcols[1]:
                    st.subheader(game['Title'])
                    st.markdown(f"**Description:** {game['Description']}")
                    st.markdown(f"**Developer:** {game['Developer']}")
                    st.markdown(f"**Publisher:** {game['Publisher']}")
                    st.markdown(f"**Genre:** {game['Genre']}")
                    st.markdown(f"**Release Date:** {game['Release Date']}")
                    st.markdown(f"**Available On:** {game['Platforms']}")
                    st.markdown(f"**Tags:** {game['Tags']}")
                    st.markdown(f"**ESRB Rating:** {game['ESRB_Rating']}")

                    # --- Display top 3 store links as styled buttons ---
                    if game.get('Stores'):
                        stores_raw = game['Stores']
                        
                        # Split and parse the stores
                        store_links = []
                        for entry in stores_raw.split(','):
                            if ':' in entry:
                                name, url = entry.split(':', 1)
                                store_links.append((name.strip(), url.strip()))

                        # Show only top 3 stores
                        for name, url in store_links[:3]:
                            st.markdown(f"""
                                <a href="{url}" target="_blank" style="
                                    display: inline-block;
                                    background-color: #6C63FF;
                                    color: white;
                                    padding: 10px 20px;
                                    border-radius: 10px;
                                    text-decoration: none;
                                    font-weight: bold;
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                                    transition: all 0.3s ease;
                                    margin: 5px 10px 5px 0;
                                " onmouseover="this.style.backgroundColor='#4b47cc'" onmouseout="this.style.backgroundColor='#6C63FF'">
                                    🛒 {name}
                                </a>
                            """, unsafe_allow_html=True)


                    if game['Website']:
                        st.markdown(f"""
                                    <a href="{game['Website']}" target="_blank" style="
                                    display: inline-block;
                                    background-color: #6C63FF;
                                    color: white;
                                    padding: 10px 20px;
                                    border-radius: 10px;
                                    text-decoration: none;
                                    font-weight: bold;
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                                    transition: all 0.3s ease;
                                " onmouseover="this.style.backgroundColor='#4b47cc'" onmouseout="this.style.backgroundColor='#6C63FF'">
                                    🌐 Official Website
                                </a>
                            """, unsafe_allow_html=True)

                # Screenshot slideshow
                st.markdown("### 📸 Screenshots")
                if game.get("Screenshots"):
                    if "screenshot_index" not in st.session_state:
                        st.session_state.screenshot_index = 0

                    screenshots = game["Screenshots"].split(",")
                    total_screens = len(screenshots)
                    current_index = st.session_state.screenshot_index

                    st.image(screenshots[current_index], use_container_width=True)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("⬅️ Previous", key="prev_btn"):
                            st.session_state.screenshot_index = (current_index - 1) % total_screens
                    with col2:
                        if st.button("Next ➡️", key="next_btn"):
                            st.session_state.screenshot_index = (current_index + 1) % total_screens
                else:
                    st.info("No screenshots available.")


            # --- Display games grid with buttons ---
            fixed_height = 275
            midnight_purple = "#6C63FF"  # midnight purple color
            num_cols = 4
            for i in range(0, len(recommendations), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(recommendations):
                        game = recommendations[idx]
                        with cols[j]:
                            # Custom img tag with fixed height and auto width to keep aspect ratio
                            st.markdown(f"""
                                <img src="{game['Poster']}" 
                                    style="height: {fixed_height}px; width: auto; display: block; margin-left: auto; margin-right: auto; border-radius: 10px;" />
                            """, unsafe_allow_html=True)
                            # Title bar styled box
                            st.markdown(f"""
                                <div style="
                                    background-color: #dcdcdc;
                                    color: {midnight_purple};
                                    padding: 6px;
                                    border-radius: 8px;
                                    text-align: center;
                                    font-weight: 600;
                                    margin-top: 4px;
                                    margin-bottom: 6px;
                                    font-size: 14px;
                                    box-shadow: 0 2px 5px rgba(108, 99, 255, 0.3);
                                ">
                                    {game['Title']}
                                </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"🛈 Details", key=f"game_button_{idx}"):
                                show_game_details(game)

elif selected == "Contact Me":
    import streamlit as st
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    from datetime import datetime

    # Google Sheets save function
    def save_to_gsheet(name, email, message):
        scope = scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
        ]

        creds_dict = st.secrets['gcp_service_account']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        sheet = client.open_by_key("1dlXnan4bMdcbdoXngU_15u4A0OVI_m4uUnRew3traXY").worksheet("Sheet1")
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            email,
            message
        ])

    # Neon-styled contact form
    st.markdown(
        """
        <div class="neon-box">
            <h2 class="neon-heading">Let's Connect 🤝🏻</h2>
            <form action="" method="POST">
            </form>
        </div>

        <style>
            .neon-box {
                background-color: #0f0f0f;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 0 15px #6C63FF;
                max-width: 600px;
                margin: 50px auto 10px auto;
                text-align: center;
            }
            .neon-heading {
                color: #6C63FF;
                font-size: 2em;
                margin-bottom: 30px;
                text-shadow: 0 0 5px #6C63FF, 0 0 10px #6C63FF;
            }
            input, textarea {
                width: 90%;
                padding: 12px;
                margin: 10px 0;
                border: none;
                border-radius: 10px;
                background-color: #1c1c1c;
                color: white;
                font-size: 1em;
            }
            input::placeholder, textarea::placeholder {
                color: #888;
            }
            button {
                background-color: #6C63FF;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 10px;
                font-size: 1em;
                cursor: pointer;
                box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
                transition: all 0.3s ease;
            }
            button:hover {
                background-color: #4b47cc;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit form section (styled inputs from above HTML apply here)
    with st.form("Contact Form"):
        name = st.text_input("Your Name", placeholder="Enter your name...")
        user_email = st.text_input("Your Email", placeholder="Enter a valid email address...")
        message = st.text_area("Your Message", placeholder="Share your thoughts...")
        submitted = st.form_submit_button("➤ Send Message")

        if submitted:
            if name and user_email and message:
                try:
                    save_to_gsheet(name, user_email, message)
                    st.success("✔ Message sent successfully! I'll get back to you soon.")
                except Exception as e:
                    st.error("✖ Oops! Something went wrong.")
                    st.exception(e)
            else:
                st.warning("⚠ Please fill in all fields before submitting.")
