"""
Static lookup tables used by the recommendation engines.

These are copied verbatim from the original `Deployment/app.py` — the
alias dictionaries are part of the "smart search" feature and must not be
altered, since they're tuned by hand against the actual dataset contents.
"""

# Movie title aliases (nicknames, abbreviations -> canonical title).
# Used by `app.recommender.movie_engine` before fuzzy matching.
MOVIE_ALIASES: dict[str, str] = {
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
    "nemo": "Finding Nemo",
}

# Game title aliases (nicknames, abbreviations -> canonical title, lowercase).
# Used by `app.recommender.game_engine` before fuzzy matching.
GAME_ALIASES: dict[str, str] = {
    # GTA
    "gta5": "grand theft auto v",
    "gta 5": "grand theft auto v",
    "gta v": "grand theft auto v",
    "gta": "grand theft auto",
    "gta 4": "grand theft auto iv",
    "gta4": "grand theft auto iv",

    # Witcher
    "witcher 3": "the witcher 3 wild hunt",
    "tw3": "the witcher 3 wild hunt",

    # Batman
    "batman": "batman arkham knight",

    # Uncharted
    "uncharted": "uncharted drakes fortune",
    "uncharted 4": "uncharted 4 a thiefs end",
    "uncharted lost": "uncharted lost legacy",

    # Test Drive
    "test drive": "test drive unlimited",

    # Forza
    "forza": "forza horizon 5",

    # Need for Speed
    "nfs": "need for speed",
    "nfs heat": "need for speed heat",
    "nfs unbound": "need for speed unbound",

    # Red Dead Redemption
    "rdr": "red dead redemption",
    "rdr2": "red dead redemption 2",
    "red dead 2": "red dead redemption 2",
    "red dead": "red dead redemption 2",

    # Zelda
    "botw": "the legend of zelda breath of the wild",
    "zelda botw": "the legend of zelda breath of the wild",

    # Elden Ring
    "elden": "elden ring",
    "elden ring": "elden ring",

    # God Of War
    "gow": "god of war",
    "god of war 4": "god of war",
    "god of war": "god of war",

    # Minecraft
    "minecraft": "minecraft",

    # Fortnite
    "fortnite": "fortnite",

    # Call Of Duty
    "cod": "call of duty",
    "call of duty": "call of duty",

    # Horizon
    "hzd": "horizon zero dawn",
    "horizon": "horizon zero dawn",

    # Spider-Man
    "spiderman": "marvels spider man",
    "spider man": "marvels spider man",
    "marvel spiderman": "marvels spider man",

    # Cyberpunk
    "cyberpunk": "cyberpunk 2077",
    "cyberpunk 2077": "cyberpunk 2077",

    # Assassin's Creed
    "ac valhalla": "assassins creed valhalla",
    "assassins creed valhalla": "assassins creed valhalla",
    "acv": "assassins creed valhalla",
    "ac": "assassins creed",
    "ac2": "assassins creed 2",

    # Resident Evil
    "re8": "resident evil village",
    "resident evil 8": "resident evil village",
    "village": "resident evil village",

    # Last Of Us
    "tlou": "the last of us",
    "tlou2": "the last of us part ii",
    "last of us": "the last of us",
    "last of us 2": "the last of us part ii",

    # Dragon Ball
    "dragon ball": "dragon ball z",
    "dragon z": "dragon ball fighterz",
    "dbz": "dragon ball z",
    "dbz budokai": "dragonal ball budokai tenkaichi",
    "dbz sparking zero": "dragon ball sparking zero",

    # Harry Potter
    "hogwarts": "hogwarts legacy",

    # Sekiro
    "sekiro": "sekiro shadows die twice",

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
    "avengers": "Marvel's Avengers",
}

# Required fields for a movie row to be eligible for the response payload.
# Mirrors the field-completeness check in the original `recommend_movies()`.
REQUIRED_MOVIE_FIELDS: list[str] = [
    "title",
    "top_cast",
    "cast_profile_path",
    "description",
    "genres",
    "languages",
    "rating",
    "poster_path",
    "release_date",
    "watch_link",
]

# Required fields for a game row to be eligible for the response payload.
# Mirrors the field-completeness check in the original `recommend_games()`.
REQUIRED_GAME_FIELDS: list[str] = [
    "title",
    "description_clean",
    "genres",
    "release_date",
    "rating",
    "tags",
    "developers",
    "publishers",
    "esrb_rating",
    "background_image_url",
    "website",
]
