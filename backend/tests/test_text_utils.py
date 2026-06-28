from app.utils.text import clean_title, clean_user_input


def test_clean_title_strips_accents_and_punctuation():
    assert clean_title("Amélie!") == "amelie"


def test_clean_title_lowercases_and_collapses_whitespace():
    assert clean_title("The   Dark Knight") == "the dark knight"


def test_clean_user_input_strips_special_characters():
    assert clean_user_input("ZNMD!!") == "znmd"


def test_clean_user_input_trims_and_lowercases():
    assert clean_user_input("  Inception  ") == "inception"
