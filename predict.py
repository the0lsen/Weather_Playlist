"""
predict.py
----------
Load trained models and return genre + audio feature recommendations
for any given weather input.

This module is a drop-in replacement for map_weather_to_mood() in
your playlist script. Import predict_mood() instead.

Usage (standalone):
    python predict.py

Usage (in playlist script):
    from predict import predict_mood
    mood = predict_mood(weather_dict)
    # mood has same shape as map_weather_to_mood() output
"""

import os
import pickle
import json
import requests
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

from weather_bins import (
    extract_weather_features,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR        = Path("model")
GENRE_MODEL_PATH = MODEL_DIR / "genre_model.pkl"
META_PATH        = MODEL_DIR / "meta.json"

# Hardcoded audio targets per weather family.
# These are used always (genre model is ML, audio targets are curated rules).
# Keys match weather_bins.ALL_FAMILIES.
AUDIO_TARGETS_BY_FAMILY = {
    "clear":   {"valence": 0.78, "energy": 0.78, "danceability": 0.70, "tempo": 118, "acousticness": 0.25},
    "cloudy":  {"valence": 0.50, "energy": 0.45, "danceability": 0.50, "tempo": 100, "acousticness": 0.45},
    "fog":     {"valence": 0.32, "energy": 0.22, "danceability": 0.30, "tempo": 78,  "acousticness": 0.75},
    "drizzle": {"valence": 0.35, "energy": 0.28, "danceability": 0.32, "tempo": 86,  "acousticness": 0.70},
    "rain":    {"valence": 0.38, "energy": 0.38, "danceability": 0.38, "tempo": 92,  "acousticness": 0.55},
    "snow":    {"valence": 0.42, "energy": 0.18, "danceability": 0.28, "tempo": 74,  "acousticness": 0.80},
    "storm":   {"valence": 0.38, "energy": 0.92, "danceability": 0.55, "tempo": 142, "acousticness": 0.10},
}

# Fallback genres if model isn't trained yet
FALLBACK_GENRES = {
    "clear":   ["pop", "indie-pop", "summer"],
    "cloudy":  ["indie", "alternative", "chill"],
    "fog":     ["ambient", "post-rock", "classical"],
    "drizzle": ["lo-fi", "acoustic", "indie"],
    "rain":    ["indie", "singer-songwriter", "folk"],
    "snow":    ["classical", "ambient", "acoustic"],
    "storm":   ["metal", "electronic", "rock"],
}

# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_genre_model = None
_meta        = None


def _load_models():
    global _genre_model, _meta

    if _genre_model is not None:
        return True  # already loaded

    if not GENRE_MODEL_PATH.exists():
        return False

    try:
        with open(GENRE_MODEL_PATH, "rb") as f:
            _genre_model = pickle.load(f)
        if META_PATH.exists():
            with open(META_PATH) as f:
                _meta = json.load(f)
        return True
    except Exception as e:
        print(f"⚠️  Could not load models: {e}")
        return False

# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_mood(weather: dict, top_n: int = 3) -> dict:
    """
    Given a raw Open-Meteo current_weather dict, return a mood dict
    compatible with get_recommendations() in your playlist script.

    Genres come from the trained Random Forest (or fallback rules).
    Audio targets (valence, energy, tempo etc.) are curated per weather family.

    Args:
        weather:  dict with keys: temperature, weathercode, windspeed, is_day
        top_n:    number of genres to return

    Returns:
        {
            "genres":        ["indie", "lo-fi", ...],
            "valence":       0.35,
            "energy":        0.28,
            "danceability":  0.32,
            "tempo":         86.0,
            "acousticness":  0.70,
            "label":         "🌦️ drizzle / cold",
            "condition":     "drizzle",
            "temp_bin":      "cold",
            "source":        "model" | "fallback",
        }
    """
    features = extract_weather_features(weather)
    label    = _make_label(features)
    family   = features["family"]
    audio    = AUDIO_TARGETS_BY_FAMILY.get(family, AUDIO_TARGETS_BY_FAMILY["cloudy"])

    # Try ML genre prediction
    if _load_models():
        try:
            genres, genre_probs = _predict_genres(features, top_n)
            return {
                "genres":       genres,
                "genre_probs":  genre_probs,
                "label":        label,
                "condition":    features["condition"],
                "temp_bin":     features["temp_bin"],
                "source":       "model",
                **audio,
            }
        except Exception as e:
            print(f"⚠️  Genre model failed ({e}), using fallback.")

    # Fallback
    return {
        "genres":    FALLBACK_GENRES.get(family, ["pop", "chill"]),
        "label":     label,
        "condition": features["condition"],
        "temp_bin":  features["temp_bin"],
        "source":    "fallback",
        **audio,
    }


def _predict_genres(features: dict, top_n: int) -> tuple[list[str], dict]:
    """Run genre model and return (top_genres, genre_probs)."""
    X        = _features_to_df(features)
    gm       = _genre_model
    pipeline = gm["pipeline"]
    genres   = gm["genres"]

    proba_list    = pipeline.predict_proba(X)
    proba_present = np.array([p[0][1] for p in proba_list])
    ranked_idx    = np.argsort(proba_present)[::-1]
    top_genres    = [genres[i] for i in ranked_idx[:top_n]]
    top_probs     = {genres[i]: float(proba_present[i]) for i in ranked_idx[:top_n]}

    return top_genres, top_probs


def _features_to_df(features: dict) -> pd.DataFrame:
    """Convert feature dict to DataFrame row for sklearn pipeline."""
    row = {col: features.get(col, "unknown") for col in CATEGORICAL_FEATURES}
    row.update({col: features.get(col, 0.0) for col in NUMERIC_FEATURES})
    return pd.DataFrame([row])


def _make_label(features: dict) -> str:
    emoji_map = {
        "clear":           "☀️",
        "mostly_clear":    "🌤️",
        "partly_cloudy":   "⛅",
        "overcast":        "☁️",
        "foggy":           "🌫️",
        "icy_fog":         "🌫️❄️",
        "light_drizzle":   "🌦️",
        "drizzle":         "🌦️",
        "heavy_drizzle":   "🌧️",
        "freezing_drizzle":"🌧️❄️",
        "heavy_freezing_drizzle": "🌧️❄️",
        "light_rain":      "🌧️",
        "rain":            "🌧️",
        "pouring_rain":    "🌧️🌧️",
        "light_freezing_rain": "🌧️❄️",
        "freezing_rain":   "🌧️❄️",
        "light_showers":   "🌦️",
        "showers":         "🌧️",
        "violent_showers": "🌧️🌧️",
        "light_snow":      "🌨️",
        "snow":            "❄️",
        "heavy_snow":      "❄️❄️",
        "snow_grains":     "🌨️",
        "light_snow_showers": "🌨️",
        "snow_showers":    "❄️",
        "thunderstorm":    "⛈️",
        "thunderstorm_hail":"⛈️🧊",
        "severe_thunderstorm": "⛈️⚡",
    }
    emoji = emoji_map.get(features["condition"], "🌡️")
    return f"{emoji} {features['condition'].replace('_', ' ')} / {features['temp_bin'].replace('_', ' ')}"

# ---------------------------------------------------------------------------
# Live weather fetch (for standalone testing)
# ---------------------------------------------------------------------------

def fetch_current_weather(lat: float, lon: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "current_weather": True,
        "timezone":        "auto",
    }
    r = requests.get(url, params=params, timeout=5).json()
    return r["current_weather"]

# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def demo():
    print("""
╔══════════════════════════════════════════╗
║   🔮  Weather Genre — Predictor Demo    ║
╚══════════════════════════════════════════╝
""")

    # Check model status
    loaded = _load_models()
    if loaded and _meta:
        print(f"✅ Models loaded — trained on {_meta['n_rows']} rows "
              f"from {_meta['n_users']} user(s)\n")
    else:
        print("⚠️  No trained models found — using rule-based fallback.\n"
              "   Run collect.py then train.py to enable ML predictions.\n")

    # Fetch real weather at current location
    lat, lon, city = None, None, "Unknown"

    # 1. Manual override from .env
    lat_env = os.getenv("LATITUDE", "").strip()
    lon_env = os.getenv("LONGITUDE", "").strip()
    if lat_env and lon_env:
        lat, lon = float(lat_env), float(lon_env)
        city = os.getenv("CITY", "Manual location")

    # 2. ipapi.co
    if lat is None:
        try:
            r = requests.get("https://ipapi.co/json/", timeout=5).json()
            if r.get("latitude") and not r.get("error"):
                lat, lon, city = float(r["latitude"]), float(r["longitude"]), r.get("city", "Unknown")
        except Exception:
            pass

    # 3. ip-api.com fallback
    if lat is None:
        try:
            r = requests.get("http://ip-api.com/json/", timeout=5).json()
            if r.get("status") == "success":
                lat, lon, city = float(r["lat"]), float(r["lon"]), r.get("city", "Unknown")
        except Exception:
            pass

    if lat is not None:
        print(f"📍 Your location: {city}")
        try:
            weather = fetch_current_weather(lat, lon)
        except Exception:
            print("⚠️  Could not fetch live weather — using demo values.")
            weather = {"temperature": 12.0, "weathercode": 53, "windspeed": 18.0, "is_day": 1}
    else:
        print("📍 Could not detect location — using demo values.")
        weather = {"temperature": 12.0, "weathercode": 53, "windspeed": 18.0, "is_day": 1}

    mood = predict_mood(weather)

    print(f"\n🌡️  Weather:       {mood['label']}")
    print(f"📡 Source:         {mood['source']}")
    print(f"\n🎸 Top genres:     {', '.join(mood['genres'])}")
    if "genre_probs" in mood:
        for g, p in mood["genre_probs"].items():
            bar = "█" * int(p * 30)
            print(f"   {g:<25} {p:.2f} {bar}")
    print(f"\n🎚️  Audio targets:")
    print(f"   Valence:       {mood['valence']:.2f}  {'😊' if mood['valence'] > 0.6 else '😐' if mood['valence'] > 0.4 else '😔'}")
    print(f"   Energy:        {mood['energy']:.2f}  {'⚡' if mood['energy'] > 0.6 else '🌊' if mood['energy'] > 0.4 else '🍃'}")
    print(f"   Danceability:  {mood['danceability']:.2f}")
    print(f"   Tempo:         {mood['tempo']:.0f} BPM")
    print(f"   Acousticness:  {mood['acousticness']:.2f}")

    print("\n💡 To use in your playlist script, replace map_weather_to_mood() with:")
    print("   from predict import predict_mood")
    print("   mood = predict_mood(weather)\n")


if __name__ == "__main__":
    demo()
