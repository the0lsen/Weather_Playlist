

import os
import csv
import time
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from weather_bins import extract_weather_features

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = Path("dataset.csv")
CACHE_PATH   = ".spotify_cache_collect"

SCOPES = " ".join([
    "user-read-recently-played",
    "user-top-read",
    "user-read-private",
])

DATASET_COLUMNS = [
    # Identity
    "user_id", "collected_at", "played_at",
    # Track info
    "track_id", "track_name", "artist_id", "artist_name",
    # Genres (semicolon-separated list)
    "genres",
    # Weather features
    "condition", "temp_bin", "wind_bin", "family", "is_day",
    "temperature", "windspeed", "wmo_code",
    # Location
    "country", "city", "latitude", "longitude",
]

# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

def get_spotify() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback"),
        scope=SCOPES,
        cache_path=CACHE_PATH,
        open_browser=True,
    ))


def get_recently_played(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    """Return up to 50 recently played tracks with timestamps."""
    results = sp.current_user_recently_played(limit=limit)
    return results.get("items", [])



def _musicbrainz_genres(artist_name: str, top_n: int = 5) -> list[str]:
    """Fetch genre tags for an artist from MusicBrainz (no API key required).
    Makes two requests (search + detail) with a 1.1 s sleep between each.
    Raises exceptions on network/API errors so the caller can log them.
    """
    headers = {"User-Agent": "weather-genre-model/1.0 (local-app)"}

    search_r = requests.get(
        "https://musicbrainz.org/ws/2/artist/",
        params={"query": artist_name, "limit": 1, "fmt": "json"},
        headers=headers,
        timeout=10,
    ).json()
    artists = search_r.get("artists", [])
    if not artists:
        return []
    mbid = artists[0]["id"]
    time.sleep(1.1)  # respect 1 req/sec rate limit between the two requests

    detail_r = requests.get(
        f"https://musicbrainz.org/ws/2/artist/{mbid}",
        params={"inc": "genres", "fmt": "json"},
        headers=headers,
        timeout=10,
    ).json()
    raw = detail_r.get("genres", [])
    raw_sorted = sorted(raw, key=lambda g: g.get("count", 0), reverse=True)
    return [g["name"].lower() for g in raw_sorted[:top_n]]


def _lastfm_genres(artist_name: str, api_key: str, top_n: int = 5) -> list[str]:
    """Fetch top tags for an artist from Last.fm (used as genre labels)."""
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method":  "artist.getTopTags",
                "artist":  artist_name,
                "api_key": api_key,
                "format":  "json",
            },
            timeout=5,
        ).json()
        tags = r.get("toptags", {}).get("tag", [])
        genres = [
            t["name"].lower() for t in tags
            if len(t["name"]) > 2 and not t["name"].isdigit()
        ]
        return genres[:top_n]
    except Exception:
        return []


def get_artist_genres(
    sp: spotipy.Spotify,
    artist_ids: list[str],
    id_to_name: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Fetch genres for each artist ID. Returns dict keyed by artist_id.

    Resolution order:
      1. Spotify batch /artists endpoint
      2. Spotify individual /artists/{id} endpoint
      3. Spotify search by artist name
      4. Last.fm artist.getTopTags  (if LASTFM_API_KEY is set in env)
      5. MusicBrainz genre tags     (no key required)
      6. Empty list → caller records as 'unknown'
    """
    out = {}
    unique = list(set(artist_ids))

    # --- attempt 1: Spotify batch ---
    batch_failed = False
    for i in range(0, len(unique), 50):
        batch = unique[i:i+50]
        try:
            results = sp.artists(batch)
            for artist in results.get("artists", []):
                if artist:
                    out[artist["id"]] = artist.get("genres", [])
        except Exception as e:
            print(f"  ⚠️  Spotify batch artist lookup failed ({e.__class__.__name__}). "
                  "Trying individual lookups...")
            batch_failed = True
            break

    if not batch_failed:
        return out

    # --- attempt 2: Spotify individual ---
    remaining = [aid for aid in unique if aid not in out]
    individual_failed = False
    for aid in remaining:
        try:
            artist = sp.artist(aid)
            out[aid] = artist.get("genres", [])
            time.sleep(0.1)
        except Exception:
            individual_failed = True
            break

    if not individual_failed:
        # Individual lookups worked but genres may still be empty (Spotify's
        # genre field is sparse). Fall through to enrich artists with no genres.
        pass

    # --- attempt 3: Spotify search by name ---
    if id_to_name:
        remaining = [aid for aid in unique if aid not in out]
        print(f"  ℹ️  /artists endpoint restricted. Trying Spotify search for "
              f"{len(remaining)} artists...")
        search_ok = True
        spotify_fetched = 0
        for aid in remaining:
            name = id_to_name.get(aid)
            if not name:
                continue
            try:
                results = sp.search(q=f"artist:{name}", type="artist", limit=1)
                items = results.get("artists", {}).get("items", [])
                genres = items[0].get("genres", []) if items else []
                out[aid] = genres
                if genres:
                    spotify_fetched += 1
                time.sleep(0.1)
            except Exception:
                search_ok = False
                break

        if search_ok:
            print(f"  ✅ Spotify search: genres found for {spotify_fetched}/{len(remaining)} artists.")
        else:
            print("  ⚠️  Spotify search also restricted.")
        # Always fall through — enrich artists that still have empty genres

    # --- attempt 4: Last.fm (if key set) for artists still missing genres ---
    lastfm_key = os.getenv("LASTFM_API_KEY", "").strip()
    if lastfm_key and id_to_name:
        missing = [aid for aid in unique if not out.get(aid)]
        if missing:
            print(f"  ℹ️  Trying Last.fm for {len(missing)} artists without genres...")
            fetched = 0
            for aid in missing:
                name = id_to_name.get(aid)
                if not name:
                    continue
                genres = _lastfm_genres(name, lastfm_key)
                if genres:
                    out[aid] = genres
                    fetched += 1
                time.sleep(0.15)
            print(f"  ✅ Last.fm: genres found for {fetched}/{len(missing)} artists.")

    # --- attempt 5: MusicBrainz for artists still missing genres (no key needed) ---
    if id_to_name:
        missing = [aid for aid in unique if not out.get(aid)]
        if missing:
            print(f"  ℹ️  Trying MusicBrainz for {len(missing)} artists without genres...")
            fetched = 0
            for aid in missing:
                name = id_to_name.get(aid)
                if not name:
                    continue
                try:
                    genres = _musicbrainz_genres(name)
                    out[aid] = genres
                    if genres:
                        fetched += 1
                        print(f"     {name}: {', '.join(genres)}")
                    else:
                        print(f"     {name}: (no genres in MusicBrainz)")
                except Exception as e:
                    print(f"     {name}: MusicBrainz error — {e}")
                time.sleep(1.1)  # MusicBrainz rate limit: 1 req/sec
            print(f"  ✅ MusicBrainz: genres found for {fetched}/{len(missing)} artists.")

    return out

# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def get_location() -> dict:
    """IP-based geolocation. Returns lat, lon, city, country.

    Resolution order:
      1. LATITUDE / LONGITUDE / CITY / COUNTRY env vars (manual override)
      2. ipapi.co
      3. ip-api.com (fallback)
    """
    # Manual override via .env
    lat_env = os.getenv("LATITUDE", "").strip()
    lon_env = os.getenv("LONGITUDE", "").strip()
    if lat_env and lon_env:
        return {
            "latitude":  float(lat_env),
            "longitude": float(lon_env),
            "city":      os.getenv("CITY", "Manual"),
            "country":   os.getenv("COUNTRY", "XX"),
        }

    # ipapi.co
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5).json()
        if r.get("latitude") and not r.get("error"):
            return {
                "latitude":  float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "city":      r.get("city", "Unknown"),
                "country":   r.get("country_code", "XX"),
            }
        reason = r.get("reason", "no location in response")
        print(f"  ⚠️  ipapi.co: {reason}. Trying fallback...")
    except Exception as e:
        print(f"  ⚠️  ipapi.co failed: {e}. Trying fallback...")

    # ip-api.com fallback (free, no key needed, 45 req/min)
    try:
        r = requests.get("http://ip-api.com/json/", timeout=5).json()
        if r.get("status") == "success":
            return {
                "latitude":  float(r["lat"]),
                "longitude": float(r["lon"]),
                "city":      r.get("city", "Unknown"),
                "country":   r.get("countryCode", "XX"),
            }
        print(f"  ⚠️  ip-api.com: {r.get('message', 'failed')}.")
    except Exception as e:
        print(f"  ⚠️  ip-api.com failed: {e}.")

    print("  ❌ Could not detect location. Set LATITUDE and LONGITUDE in .env to fix this.")
    return {"latitude": 0.0, "longitude": 0.0, "city": "Unknown", "country": "XX"}

# ---------------------------------------------------------------------------
# Weather helpers
# ---------------------------------------------------------------------------

def fetch_historical_weather(lat: float, lon: float, date_str: str) -> dict | None:
    """
    Fetch hourly weather for a specific date from Open-Meteo historical API.
    date_str format: 'YYYY-MM-DD'
    Returns a dict of hour -> weather dict, or None on failure.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": "temperature_2m,weathercode,windspeed_10m,is_day",
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=10).json()
        hourly = r.get("hourly", {})
        times      = hourly.get("time", [])
        temps      = hourly.get("temperature_2m", [])
        codes      = hourly.get("weathercode", [])
        winds      = hourly.get("windspeed_10m", [])
        is_days    = hourly.get("is_day", [])

        result = {}
        for i, t in enumerate(times):
            hour = t[11:13]  # "HH" from "YYYY-MM-DDTHH:00"
            result[hour] = {
                "temperature": temps[i] if i < len(temps) else 15.0,
                "weathercode": codes[i] if i < len(codes) else 0,
                "windspeed":   winds[i] if i < len(winds) else 0.0,
                "is_day":      is_days[i] if i < len(is_days) else 1,
            }
        return result
    except Exception as e:
        print(f"  ⚠️  Weather fetch failed for {date_str}: {e}")
        return None


def get_weather_for_timestamp(lat: float, lon: float, played_at: str) -> dict | None:
    """
    Given an ISO8601 played_at timestamp and coordinates, return weather features.
    played_at example: '2024-03-15T14:22:31.000Z'
    """
    dt = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
    date_str = dt.strftime("%Y-%m-%d")
    hour_str = dt.strftime("%H")

    hourly = fetch_historical_weather(lat, lon, date_str)
    if not hourly:
        return None

    weather_raw = hourly.get(hour_str)
    if not weather_raw:
        # Fall back to closest available hour
        weather_raw = list(hourly.values())[0]

    return extract_weather_features(weather_raw)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_existing_ids() -> set[str]:
    """Load set of (user_id, track_id, played_at) combos already in dataset."""
    if not DATASET_PATH.exists():
        return set()
    seen = set()
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(f"{row['user_id']}|{row['track_id']}|{row['played_at']}")
    return seen


def append_rows(rows: list[dict]) -> None:
    """Append rows to dataset.csv, creating it with headers if needed."""
    is_new = not DATASET_PATH.exists()
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def collect():
    print("""
╔══════════════════════════════════════════╗
║   🎵  Weather Genre — Data Collector    ║
╚══════════════════════════════════════════╝
""")

    # Auth
    print("🔐 Connecting to Spotify...")
    sp = get_spotify()
    user    = sp.current_user()
    user_id = user["id"]
    print(f"👤 Logged in as: {user['display_name']} ({user_id})\n")

    # Location
    print("📍 Detecting location...")
    loc = get_location()
    print(f"📍 {loc['city']}, {loc['country']} ({loc['latitude']:.2f}, {loc['longitude']:.2f})\n")

    # Recently played
    print("🎧 Fetching recent listening history (up to 50 tracks)...")
    items = get_recently_played(sp, limit=50)
    if not items:
        print("❌ No recent tracks found.")
        return
    print(f"   Found {len(items)} recent tracks.\n")

    # Deduplicate against existing dataset
    existing = load_existing_ids()
    items = [
        item for item in items
        if f"{user_id}|{item['track']['id']}|{item['played_at']}" not in existing
    ]
    if not items:
        print("✅ No new tracks to add — all already in dataset.")
        return
    print(f"   {len(items)} new tracks to process.\n")

    # Batch-fetch artist genres
    artist_ids   = [item["track"]["artists"][0]["id"]   for item in items]
    id_to_name   = {
        item["track"]["artists"][0]["id"]: item["track"]["artists"][0]["name"]
        for item in items
    }

    print("🏷️  Fetching artist genres...")
    artist_genres = get_artist_genres(sp, artist_ids, id_to_name=id_to_name)

    # Build rows
    print("🌤️  Fetching historical weather for each track timestamp...")
    rows = []
    now  = datetime.now(timezone.utc).isoformat()

    # Cache weather by (date, hour) to avoid redundant API calls
    weather_cache: dict[str, dict | None] = {}

    for i, item in enumerate(items):
        track     = item["track"]
        played_at = item["played_at"]
        track_id  = track["id"]
        artist    = track["artists"][0]

        # Weather — cache by date+hour to limit API calls
        dt       = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
        cache_key = f"{dt.strftime('%Y-%m-%d')}_{dt.strftime('%H')}"
        if cache_key not in weather_cache:
            weather_cache[cache_key] = get_weather_for_timestamp(
                loc["latitude"], loc["longitude"], played_at
            )
            time.sleep(0.3)  # gentle rate limiting

        weather = weather_cache[cache_key]
        if weather is None:
            print(f"  ⚠️  Skipping {track['name']} — no weather data.")
            continue

        # Genres
        genres = artist_genres.get(artist["id"], [])
        genres_str = ";".join(genres) if genres else "unknown"

        row = {
            "user_id":      user_id,
            "collected_at": now,
            "played_at":    played_at,
            "track_id":     track_id,
            "track_name":   track["name"],
            "artist_id":    artist["id"],
            "artist_name":  artist["name"],
            "genres":       genres_str,
            # Weather
            "condition":    weather["condition"],
            "temp_bin":     weather["temp_bin"],
            "wind_bin":     weather["wind_bin"],
            "family":       weather["family"],
            "is_day":       weather["is_day"],
            "temperature":  weather["temperature"],
            "windspeed":    weather["windspeed"],
            "wmo_code":     weather["wmo_code"],
            # Location
            "country":      loc["country"],
            "city":         loc["city"],
            "latitude":     loc["latitude"],
            "longitude":    loc["longitude"],
        }
        rows.append(row)
        print(f"  ✅ [{i+1}/{len(items)}] {track['name']} — {weather['condition']} / {weather['temp_bin']}")

    # Save
    if rows:
        append_rows(rows)
        print(f"\n💾 Added {len(rows)} rows to {DATASET_PATH}")
        print(f"   Total dataset size: {sum(1 for _ in open(DATASET_PATH)) - 1} rows")
    else:
        print("\n⚠️  No rows were saved.")

    print("\n✅ Collection complete. Run again later or share with friends to build more data.\n")


if __name__ == "__main__":
    collect()

