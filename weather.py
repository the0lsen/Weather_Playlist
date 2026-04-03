import requests


def get_location():
    """Use IP geolocation to get approximate lat/lon and city name."""
    try:
        res = requests.get("https://ipapi.co/json/", timeout=5).json()
        lat = res["latitude"]
        lon = res["longitude"]
        city = res.get("city", "Unknown")
        return lat, lon, city
    except Exception as e:
        raise RuntimeError(f"Could not determine location: {e}")


def get_weather(lat, lon):
    """Fetch current weather from Open-Meteo"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "precipitation_probability",
        "timezone": "auto",
    }
    try:
        data = requests.get(url, params=params, timeout=5).json()
        return data["current_weather"]
    except Exception as e:
        raise RuntimeError(f"Could not fetch weather: {e}")
