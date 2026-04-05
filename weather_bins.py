"""
weather_bins.py

Single source for all weather classification used incollect.py, train.py, and predict.py.

WMO Weather Interpretation Codes:
https://open-meteo.com/en/docs#weathervariables
"""
#Celsius 
TEMP_BINS = [
    (-999, 0,  "freezing"),   # <0°C
    (0,    8,  "very_cold"),  # 0–8°C
    (8,   14,  "cold"),       # 8–14°C
    (14,  18,  "cool"),       # 14–18°C
    (18,  22,  "mild"),       # 18–22°C
    (22,  27,  "warm"),       # 22–27°C
    (27,  33,  "hot"),        # 27–33°C
    (33, 999,  "scorching"),  # 33°C+
]


def bin_temperature(temp_c: float) -> str:
    for low, high, label in TEMP_BINS:
        if low <= temp_c < high:
            return label
    return "mild"



# WMO code -> granular weather condition

# Each entry: (wmo_code_or_range, label, description)
# Store as a list of (codes, label) to handle ranges.

WMO_CONDITIONS = [
    # Clear
    ([0],                       "clear"),

    # Mostly clear / light cloud
    ([1],                       "mostly_clear"),
    ([2],                       "partly_cloudy"),
    ([3],                       "overcast"),

    # Fog / mist
    ([45],                      "foggy"),
    ([48],                      "icy_fog"),          # rime ice fog

    # Light drizzle 
    ([51],                      "light_drizzle"),
    ([53],                      "drizzle"),
    ([55],                      "heavy_drizzle"),

    # Freezing drizzle
    ([56],                      "freezing_drizzle"),
    ([57],                      "heavy_freezing_drizzle"),

    # Rain — distinct from drizzle (larger drops, heavier)
    ([61],                      "light_rain"),
    ([63],                      "rain"),
    ([65],                      "pouring_rain"),

    # Freezing rain
    ([66],                      "light_freezing_rain"),
    ([67],                      "freezing_rain"),

    # Snow
    ([71],                      "light_snow"),
    ([73],                      "snow"),
    ([75],                      "heavy_snow"),
    ([77],                      "snow_grains"),

    # Rain showers (short bursts)
    ([80],                      "light_showers"),
    ([81],                      "showers"),
    ([82],                      "violent_showers"),

    # Snow showers
    ([85],                      "light_snow_showers"),
    ([86],                      "snow_showers"),

    # Thunderstorm
    ([95],                      "thunderstorm"),
    ([96],                      "thunderstorm_hail"),
    ([99],                      "severe_thunderstorm"),
]

# Build a flat lookup dict: wmo_code (int) -> label (str)
_WMO_LOOKUP: dict[int, str] = {}
for _codes, _label in WMO_CONDITIONS:
    for _c in _codes:
        _WMO_LOOKUP[_c] = _label


def bin_weather_code(wmo_code: int) -> str:
    """Return a weather condition label for a WMO code."""
    return _WMO_LOOKUP.get(wmo_code, "partly_cloudy")


# Wind speed bins (km/h)


WIND_BINS = [
    (0,   10,  "calm"),
    (10,  30,  "breezy"),
    (30,  55,  "windy"),
    (55, 999,  "stormy_wind"),
]


def bin_wind(speed_kmh: float) -> str:
    for low, high, label in WIND_BINS:
        if low <= speed_kmh < high:
            return label
    return "calm"



# Composite feature extraction


def extract_weather_features(weather: dict) -> dict:
    """
    Given Open-Meteo current_weather dict, return a dict of features.

    weather dict keys expected:
        temperature     float   (°C)
        weathercode     int     (WMO code)
        windspeed       float   (km/h)
        is_day          int     (1 = day, 0 = night)
    """
    temp      = weather["temperature"]
    code      = int(weather["weathercode"])
    wind      = weather.get("windspeed", 0.0)
    is_day    = int(weather.get("is_day", 1))

    condition = bin_weather_code(code)
    temp_bin  = bin_temperature(temp)
    wind_bin  = bin_wind(wind)

    # Broad condition family (for coarser grouping / display)
    family = _condition_family(condition)

    return {
        "condition":    condition,
        "temp_bin":     temp_bin,
        "wind_bin":     wind_bin,
        "is_day":       is_day,
        "family":       family,

        "temperature":  temp,
        "windspeed":    wind,
        "wmo_code":     code,
    }


def _condition_family(condition: str) -> str:
    """
    Map granular condition to a family for display use.
    """
    families = {
        "clear":              "clear",
        "mostly_clear":       "clear",
        "partly_cloudy":      "cloudy",
        "overcast":           "cloudy",
        "foggy":              "fog",
        "icy_fog":            "fog",
        "light_drizzle":      "drizzle",
        "drizzle":            "drizzle",
        "heavy_drizzle":      "drizzle",
        "freezing_drizzle":   "drizzle",
        "heavy_freezing_drizzle": "drizzle",
        "light_rain":         "rain",
        "rain":               "rain",
        "pouring_rain":       "rain",
        "light_freezing_rain":"rain",
        "freezing_rain":      "rain",
        "light_showers":      "rain",
        "showers":            "rain",
        "violent_showers":    "rain",
        "light_snow":         "snow",
        "snow":               "snow",
        "heavy_snow":         "snow",
        "snow_grains":        "snow",
        "light_snow_showers": "snow",
        "snow_showers":       "snow",
        "thunderstorm":       "storm",
        "thunderstorm_hail":  "storm",
        "severe_thunderstorm":"storm",
    }
    return families.get(condition, "cloudy")



# All unique categorical values — needed by train.py

ALL_CONDITIONS = [label for _, label in WMO_CONDITIONS]
ALL_TEMP_BINS  = [label for *_, label in TEMP_BINS]
ALL_WIND_BINS  = [label for *_, label in WIND_BINS]
ALL_FAMILIES   = ["clear", "cloudy", "fog", "drizzle", "rain", "snow", "storm"]

CATEGORICAL_FEATURES = ["condition", "temp_bin", "wind_bin", "family"]
NUMERIC_FEATURES     = ["temperature", "windspeed", "is_day"]
ALL_FEATURE_COLS     = CATEGORICAL_FEATURES + NUMERIC_FEATURES

