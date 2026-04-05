"""
train.py
--------
Train a genre model on the collected dataset:

  Genre model  — Random Forest multi-label classifier
                 Input:  weather features
                 Output: probability score for each genre

Note: Audio feature targets (valence, energy, tempo etc.)
are handled by hardcoded per-weather-family fallbacks in predict.py.

Usage:
    python train.py
"""

import csv
import pickle
import warnings
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from weather_bins import CATEGORICAL_FEATURES, NUMERIC_FEATURES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH     = Path("dataset.csv")
MODEL_DIR        = Path("model")
GENRE_MODEL_PATH = MODEL_DIR / "genre_model.pkl"
META_PATH        = MODEL_DIR / "meta.json"

# Minimum number of rows to attempt training
MIN_ROWS = 20

# Genres that appear fewer than this many times will be dropped
MIN_GENRE_COUNT = 3


# Data loading


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run collect.py first."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"📂 Loaded {len(df)} rows from {DATASET_PATH}")
    return df


def parse_genres(genre_str: str) -> list[str]:
    if not genre_str or genre_str == "unknown":
        return []
    return [g.strip() for g in genre_str.split(";") if g.strip()]


def filter_rare_genres(genre_lists: list[list[str]], min_count: int) -> list[list[str]]:
    """Remove genres that appear fewer than min_count times across all rows."""
    counter: Counter = Counter()
    for genres in genre_lists:
        counter.update(genres)
    valid = {g for g, c in counter.items() if c >= min_count}
    return [[g for g in genres if g in valid] for genres in genre_lists]


# Feature engineering

def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


# Training

def train_genre_model(X: pd.DataFrame, genre_lists: list[list[str]]):
    """
    Train a Random Forest multi-label genre classifier.
    Returns (pipeline, mlb, genre_names).
    """
    # Binarize genres
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(genre_lists)

    if y.shape[1] == 0:
        raise ValueError("No genres remain after filtering. Collect more data.")

    genre_names = list(mlb.classes_)
    print(f"\n🏷️  Training genre model on {len(genre_names)} genres:")
    for g in sorted(genre_names):
        print(f"     • {g}")

    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])

    pipeline.fit(X, y)

    # Quick cross-val score on a single label for sanity check
    if len(genre_names) > 0:
        scores = cross_val_score(
            Pipeline([("preprocessor", build_preprocessor()),
                      ("classifier",   RandomForestClassifier(n_estimators=50, random_state=42))]),
            X, y[:, 0], cv=min(3, len(X)), scoring="f1_weighted"
        )
        print(f"\n   Cross-val F1 (genre[0]): {scores.mean():.3f} ± {scores.std():.3f}")

    return pipeline, mlb, genre_names



# Feature importance display


def print_feature_importances(genre_pipeline, genre_names: list[str], X: pd.DataFrame):
    """Print top weather features driving genre predictions."""
    clf   = genre_pipeline.named_steps["classifier"]
    prep  = genre_pipeline.named_steps["preprocessor"]

    try:
        ohe_features = prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
        all_features = list(ohe_features) + NUMERIC_FEATURES
        importances  = clf.feature_importances_

        pairs = sorted(zip(all_features, importances), key=lambda x: -x[1])
        print("\n🔍 Top 10 weather features driving genre predictions:")
        for feat, imp in pairs[:10]:
            bar = "█" * int(imp * 200)
            print(f"   {feat:<35} {imp:.4f} {bar}")
    except Exception:
        pass  # Non-critical


# Main


def train():
    print("""
╔══════════════════════════════════════════╗
║     Weather Genre — Model Trainer     ║
╚══════════════════════════════════════════╝
""")

    MODEL_DIR.mkdir(exist_ok=True)

    df = load_dataset()

    if len(df) < MIN_ROWS:
        print(f"  Only {len(df)} rows — need at least {MIN_ROWS} to train reliably.")
        print("   Run collect.py more times or share with others to gather data.")
        return

    # Feature matrix
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("unknown")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    # --- Genre model ---
    genre_lists = [parse_genres(g) for g in df["genres"].astype(str)]
    genre_lists = filter_rare_genres(genre_lists, MIN_GENRE_COUNT)

    genre_pipeline, mlb, genre_names = train_genre_model(X, genre_lists)

    with open(GENRE_MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": genre_pipeline, "mlb": mlb, "genres": genre_names}, f)
    print(f"\n Genre model saved → {GENRE_MODEL_PATH}")

    # --- Metadata ---
    meta = {
        "trained_at":  pd.Timestamp.now().isoformat(),
        "n_rows":      len(df),
        "n_users":     df["user_id"].nunique(),
        "n_genres":    len(genre_names),
        "genre_names": genre_names,
        "countries":   df["country"].dropna().unique().tolist(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f" Metadata saved    → {META_PATH}")

    # Feature importances
    print_feature_importances(genre_pipeline, genre_names, X)

    print(f"""
✅ Training complete.
   Rows used:  {len(df)}
   Users:      {df['user_id'].nunique()}
   Genres:     {len(genre_names)}

   upgrade to librosa later if you want ML-based audio targets.

Run predict.py to test, or import predict_mood() in your playlist script.
""")


if __name__ == "__main__":
    train()

