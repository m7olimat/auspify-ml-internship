"""
Task 1 - Netflix content recommendation system

Content-based recommender: every title is turned into a small "bag of tokens"
(genres, type, rating, country, director), vectorised with TF-IDF, and titles
are ranked by cosine similarity.

Run:
    python task1_recommender.py
    python task1_recommender.py --title "Narcos" --n 8
"""
import argparse
import ast
import difflib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, we only save the figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "netflix_clean.csv"
OUT_DIR = BASE_DIR / "outputs" / "task1"

SAMPLE_TITLES = ["Stranger Things", "The Crown", "Narcos", "Dangal", "Kung Fu Panda"]

# how many times each kind of token is repeated before TF-IDF.
# genres matter most, director the least.
WEIGHTS = {
    "genres only":            {"genre": 1},
    "genres + type + rating": {"genre": 3, "type": 2, "rating": 1},
    "all attributes":         {"genre": 3, "type": 2, "rating": 1, "country": 1, "director": 1},
}
DEFAULT_VARIANT = "all attributes"

# rough audience buckets, only used for evaluation
RATING_GROUP = {
    "TV-Y": "kids", "TV-Y7": "kids", "TV-Y7-FV": "kids", "TV-G": "kids", "G": "kids",
    "TV-PG": "family/teen", "PG": "family/teen", "TV-14": "family/teen", "PG-13": "family/teen",
    "TV-MA": "adult", "R": "adult", "NC-17": "adult",
    "NR": "unrated", "UR": "unrated",
}


# ---------------------------------------------------------------- data prep

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["title"] = df["title"].str.strip()

    # these two columns were saved as strings like "['Drama', 'Comedy']"
    df["genres"] = df["genres"].apply(ast.literal_eval)
    df["countries"] = df["countries"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
    df["directors"] = df["director"].fillna("").apply(
        lambda s: [d.strip() for d in s.split(",") if d.strip()]
    )
    df["rating_group"] = df["rating"].map(RATING_GROUP).fillna("unrated")
    return df.reset_index(drop=True)


def clean_token(prefix, value):
    """'Children & Family Movies' -> 'genre_children_family_movies'"""
    value = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return f"{prefix}_{value}"


def build_soup(df, weights):
    """One string of tokens per title, repeating tokens according to their weight."""
    soups = []
    for row in df.itertuples():
        tokens = []
        tokens += [clean_token("genre", g) for g in row.genres] * weights.get("genre", 0)
        tokens += [clean_token("type", row.type)] * weights.get("type", 0)
        tokens += [clean_token("rating", row.rating)] * weights.get("rating", 0)
        tokens += [clean_token("country", c) for c in row.countries] * weights.get("country", 0)
        tokens += [clean_token("director", d) for d in row.directors] * weights.get("director", 0)
        soups.append(" ".join(tokens))
    return soups


# -------------------------------------------------------------- recommender

class ContentRecommender:
    def __init__(self, df, weights):
        self.df = df
        soup = build_soup(df, weights)
        # token_pattern=\S+ keeps our tokens intact (default would split on underscores)
        self.vectorizer = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
        self.matrix = self.vectorizer.fit_transform(soup)  # rows are L2-normalised

    def find_title(self, title):
        hits = self.df.index[self.df["title"].str.lower() == title.strip().lower()]
        if len(hits) == 0:
            close = difflib.get_close_matches(title, self.df["title"].tolist(), n=5, cutoff=0.6)
            msg = f"Title '{title}' was not found."
            if close:
                msg += " Did you mean: " + ", ".join(close) + "?"
            raise ValueError(msg)
        return hits[0]

    def recommend(self, title, n=10):
        idx = self.find_title(title)
        scores = linear_kernel(self.matrix[idx], self.matrix).ravel()

        # a few titles show up twice in the dataset, don't recommend a title to itself
        scores[(self.df["title"] == self.df.at[idx, "title"]).to_numpy()] = -1
        best = np.argsort(-scores, kind="stable")[:n]

        out = self.df.loc[best, ["title", "type", "rating", "listed_in", "main_country"]].copy()
        out.insert(0, "similarity", scores[best].round(3))
        return out.reset_index(drop=True)

    def top_k_for_everything(self, k=10, chunk=500):
        """Top-k neighbour indices for every title (done in chunks to keep memory small)."""
        n = self.matrix.shape[0]
        top = np.zeros((n, k), dtype=int)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            sims = linear_kernel(self.matrix[start:stop], self.matrix)
            sims[np.arange(stop - start), np.arange(start, stop)] = -1  # ignore itself
            top[start:stop] = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return top


# --------------------------------------------------------------- evaluation
# There is no "right answer" for a recommender here (no user data), so I check
# how consistent the recommended titles are with the title they came from.

def genre_matrix(df):
    vocab = sorted({g for gs in df["genres"] for g in gs})
    col = {g: i for i, g in enumerate(vocab)}
    rows, cols = [], []
    for i, gs in enumerate(df["genres"]):
        for g in gs:
            rows.append(i)
            cols.append(col[g])
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(vocab)))


def same_value_rate(values, top, skip_missing=False):
    codes, _ = pd.factorize(values)  # missing values get code -1
    same = codes[:, None] == codes[top]
    if skip_missing:
        same &= (codes[:, None] >= 0)
    return same.mean()


def score_recommendations(df, top):
    G = genre_matrix(df)
    sizes = np.asarray(G.sum(axis=1)).ravel()
    n, k = top.shape

    inter = np.zeros((n, k))
    for r in range(k):
        inter[:, r] = np.asarray(G.multiply(G[top[:, r]]).sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[top] - inter

    return {
        "genre_jaccard": (inter / union).mean(),
        "shares_a_genre": (inter > 0).mean(),
        "same_type": same_value_rate(df["type"], top),
        "same_rating_group": same_value_rate(df["rating_group"], top),
        "same_country": same_value_rate(df["main_country"], top, skip_missing=True),
        "coverage": len(np.unique(top)) / n,
    }


def random_top(n, k, seed=42):
    rng = np.random.default_rng(seed)
    top = rng.integers(0, n - 1, size=(n, k))
    top += top >= np.arange(n)[:, None]  # shift so a title can't pick itself
    return top


def run_evaluation(df, k=10):
    results = {"random baseline": score_recommendations(df, random_top(len(df), k))}
    for name, weights in WEIGHTS.items():
        rec = ContentRecommender(df, weights)
        results[name] = score_recommendations(df, rec.top_k_for_everything(k))
    return pd.DataFrame(results).T


def save_evaluation_plot(results, path):
    blue, grey, ink, grid = "#2a78d6", "#b9b8b2", "#0b0b0b", "#e4e3de"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")

    panels = [("genre_jaccard", "Genre overlap with the query (Jaccard)"),
              ("coverage", "Share of catalogue ever recommended")]
    for ax, (col, title) in zip(axes, panels):
        vals = results[col]
        colors = [grey if name == "random baseline" else blue for name in vals.index]
        bars = ax.barh(vals.index, vals.values, color=colors, height=0.6)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xlim(0, 1.08)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        for bar, v in zip(bars, vals.values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", color=ink, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Netflix content-based recommender")
    parser.add_argument("--title", help="get recommendations for just this title")
    parser.add_argument("--n", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()

    df = load_data()
    recommender = ContentRecommender(df, WEIGHTS[DEFAULT_VARIANT])

    if args.title:
        try:
            print(recommender.recommend(args.title, args.n).to_string(index=False))
        except ValueError as err:
            print(err)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    print(f"Loaded {len(df):,} titles, TF-IDF matrix: {recommender.matrix.shape[0]} x {recommender.matrix.shape[1]}\n")

    # step 4 - recommendations for a few well known titles
    frames = []
    for title in SAMPLE_TITLES:
        recs = recommender.recommend(title, args.n)
        print(f"Because you watched '{title}':")
        print(recs.to_string(index=False), "\n")
        recs.insert(0, "query_title", title)
        frames.append(recs)
    pd.concat(frames).to_csv(OUT_DIR / "sample_recommendations.csv", index=False)

    # step 5 - evaluation
    results = run_evaluation(df, k=10)
    print("Evaluation (top-10 recommendations for every title in the catalogue):")
    print(results.round(3).to_string(), "\n")
    results.round(4).to_csv(OUT_DIR / "evaluation_metrics.csv")
    save_evaluation_plot(results, OUT_DIR / "evaluation.png")
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
