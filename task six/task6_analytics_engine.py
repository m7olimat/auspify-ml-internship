"""
Task 6 - Netflix content success analytics engine (end-to-end pipeline)

IMPORTANT - how "success" is defined here
The dataset has no views, watch-hours or review scores, so true success can't be
measured. As a proxy I use "fresh release": the title was added to Netflix in the
same year as its release or the year after. Those are the titles Netflix moved on
quickly (originals, exclusives, new licences) instead of back-catalogue filler.
If you get a real performance metric later, only `add_target()` needs to change.

Pipeline (follows the task list):
  1. advanced feature engineering
  2. build several models (+ tune the best one)
  3. compare them on cross-validation and on a held-out test set
  4. generate insights automatically from the data and the model
  5. visual reports + a saved model that can score new titles

Run:
    python task6_analytics_engine.py
"""
import ast
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "netflix_clean.csv"
OUT_DIR = BASE_DIR / "outputs" / "task6"

SEED = 42
# Number of parallel workers. Kept at 1 on purpose: on some Windows setups (antivirus,
# restricted folders) joblib can't start worker processes and crashes with
# "PermissionError: [WinError 5]". Raise it (or use -1) if your machine allows it.
N_JOBS = 1
MIN_GENRE_COUNT = 40  # genres rarer than this don't get their own column

KIDS = ["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G"]
MATURE = ["TV-MA", "R", "NC-17"]
RATING_GROUP = {**{r: "kids" for r in KIDS},
                **{r: "family/teen" for r in ["TV-PG", "PG", "TV-14", "PG-13"]},
                **{r: "adult" for r in MATURE},
                **{r: "unrated" for r in ["NR", "UR"]}}

CAT_COLS = ["type", "rating", "main_country"]
NUM_COLS = ["n_countries", "n_genres", "n_directors", "duration_min", "n_seasons",
            "title_words", "year_added", "month_added"]

INK, BLUE, ORANGE, AQUA, VIOLET, GREY, GRID, SURFACE = "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#b9b8b2", "#e4e3de", "#fcfcfb"


# ------------------------------------------------------------ data + target

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["title"] = df["title"].str.strip()
    df["genres"] = df["genres"].apply(ast.literal_eval)
    df["countries"] = df["countries"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
    df["rating_group"] = df["rating"].map(RATING_GROUP).fillna("unrated")
    return df


def add_target(df):
    """Success proxy: added to Netflix within a year of its release."""
    df["fresh_release"] = ((df["year_added"] - df["release_year"]) <= 1).astype(int)
    return df


# ------------------------------------------------ step 1: feature engineering

def engineer_features(df, genre_vocab):
    """Raw rows -> model features. release_year is NOT used (it defines the target)."""
    X = pd.DataFrame(index=df.index)
    X["type"] = df["type"]
    X["rating"] = df["rating"]
    X["main_country"] = df["main_country"].fillna("Unknown")

    X["n_countries"] = df["countries"].apply(len)
    X["n_genres"] = df["genres"].apply(len)
    X["n_directors"] = df["director"].fillna("").apply(lambda s: len([d for d in s.split(",") if d.strip()]))
    X["has_director"] = (X["n_directors"] > 0).astype(int)

    # movies have a runtime and no seasons, TV shows the other way round -> 0 when not applicable
    X["duration_min"] = df["duration_min"].fillna(0)
    X["n_seasons"] = df["n_seasons"].fillna(0)

    # small hints from the title: sequels / spin-offs tend to carry numbers or colons
    X["title_words"] = df["title"].str.split().str.len()
    X["title_has_number"] = df["title"].str.contains(r"\d").astype(int)
    X["title_has_colon"] = df["title"].str.contains(":").astype(int)

    X["is_kids"] = df["rating"].isin(KIDS).astype(int)
    X["is_mature"] = df["rating"].isin(MATURE).astype(int)
    X["is_international"] = ((X["main_country"] != "United States") & (X["main_country"] != "Unknown")).astype(int)

    X["year_added"] = df["year_added"]
    X["month_added"] = df["month_added"]

    for g in genre_vocab:
        X[f"genre_{g}"] = df["genres"].apply(lambda gs, g=g: int(g in gs))
    return X


def get_genre_vocab(df):
    counts = df["genres"].explode().value_counts()
    return sorted(counts[counts >= MIN_GENRE_COUNT].index)


# ------------------------------------------------------ step 2: the models

def make_preprocessor(scale):
    """One-hot for categories, optional scaling for numbers, everything else (flags, genres) passes through."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), CAT_COLS),
            ("num", StandardScaler() if scale else "passthrough", NUM_COLS),
        ],
        remainder="passthrough",
    )


def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("model", LogisticRegression(max_iter=3000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=N_JOBS, random_state=SEED)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("model", HistGradientBoostingClassifier(random_state=SEED)),
        ]),
    }


SEARCH_SPACES = {
    "Logistic Regression": {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 3, 5, 10],
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 300],
        "model__max_leaf_nodes": [8, 15, 31],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def cross_validate_models(models, X, y, cv):
    rows = {}
    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=N_JOBS)
        rows[name] = {
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1": scores["test_f1"].mean(),
            "cv_auc": scores["test_roc_auc"].mean(),
            "cv_auc_std": scores["test_roc_auc"].std(),
        }
    return pd.DataFrame(rows).T.sort_values("cv_auc", ascending=False)


def tune(name, pipe, X, y, cv):
    space = SEARCH_SPACES[name]
    grid_size = int(np.prod([len(v) for v in space.values()]))
    search = RandomizedSearchCV(pipe, space, n_iter=min(12, grid_size), scoring="roc_auc",
                                cv=cv, random_state=SEED, n_jobs=N_JOBS)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def test_metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


# ------------------------------------------------------ step 4: insights

def rate_table(df, column, min_n):
    t = df.groupby(column)["fresh_release"].agg(n="size", fresh_rate="mean")
    return t[t["n"] >= min_n].sort_values("fresh_rate", ascending=False)


def genre_table(df, min_n=150):
    exploded = df.explode("genres")
    return rate_table(exploded, "genres", min_n)


def build_insights(df, best_name, test_row, importance):
    overall = df["fresh_release"].mean()
    by_type = df.groupby("type")["fresh_release"].mean()
    genres = genre_table(df)
    countries = rate_table(df, "main_country", 100)
    groups = rate_table(df, "rating_group", 100)
    years = rate_table(df, "year_added", 100).sort_index()

    top_genres = df["genres"].explode().value_counts().head(3)
    top_countries = df["main_country"].value_counts().head(3)
    movie_share = (df["type"] == "Movie").mean()

    def fmt(table, idx, with_n=True):
        r = table.iloc[idx]
        return f"{table.index[idx]} ({r['fresh_rate']:.0%}" + (f", n={int(r['n'])})" if with_n else ")")

    lines = [
        f"The catalogue holds {len(df):,} titles: {movie_share:.0%} movies and {1 - movie_share:.0%} TV shows.",
        "Most common genres: " + ", ".join(f"{g} ({n:,})" for g, n in top_genres.items()) + ".",
        "Biggest producing countries: " + ", ".join(f"{c} ({n:,})" for c, n in top_countries.items()) + ".",
        f"{overall:.0%} of titles are 'fresh releases' (added within a year of release). "
        f"TV shows are more often fresh ({by_type['TV Show']:.0%}) than movies ({by_type['Movie']:.0%}).",
        f"Genres with the highest fresh share: {fmt(genres, 0)}, {fmt(genres, 1)}, {fmt(genres, 2)}.",
        f"Genres leaning on older titles: {fmt(genres, -1)}, {fmt(genres, -2)}, {fmt(genres, -3)}.",
        f"Countries (100+ titles) with the freshest content: {fmt(countries, 0)}, {fmt(countries, 1)}; "
        f"the oldest-leaning: {fmt(countries, -1)}, {fmt(countries, -2)}.",
        "By audience: " + ", ".join(f"{g} {r['fresh_rate']:.0%}" for g, r in groups.iterrows()) + " fresh.",
    ]
    if len(years) >= 2:
        first, last = years.index[0], years.index[-1]
        lines.append(f"Fresh share moved from {years.iloc[0]['fresh_rate']:.0%} in {first} "
                     f"to {years.iloc[-1]['fresh_rate']:.0%} in {last} (years with 100+ additions).")

    drivers = ", ".join(importance.index[:5].str.replace("genre_", "genre: ").str.replace("_", " "))
    lines += [
        f"Best model: {best_name} (test ROC-AUC {test_row['roc_auc']:.2f}, accuracy {test_row['accuracy']:.0%}, "
        f"F1 {test_row['f1']:.2f}).",
        f"Attributes that matter most to the model: {drivers}.",
        "Reminder: 'fresh release' is a proxy, not real viewership. Treat these as patterns in how the "
        "catalogue is built, not as proof of what audiences like.",
    ]
    return lines


# ------------------------------------------------------ step 5: visuals

def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK)


def hbar(ax, table, overall, title, value_col="fresh_rate"):
    labels = table.index.astype(str)
    ax.barh(labels, table[value_col], color=BLUE, height=0.65)
    ax.axvline(overall, color=INK, lw=1, ls="--")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "x")
    for i, v in enumerate(table[value_col]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8, color=INK,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5})


def plot_business_overview(df, path):
    overall = df["fresh_release"].mean()
    genres = genre_table(df)
    countries = rate_table(df, "main_country", 100).head(12)
    years = rate_table(df, "year_added", 100).sort_index()
    groups = rate_table(df, "rating_group", 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE)
    hbar(axes[0, 0], genres, overall, "Fresh-release share by genre (150+ titles)")
    hbar(axes[0, 1], countries, overall, "Fresh-release share by country (top 12, 100+ titles)")

    ax = axes[1, 0]
    ax.plot(years.index.astype(str), years["fresh_rate"], color=BLUE, lw=2, marker="o")
    ax.axhline(overall, color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Fresh-release share by year added", loc="left", color=INK, fontsize=11)
    style(ax)

    ax = axes[1, 1]
    ax.bar(groups.index, groups["fresh_rate"], color=BLUE, width=0.6)
    ax.axhline(overall, color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Fresh-release share by audience group", loc="left", color=INK, fontsize=11)
    style(ax)
    for i, v in enumerate(groups["fresh_rate"]):
        ax.text(i, v + 0.015, f"{v:.0%}", ha="center", fontsize=9, color=INK)

    fig.suptitle(f"Dashed line = overall average ({overall:.0%})", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_report(cv_table, test_table, roc_data, best_name, cm, importance, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE)

    # (a) cross-validation vs test AUC
    ax = axes[0, 0]
    names = list(cv_table.index)
    x = np.arange(len(names))
    ax.bar(x - 0.2, cv_table["cv_auc"], 0.38, yerr=cv_table["cv_auc_std"], color=BLUE, label="5-fold CV (train)",
           error_kw={"ecolor": INK, "lw": 1})
    ax.bar(x + 0.2, test_table.loc[names, "roc_auc"], 0.38, color=ORANGE, label="Held-out test")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.5, 1)
    ax.set_title("ROC-AUC by model", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    for i, n in enumerate(names):
        ax.text(i - 0.2, cv_table.loc[n, "cv_auc"] + cv_table.loc[n, "cv_auc_std"] + 0.005,
                f"{cv_table.loc[n, 'cv_auc']:.3f}", ha="center", fontsize=9, color=INK)
        ax.text(i + 0.2, test_table.loc[n, "roc_auc"] + 0.005, f"{test_table.loc[n, 'roc_auc']:.3f}",
                ha="center", fontsize=9, color=INK)

    # (b) ROC curves
    ax = axes[0, 1]
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), [BLUE, ORANGE, AQUA, VIOLET]):
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    style(ax)

    # (c) permutation importance
    ax = axes[1, 0]
    top = importance.head(12)
    ax.barh(top.index.str.replace("genre_", "genre: ").str.replace("_", " "), top["mean"], xerr=top["std"],
            color=BLUE, height=0.65, error_kw={"ecolor": INK, "lw": 1})
    ax.invert_yaxis()
    ax.set_title(f"What drives the {best_name} model (permutation importance, drop in AUC)",
                 loc="left", color=INK, fontsize=11)
    style(ax, "x")

    # (d) confusion matrix
    ax = axes[1, 1]
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Back-catalogue", "Fresh release"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Back-catalogue", "Fresh release"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {best_name}, test set", loc="left", color=INK, fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = add_target(load_data())
    genre_vocab = get_genre_vocab(df)
    X = engineer_features(df, genre_vocab)
    y = df["fresh_release"]
    print(f"{len(df):,} titles, {X.shape[1]} features, {y.mean():.1%} fresh releases\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # step 2 + 3: baseline models, compared with cross-validation on the training data only
    models = get_models()
    cv_table = cross_validate_models(models, X_train, y_train, cv)
    print("Cross-validation on the training set:")
    print(cv_table.round(3).to_string(), "\n")

    # tune whichever model won the cross-validation (the test set is not touched for this decision)
    best_name = cv_table.index[0]
    tuned, best_params, tuned_cv_auc = tune(best_name, models[best_name], X_train, y_train, cv)
    print(f"Tuned {best_name}: CV AUC {cv_table.loc[best_name, 'cv_auc']:.3f} -> {tuned_cv_auc:.3f}")
    print("Best parameters:", {k.replace("model__", ""): v for k, v in best_params.items()}, "\n")

    # final comparison on the held-out test set
    probas, roc_data, rows = {}, {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        probas[name] = pipe.predict_proba(X_test)[:, 1]
    probas[f"{best_name} (tuned)"] = tuned.predict_proba(X_test)[:, 1]
    for name, p in probas.items():
        rows[name] = test_metrics(y_test, p)
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_data[name] = (fpr, tpr, rows[name]["roc_auc"])
    test_table = pd.DataFrame(rows).T
    print("Held-out test set:")
    print(test_table.round(3).to_string(), "\n")

    # keep the tuned version only if it really is better on cross-validation, otherwise stay with the default
    use_tuned = tuned_cv_auc >= cv_table.loc[best_name, "cv_auc"]
    final_name = f"{best_name} (tuned)" if use_tuned else best_name
    final_model = tuned if use_tuned else models[best_name]
    final_proba = probas[final_name]
    cm = confusion_matrix(y_test, (final_proba >= 0.5).astype(int))

    # what is the model actually using?
    perm = permutation_importance(final_model, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                  random_state=SEED, n_jobs=N_JOBS)
    importance = (pd.DataFrame({"mean": perm.importances_mean, "std": perm.importances_std}, index=X_test.columns)
                  .sort_values("mean", ascending=False))

    # step 4: automated insights
    insights = build_insights(df, final_name, test_table.loc[final_name], importance)
    print("Automated insights:")
    for line in insights:
        print(" -", line)
    (OUT_DIR / "insights.md").write_text("# Automated insights\n\n" + "\n".join(f"- {l}" for l in insights) + "\n")

    # step 5: reports
    plot_business_overview(df, OUT_DIR / "business_overview.png")
    plot_model_report(cv_table, test_table.loc[cv_table.index], roc_data, final_name, cm, importance, OUT_DIR / "model_report.png")
    cv_table.round(4).to_csv(OUT_DIR / "cv_results.csv")
    test_table.round(4).to_csv(OUT_DIR / "test_results.csv")
    importance.round(5).to_csv(OUT_DIR / "feature_importance.csv")

    # save the model so new titles can be scored without retraining
    bundle = {"pipeline": final_model, "genre_vocab": genre_vocab, "model_name": final_name}
    joblib.dump(bundle, OUT_DIR / "fresh_release_model.joblib")
    reloaded = joblib.load(OUT_DIR / "fresh_release_model.joblib")
    sample = df.loc[X_test.index[:5]]
    new_proba = reloaded["pipeline"].predict_proba(engineer_features(sample, reloaded["genre_vocab"]))[:, 1]
    assert np.allclose(new_proba, final_proba[:5]), "reloaded model gives different predictions"
    print("\nSaved model reloads fine. Example scores:")
    print(pd.DataFrame({"title": sample["title"].values, "p_fresh": new_proba.round(3),
                        "actual": sample["fresh_release"].values}).to_string(index=False))
    print(f"\nOutputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
