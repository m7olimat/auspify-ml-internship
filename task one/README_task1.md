# Task 1 - Netflix Content Recommendation System

For this task I built a content-based recommender. You give it a title and it
returns the Netflix titles that are most similar to it.

## What I used
- Python, pandas, numpy
- scikit-learn (TF-IDF and cosine similarity)
- matplotlib for the evaluation chart

## How it works
1. **Prepare features.** The dataset has no plot descriptions, so I used what
   it does have: genres, type (movie or TV show), rating, country and director.
   The genre and country columns were stored as strings like `"['Drama', 'Comedy']"`,
   so I parsed them back into real lists first.
2. **Make the text machine-readable.** Each title becomes a short list of tokens
   such as `genre_dramas`, `type_movie`, `rating_tv_ma`. I repeat the more
   important tokens (genre x3, type x2) so they count for more. Then I run TF-IDF
   over these token lists, which gives rare tokens (like a specific director)
   more weight than very common ones.
3. **Similarity scores.** Cosine similarity between the TF-IDF vectors.
4. **Recommendations.** `ContentRecommender.recommend(title, n)` returns the
   top n matches. If a title is misspelled it suggests close names instead of
   crashing. The script prints results for five example titles (Stranger Things,
   The Crown, Narcos, Dangal, Kung Fu Panda).
5. **Evaluation.** See below.

## How to run
    pip install -r requirements.txt
    python task1_recommender.py
    python task1_recommender.py --title "Narcos" --n 8

It takes about 15 seconds. Outputs go to `outputs/task1/`:
`sample_recommendations.csv`, `evaluation_metrics.csv`, `evaluation.png`.

## Evaluation
There is no user or viewing data in this dataset, so I couldn't measure real
"did the user like it" accuracy. Instead I generated the top 10 for every one of
the 8,790 titles and checked how consistent they are with the title they came
from, compared with picking 10 random titles.

| Version | Genre overlap | Same rating group | Same country | Catalogue coverage |
|---|---|---|---|---|
| Random baseline | 0.09 | 0.41 | 0.16 | 1.00 |
| Genres only | 0.97 | 0.58 | 0.31 | 0.29 |
| Genres + type + rating | 0.97 | 0.89 | 0.36 | 0.51 |
| All attributes (final) | 0.95 | 0.75 | 0.62 | 0.71 |

I went with "all attributes". It loses a tiny bit of genre overlap but the
recommendations are much more varied (71% of the catalogue shows up in someone's
top 10, against 29% for genres only) and they respect the country much better.
Every version returned the same type (movie/TV) as the query 100% of the time.

## Limitations
- Genre overlap is high partly because genre is an input, so that number is more
  of a sanity check than proof the suggestions are good.
- Without plot text, two titles with the same genres, rating and country look
  identical to the model, so you will see ties in the similarity scores.
- It is content-based only. It can't learn what people actually watch.
- A few titles appear twice in the data (for example "22-Jul"). The recommender
  never suggests a title to itself, including its duplicate.
