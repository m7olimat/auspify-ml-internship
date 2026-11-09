# Task 6 - Netflix Content Success Analytics Engine

An end-to-end pipeline: feature engineering, several models, model comparison,
automatically generated insights, visual reports and a saved model.

## An important note about "success"
The dataset has no views, watch time or review scores, so real success can't be
measured. I used a proxy instead: a title counts as a **fresh release** if it was
added to Netflix in the same year it was released, or the year after. The idea is
that these are titles Netflix moved on quickly (originals, exclusives, new
licences) rather than older back-catalogue. 55% of the titles are fresh releases,
so the two classes are well balanced.

So the insights below describe how the catalogue is built, not what audiences
enjoy. If a real performance metric becomes available, only `add_target()` in the
script needs to change.

## What I used
- Python, pandas, numpy
- scikit-learn (pipelines, Logistic Regression, Random Forest, Gradient Boosting,
  cross-validation, randomized search, permutation importance)
- joblib to save the model
- matplotlib

## How it works
1. **Feature engineering.** 57 features: type, rating, main country, number of
   countries / genres / directors, runtime (movies) and seasons (TV), word count
   and number/colon flags from the title, kids / mature / international flags,
   year and month added, and one yes/no column for each genre with at least 40
   titles. I did not use `release_year` as a feature because it is part of the
   target definition, which would leak the answer.
2. **Models.** Logistic Regression, Random Forest and Gradient Boosting, each in
   a pipeline with its own preprocessing (one-hot encoding for categories, scaling
   only for the logistic regression).
3. **Comparison.** 80/20 train/test split. Models are compared with 5-fold
   cross-validation on the training part only. The best one then gets a
   randomized hyperparameter search, and finally everything is scored on the test
   set that was never touched during the choices.
4. **Automated insights.** `insights.md` is written by code straight from the
   data and the model, so it updates if the data changes.
5. **Visual reports.** `business_overview.png` (fresh share by genre, country,
   year and audience group) and `model_report.png` (model scores, ROC curves,
   feature importance, confusion matrix).
6. **Reusable model.** The fitted pipeline is saved to `fresh_release_model.joblib`
   along with the genre list, and the script reloads it and checks it gives the
   same predictions.

## How to run
    pip install -r requirements.txt
    python task6_analytics_engine.py

Takes about 40 seconds. Everything goes to `outputs/task6/`.

## Results
| Model | CV AUC | Test accuracy | Test F1 | Test AUC |
|---|---|---|---|---|
| Logistic Regression | 0.767 | 0.712 | 0.753 | 0.762 |
| Random Forest | 0.802 | 0.733 | 0.773 | 0.799 |
| Gradient Boosting | 0.808 | 0.743 | 0.781 | 0.811 |

Gradient Boosting won. The tuned version scored the same AUC and no better in
cross-validation, so I kept the default settings.

## Main findings
- TV shows are more often fresh (67%) than movies (50%).
- Highest fresh share by genre: TV Action & Adventure (80%), Stand-Up Comedy
  (78%), Crime TV Shows (77%). Most tied to older titles: Action & Adventure
  (34%), Sci-Fi & Fantasy (36%), Romantic Movies (40%).
- By country (100+ titles): Spain is the freshest at 89%, Egypt the oldest-leaning
  at 17%.
- The fresh share slowly dropped from 66% in 2016 to 51% in 2021.
- What the model relies on most: rating, main country, year added, runtime and
  month added.

## Limitations
- The target is a proxy (see the note at the top), so a high score means "good at
  telling fresh from older titles", not "good at predicting hits".
- A test AUC of 0.81 is decent but far from perfect. About 1 in 4 titles is
  classified wrongly.
- Permutation importance shows what the model uses, not what causes a title to be
  fresh.
- Some countries and genres have only 100 to 150 titles, so their percentages
  move around a lot.
