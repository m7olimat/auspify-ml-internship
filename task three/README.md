# Task 3: Netflix Audience Rating Classification

## What is this?
This is a multi-class classification model. It predicts the **age group** of a Netflix title (Kids, Family, Teen, or Adult) from the title's attributes. The dataset has 8,790 titles.

## What did I do?
1. **Grouping the ratings**
   The `rating` column has 14 different values from two systems (movie ratings like PG-13 and TV ratings like TV-14). Some of them have only 3 to 6 titles, which is too few for a model to learn from. So I grouped them into 4 age groups:

   | Age group | Ratings | Titles |
   |---|---|---|
   | Kids | TV-Y, TV-Y7, TV-Y7-FV, TV-G, G | 906 |
   | Family | PG, TV-PG | 1,148 |
   | Teen | PG-13, TV-14 | 2,647 |
   | Adult | R, TV-MA, NC-17 | 4,007 |

   I removed `NR` and `UR` (82 titles). They mean "not rated", so they are not an age level and cannot be predicted from the content.
2. **Features and target**
   - The target is `age_group`. The original `rating` column is **not** used as a feature (it would leak the answer).
   - The features are: `release_year`, `year_added`, `month_added`, `main_country`, `director`, and the genres.
   - I split the `listed_in` column into one 0/1 column for each of the 42 genres.
   - Split the data 80% train / 20% test (stratified).
3. **Encoding**
   One-Hot Encoding for `main_country` and `director` (I kept the 20 most frequent values and grouped the rest as `Other`), StandardScaler for the numeric columns, and the genre columns passed as they are because they are already 0/1.
4. **Training and tuning**
   I trained Logistic Regression, Decision Tree, and Random Forest. Then I tuned the Decision Tree and the Random Forest with `GridSearchCV` (5-fold cross-validation on the training set only, scored with macro F1).

I use **macro F1** as the main metric, because the groups are not balanced. If a model predicts "Adult" for everything, it gets about 46% accuracy without learning anything.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | **0.630** | **0.581** |
| Decision Tree | 0.572 | 0.548 |
| Decision Tree (tuned) | 0.601 | 0.561 |
| Random Forest | 0.607 | 0.571 |
| Random Forest (tuned) | 0.626 | 0.570 |

Best parameters from the grid search:
- Decision Tree: `max_depth=25`, `min_samples_leaf=9`
- Random Forest: `n_estimators=200`, `max_depth=20`, `min_samples_leaf=1`

### Effect of adding the genres
My first version did not use the genres, and the results were close to the 46% baseline:

| Model | Macro F1 (no genres) | Macro F1 (with genres) |
|---|---|---|
| Logistic Regression | 0.325 | 0.581 |
| Decision Tree | 0.361 | 0.548 |
| Random Forest | 0.382 | 0.571 |

### Best model: Logistic Regression

| Age group | Precision | Recall | F1 | Titles in test set |
|---|---|---|---|---|
| Adult | 0.65 | 0.84 | 0.73 | 802 |
| Family | 0.57 | 0.25 | 0.35 | 230 |
| Kids | 0.81 | 0.69 | 0.75 | 181 |
| Teen | 0.55 | 0.46 | 0.50 | 529 |

![Confusion matrix of Logistic Regression](task3_confusion_matrix.png)

## What I found
1. **The genres are the most important feature.** Adding them raised the macro F1 from about 0.33 to 0.38 up to about 0.55 to 0.58. Dates, country, and director alone say very little about the age group.
2. **Tuning helped only a little.** It improved the Decision Tree by about 1.3 points of F1, and the Random Forest did not improve in F1. The simple Logistic Regression is still the best model, but the gap between the models is small (about 1 point), so I do not claim it is clearly better.
3. **The model is weak on the Family group.** Its recall is only 0.25. The model finds only 58 of the 230 Family titles, and the rest are spread over Adult (78), Teen (66), and Kids (28).
4. **Teen and Adult are mixed up the most.** 264 Teen titles were predicted as Adult, and 126 Adult titles were predicted as Teen.
5. **Kids is the easiest group.** It has the best precision (0.81), because its genres are very specific (kids' shows, family movies).

## Why?
The model tends to predict the biggest group (Adult), so the smaller groups lose recall. Teen and Adult are close to each other (for example TV-14 vs TV-MA, or PG-13 vs R), and the same genres appear in both groups, so the genres cannot separate them well. Family is small (only 230 test titles) and sits between Kids and Teen, which is why the model rarely finds it.

## Files
- `task3_age_rating_classification.py`: the full code (grouping, features, encoding, training, tuning, evaluation).
- `task3_confusion_matrix.png`: confusion matrix of the best model.
- The code reads `netflix_clean.csv` (the output of the cleaning step) from the same folder.
