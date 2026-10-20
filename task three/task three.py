"""
Task 3: Netflix Audience Rating Classification

Predict the age group of a Netflix title (Kids / Family / Teen / Adult)
from its attributes (genres, country, director, release/added dates).

Input: netflix_clean.csv (output of the cleaning step, same folder as this script)
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 1. Load the data and group the 14 ratings into age groups
# ---------------------------------------------------------------
df = pd.read_csv('netflix_clean.csv')

age_rating = {
    'TV-Y': 'Kids',
    'TV-Y7': 'Kids',
    'TV-Y7-FV': 'Kids',
    'TV-G': 'Kids',
    'G': 'Kids',
    'PG': 'Family',
    'TV-PG': 'Family',
    'TV-14': 'Teen',
    'PG-13': 'Teen',
    'R': 'Adult',
    'TV-MA': 'Adult',
    'NC-17': 'Adult',
    'NR': 'NR',
    'UR': 'NR',
}

df['age_group'] = df['rating'].map(age_rating)
df = df[df['age_group'] != 'NR'].copy()  # NR / UR mean "no rating", not an age level
print(df['age_group'].value_counts(), '\n')

# ---------------------------------------------------------------
# 2. Features and target (the rating itself is NOT used as a feature)
# ---------------------------------------------------------------
genres = df['listed_in'].str.get_dummies(sep=', ')  # one 0/1 column per genre
features = ['release_year', 'year_added', 'month_added', 'main_country', 'director']

y = df['age_group']
X = df[features].copy()
X = pd.concat([X, genres], axis=1)
X['main_country'] = X['main_country'].fillna('Unknown')
X['director'] = X['director'].fillna('Unknown')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 3. Encoding: keep the 20 most frequent countries/directors, group the rest
# ---------------------------------------------------------------
cat_cols = ['main_country', 'director']
num_cols = ['release_year', 'year_added', 'month_added']

for col in cat_cols:
    top = X_train[col].value_counts().head(20).index
    X_train[col] = X_train[col].where(X_train[col].isin(top), 'Other')
    X_test[col] = X_test[col].where(X_test[col].isin(top), 'Other')

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols),
    ('genres', 'passthrough', list(genres.columns)),  # already 0/1
])

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# ---------------------------------------------------------------
# 4. Train three models
# ---------------------------------------------------------------
models = {
    'LogisticRegression': LogisticRegression(max_iter=10000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train_enc, y_train)
    pred = model.predict(X_test_enc)
    results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'f1_macro': f1_score(y_test, pred, average='macro'),
    }
    predictions[name] = pred

# ---------------------------------------------------------------
# 5. Hyperparameter tuning (GridSearchCV on the training set only)
# ---------------------------------------------------------------
tree_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {
        'max_depth': [6, 8, 9, 10, 12, 15, 20, 25, 27, 30, 35, 40, 45],
        'min_samples_leaf': [5, 6, 7, 8, 9, 10, 12, 15, 20, 25],
    },
    scoring='f1_macro', cv=5,
)
tree_grid.fit(X_train_enc, y_train)
tree_pred = tree_grid.predict(X_test_enc)
results['DecisionTree tuned'] = {
    'accuracy': accuracy_score(y_test, tree_pred),
    'f1_macro': f1_score(y_test, tree_pred, average='macro'),
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 5],
    },
    scoring='f1_macro', cv=5,
)
rf_grid.fit(X_train_enc, y_train)
rf_pred = rf_grid.predict(X_test_enc)
results['RandomForest tuned'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'f1_macro': f1_score(y_test, rf_pred, average='macro'),
}

print('Decision Tree best params:', tree_grid.best_params_, '| CV f1:', round(tree_grid.best_score_, 3))
print('Random Forest best params:', rf_grid.best_params_, '| CV f1:', round(rf_grid.best_score_, 3), '\n')

# ---------------------------------------------------------------
# 6. Results and evaluation of the best model (Logistic Regression)
# ---------------------------------------------------------------
print(pd.DataFrame(results).T.round(3), '\n')

best_pred = predictions['LogisticRegression']
print(classification_report(y_test, best_pred))

labels = models['LogisticRegression'].classes_
cm = confusion_matrix(y_test, best_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.savefig('task3_confusion_matrix.png', dpi=150)
plt.show()