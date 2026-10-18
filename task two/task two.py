import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score,accuracy_score

df=pd.read_csv(r'C:\Users\Lenovo\Desktop\mohammad\auspify\cleaning data\netflix_clean.csv')
y=df['type']=='TV Show'
features=['release_year','year_added','month_added','rating','main_country','director']
X=df[features].copy()
X['main_country']=X['main_country'].fillna('Unknown')
X['director']=X['director'].fillna('Unknown')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

cat_cols=['rating','main_country','director']
num_cols=['release_year','year_added','month_added']

for col in ['main_country', 'director']:
    top = X_train[col].value_counts().head(20).index
    X_train[col] = X_train[col].where(X_train[col].isin(top), 'Other')
    X_test[col] = X_test[col].where(X_test[col].isin(top), 'Other')

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols),
])

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

print(X_train_enc.shape)
print(X_test_enc.shape)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_enc, y_train)
    pred = model.predict(X_test_enc)
    results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred),
    }

modell=models['Logistic Regression']
preddi=modell.predict(X_test_enc)
print("logistic regression ",confusion_matrix(y_test, preddi))
print(pd.DataFrame(results).T)
print("Random forest ",confusion_matrix(y_test, pred))
print(classification_report(y_test, preddi))

cm = confusion_matrix(y_test, preddi)
sns.heatmap(cm, annot=True, fmt="d",xticklabels=['movies','Tv shows'],yticklabels=['movies','Tv shows'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('logistic regression confusion matrix')
plt.show()