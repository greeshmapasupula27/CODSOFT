import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Select useful columns
df = df[['Genre', 'Director', 'Actor 1', 'Rating']]

# Remove missing target values
df = df.dropna(subset=['Rating'])

# Features and target
X = df[['Genre', 'Director', 'Actor 1']]
y = df['Rating']

# Preprocessing
text_features = ['Genre', 'Director', 'Actor 1']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('tfidf', TfidfVectorizer())
        ]), 'Genre')
    ],
    remainder='drop'
)

# Convert all text columns into one
X['combined'] = (
    X['Genre'].fillna('') + ' ' +
    X['Director'].fillna('') + ' ' +
    X['Actor 1'].fillna('')
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X['combined'], y, test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)