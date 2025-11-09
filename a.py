import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import re

def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load the new balanced dataset
print("Loading dataset...")
df = pd.read_csv('enhanced_5000_dataset.csv')

print("Dataset shape:", df.shape)
print("Category distribution:")
print(df['category'].value_counts())

# Clean the text
df['cleaned_message'] = df['message'].apply(clean_text)

# Prepare features and labels
X = df['cleaned_message']
y = df['category']

# Convert labels to numerical values
label_mapping = {'spam': 0, 'ham': 1, 'promo': 2}
y_numeric = y.map(label_mapping)

print(f"Label distribution: {np.bincount(y_numeric)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
)

# Create TF-IDF features
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Train the model
print("Training model...")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',  # Important for imbalanced data
    C=1.0
)

model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)

print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['spam', 'ham', 'promo']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save the model and vectorizer
print("\nSaving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model training completed and saved!")