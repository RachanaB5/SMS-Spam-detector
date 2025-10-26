import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC # <-- Fast linear model for subtype
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder # Non-negative scaling for MNB
import pickle
import random

# Set the filename for your new, rich dataset
DATASET_FILE = "enhanced_scam_dataset.csv"

# --- Feature Definitions (Must match the dataset) ---
NUMERICAL_FEATURES = ["contains_link", "sender_trust_score", "link_reputation_score", "ai_generated_score"]
CATEGORICAL_FEATURES = ["language_type"]
TEXT_FEATURE = "message"

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s/.:-]", " ", text)
        return ' '.join(text.split())
    return ""

def load_and_prepare_data(filepath=DATASET_FILE):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{filepath}' not found. Please run data generation.")
        return None

    df[TEXT_FEATURE] = df[TEXT_FEATURE].astype(str).apply(preprocess_text)
    df = df[df[TEXT_FEATURE].str.strip() != ""]
    
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df["label_main_int"] = df["category"].map({"ham": 0, "scam": 1, "promo": 2})
    df.loc[df["label_subtype"].isna(), "label_subtype"] = "Unknown"
    return df

def create_feature_pipeline():
    """Uses MinMaxScaler and reduced max_features."""
    text_transformer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())]) 
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, TEXT_FEATURE),
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def train_combined_binary_model(df):
    df['binary_target'] = df['label_main_int'].apply(lambda x: 0 if x == 0 else 1)
    X = df[[TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df['binary_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', create_feature_pipeline()),
        ('classifier', MultinomialNB(alpha=0.1)) # Fastest classifier for binary task
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n📊 COMBINED Binary Classification Report (MNB):\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam/Promo"]))
    return pipeline

def train_combined_subtype_model(df, feature_pipeline):
    """Uses LinearSVC for fast multi-class prediction based on all features."""
    non_ham_df = df[df["label_main_int"] != 0].copy()
    X = non_ham_df[[TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = non_ham_df["label_subtype"]
    
    value_counts = y.value_counts()
    to_keep = value_counts[value_counts >= 2].index
    non_ham_df = non_ham_df[non_ham_df['label_subtype'].isin(to_keep)]
    X = non_ham_df[[TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = non_ham_df["label_subtype"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = feature_pipeline.named_steps['preprocessor']
    
    # LinearSVC: Fast and robust for sparse, multi-class text data
    subtype_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(random_state=42, dual=False, max_iter=2000)) 
    ])
    
    X_train_transformed = preprocessor.transform(X_train)
    subtype_pipeline.named_steps['classifier'].fit(X_train_transformed, y_train)
    y_pred = subtype_pipeline.predict(X_test)
    print("\n🧩 COMBINED Subtype Classification Report (LinearSVC):\n", classification_report(y_test, y_pred))

    return subtype_pipeline

def save_models(binary_pipeline, subtype_pipeline):
    with open("combined_binary_pipeline.pkl", "wb") as f:
        pickle.dump(binary_pipeline, f)
    with open("combined_subtype_pipeline.pkl", "wb") as f:
        pickle.dump(subtype_pipeline, f)
    print("\n✅ Combined Pipelines saved successfully!")

if __name__ == "__main__":
    df = load_and_prepare_data(DATASET_FILE)
    if df is None: exit()
    print("="*60)
    print("🧠 Training COMBINED Binary Pipeline (MNB)...")
    combined_binary_pipeline = train_combined_binary_model(df)
    print("\n🧩 Training COMBINED Subtype Pipeline (LinearSVC)...")
    combined_subtype_pipeline = train_combined_subtype_model(df, combined_binary_pipeline)
    print("\n💾 Saving Models...")
    save_models(combined_binary_pipeline, combined_subtype_pipeline)
    print("="*60)