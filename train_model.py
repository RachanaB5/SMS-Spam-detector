import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import json
from datetime import datetime

# Load the enhanced dataset
DATASET_FILE = "enhanced_5000_dataset.csv"

# Feature Definitions
NUMERICAL_FEATURES = ["contains_link", "sender_trust_score", "link_reputation_score", "ai_generated_score"]
CATEGORICAL_FEATURES = ["language_type"]
TEXT_FEATURE = "message"

def preprocess_text(text):
    """Enhanced text preprocessing for multilingual support"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s/.:\-@\u0900-\u097F\u0C80-\u0CFF]", " ", text)
        return ' '.join(text.split())
    return ""

def load_and_prepare_data(filepath=DATASET_FILE):
    """Load and prepare the enhanced dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{filepath}' not found.")
        return None

    # Basic validation
    print("Dataset columns:", df.columns.tolist())
    print("Category distribution:")
    print(df['category'].value_counts())

    # Text preprocessing
    df[TEXT_FEATURE] = df[TEXT_FEATURE].astype(str).apply(preprocess_text)
    
    # Remove empty messages
    initial_count = len(df)
    df = df[df[TEXT_FEATURE].str.strip() != ""]
    print(f"Removed {initial_count - len(df)} empty messages")
    
    # Ensure numerical features are properly formatted
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Map categories to integers for 3-class classification
    df["label_main_int"] = df["category"].map({"ham": 0, "spam": 1, "promo": 2})
    
    # Handle missing subtypes
    df.loc[df["label_subtype"].isna(), "label_subtype"] = "Unknown"
    
    print(f"✅ Final dataset: {len(df)} samples")
    print("Final category distribution:")
    print(df['category'].value_counts())
    
    return df

def create_feature_pipeline():
    """Create optimized feature pipeline for multilingual text"""
    text_transformer = TfidfVectorizer(
        max_features=2000, 
        ngram_range=(1, 2),
        stop_words=None,
        min_df=2,
        max_df=0.8
    )
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, TEXT_FEATURE),
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        n_jobs=-1
    )
    return preprocessor

def train_main_3class_model(df):
    """Train the main 3-class classifier"""
    print("🔄 Preparing data for main 3-class model...")
    
    X = df[[TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df['label_main_int']
    
    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution: {y.value_counts().sort_index()}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Enhanced pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', create_feature_pipeline()),
        ('classifier', MultinomialNB(alpha=0.01, fit_prior=True))
    ])
    
    print("🔄 Training main 3-class model...")
    pipeline.fit(X_train, y_train)
    
    # Calculate metrics for saving
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save metrics to file for frontend
    metrics_data = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'test_samples': len(y_test),
        'training_samples': len(y_train)
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print("✅ Model metrics saved to model_metrics.json")
    
    return pipeline

def train_subtype_models(df):
    """Train separate subtype classifiers for spam and promo"""
    print("\n🔄 Training subtype classifiers...")
    subtype_pipelines = {}
    
    # Scam subtype classifier
    scam_df = df[df["category"] == "spam"].copy()
    if len(scam_df) > 0:
        scam_subtypes = scam_df["label_subtype"].value_counts()
        valid_scam_subtypes = scam_subtypes[scam_subtypes >= 3].index
        scam_df_filtered = scam_df[scam_df['label_subtype'].isin(valid_scam_subtypes)]
        
        if len(scam_df_filtered) > 1 and len(scam_df_filtered['label_subtype'].unique()) > 1:
            print(f"Training scam subtype classifier with {len(scam_df_filtered)} samples")
            
            X_scam = scam_df_filtered[[TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
            y_scam = scam_df_filtered["label_subtype"]
            
            scam_pipeline = Pipeline(steps=[
                ('preprocessor', create_feature_pipeline()),
                ('classifier', LinearSVC(random_state=42, dual=False, max_iter=2000, C=1.0, class_weight='balanced'))
            ])
            
            scam_pipeline.fit(X_scam, y_scam)
            subtype_pipelines['scam'] = scam_pipeline
    
    # Promo subtype classifier
    promo_df = df[df["category"] == "promo"].copy()
    if len(promo_df) > 0:
        promo_subtypes = promo_df["label_subtype"].value_counts()
        valid_promo_subtypes = promo_subtypes[promo_subtypes >= 3].index
        promo_df_filtered = promo_df[promo_df['label_subtype'].isin(valid_promo_subtypes)]
        
        if len(promo_df_filtered) > 1 and len(promo_df_filtered['label_subtype'].unique()) > 1:
            print(f"Training promo subtype classifier with {len(promo_df_filtered)} samples")
            
            X_promo = promo_df_filtered[[TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
            y_promo = promo_df_filtered["label_subtype"]
            
            promo_pipeline = Pipeline(steps=[
                ('preprocessor', create_feature_pipeline()),
                ('classifier', LinearSVC(random_state=42, dual=False, max_iter=2000, C=1.0, class_weight='balanced'))
            ])
            
            promo_pipeline.fit(X_promo, y_promo)
            subtype_pipelines['promo'] = promo_pipeline
    
    return subtype_pipelines

def save_models(main_pipeline, subtype_pipelines):
    """Save trained models with version info"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main model
    main_filename = "main_3class_pipeline.pkl"
    with open(main_filename, "wb") as f:
        pickle.dump(main_pipeline, f)
    print(f"✅ Main 3-class pipeline saved as {main_filename}")
    
    # Save subtype models
    for subtype, pipeline in subtype_pipelines.items():
        filename = f"{subtype}_subtype_pipeline.pkl"
        with open(filename, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"✅ {subtype.capitalize()} subtype pipeline saved as {filename}")
    
    # Save model info
    model_info = {
        'timestamp': timestamp,
        'dataset': DATASET_FILE,
        'features_used': [TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        'model_type': 'MultinomialNB (3-class) + LinearSVC (subtypes)'
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model info saved as model_info.json")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Enhanced 3-Class SMS Classifier Training")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None: 
        exit()
    
    print(f"\n📊 Dataset Overview:")
    print(f"Total samples: {df.shape[0]}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    
    # Train main 3-class model
    print("\n" + "=" * 60)
    print("🏗️ Training Main 3-Class Classifier...")
    main_pipeline = train_main_3class_model(df)
    
    # Train subtype models
    print("\n" + "=" * 60)
    print("🧩 Training Subtype Classifiers...")
    subtype_pipelines = train_subtype_models(df)
    
    # Save models
    print("\n" + "=" * 60)
    print("💾 Saving Models...")
    save_models(main_pipeline, subtype_pipelines)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("🎯 Models are ready for deployment!")
    print("=" * 60)
