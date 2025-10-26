from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
import re
import random
import os
import threading 

# --- Configuration ---
app = Flask(__name__)
CORS(app)
DB_NAME = 'sms_predictions.db'

# --- Database Setup and Asynchronous Logging ---

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            sms_text TEXT NOT NULL, 
            prediction TEXT NOT NULL, 
            confidence REAL NOT NULL, 
            threat_level TEXT, 
            fraud_type TEXT, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            sms_text TEXT NOT NULL, 
            reported_as TEXT NOT NULL, 
            user_comment TEXT, 
            phone_number TEXT, 
            feedback_type TEXT, 
            prediction_id INTEGER, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS threat_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            pattern_text TEXT NOT NULL, 
            fraud_type TEXT NOT NULL, 
            threat_level TEXT NOT NULL, 
            count INTEGER DEFAULT 1, 
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blocked_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            entity TEXT NOT NULL UNIQUE, 
            entity_type TEXT NOT NULL, 
            reason TEXT, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_prediction_async(sms_text, result, confidence, threat_level, fraud_type, prediction_id=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO predictions (sms_text, prediction, confidence, threat_level, fraud_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (sms_text, result, confidence, threat_level, fraud_type))
        
        if result == 'spam' and fraud_type:
            pattern_key = sms_text[:100]
            cursor.execute('''
                INSERT OR REPLACE INTO threat_patterns (pattern_text, fraud_type, threat_level, count, last_seen)
                VALUES (?, ?, ?, 
                    COALESCE((SELECT count + 1 FROM threat_patterns WHERE pattern_text = ?), 1),
                    CURRENT_TIMESTAMP)
            ''', (pattern_key, fraud_type, threat_level, pattern_key))
        conn.commit()
    except Exception as e:
        print(f"Error during async DB log: {e}")
    finally:
        conn.close()

# --- Model Loading ---
binary_pipeline = None
subtype_pipeline = None
vectorizer = None

try:
    with open('combined_binary_pipeline.pkl', 'rb') as f:
        binary_pipeline = pickle.load(f)
    with open('combined_subtype_pipeline.pkl', 'rb') as f:
        subtype_pipeline = pickle.load(f) 
    
    preprocessor = binary_pipeline.named_steps['preprocessor']
    vectorizer = preprocessor.named_transformers_['text']
    print("Model pipelines loaded successfully! (MNB + LinearSVC)")
except Exception as e:
    print(f"WARNING: Model files not found. Using dummy model. Error: {e}")

# --- Feature Extraction and Rules ---
RISKY_WORDS = ["click", "offer", "sale", "free", "account", "verify", "password", "urgent", "congratulations", "win", "reward", "loan", "deposit", "limited"]

# UPDATED: Added promo patterns
FRAUD_PATTERNS = {
    'financial_scams': {'keywords': ['bank', 'account', 'credit card', 'verify', 'suspend', 'block', 'loan', 'kyc', 'password'], 'threat_level': 'HIGH'},
    'prize_clickbait_scams': {'keywords': ['won', 'winner', 'prize', 'lottery', 'congratulations', 'claim', 'gift card', 'click here', 'reward'], 'threat_level': 'MEDIUM'},
    'phishing_urgency_scams': {'keywords': ['update', 'expires', 'urgent', 'immediately', 'act now', 'confirm', 'security', 'alert'], 'threat_level': 'HIGH'},
    'promotional_offers': {'keywords': ['sale', 'offer', 'discount', 'free', 'limited stock', 'shop now', 'deal', 'promo', 'buy now', 'special offer'], 'threat_level': 'LOW'},
    'government_impersonation': {'keywords': ['government', 'irs', 'police', 'court', 'legal action', 'arrest', 'warrant', 'aadhaar', 'pan'], 'threat_level': 'CRITICAL'},
    'relationship_emotional_scams': {'keywords': ['love', 'miss you', 'emergency', 'help me', 'family', 'friend in need'], 'threat_level': 'MEDIUM'},
    'employment_opportunity_scams': {'keywords': ['job', 'work from home', 'salary', 'interview', 'hiring', 'opportunity', 'earn money'], 'threat_level': 'MEDIUM'},
}

def contains_link(text): 
    return int(bool(re.search(r"http\S+|www\.\S+", text)))

def find_suspicious_words(text): 
    return [w for w in RISKY_WORDS if w in text.lower()]

def detect_language(text):
    if re.search(r"\b(hai|na|kal|kya|kyc|kyun|yaar)\b", text.lower()): 
        return "Hinglish"
    elif any(ord(c) > 127 for c in text): 
        return "Regional"
    else: 
        return "English"

def detect_ai_generated_score(text):
    patterns = ["congratulations", "urgent", "limited offer", "click here", "act now", "dear user", "verify now"]
    score = sum(1 for p in patterns if p in text.lower()) / 7.0
    return round(min(score + random.uniform(0.0, 0.2), 0.99), 2)

def simulate_trust_scores(label_main, has_link):
    sender_trust_score = round(random.uniform(0.3, 0.6), 2)
    link_reputation_score = round(random.uniform(0.1, 0.5), 2) if has_link else 1.0 
    return sender_trust_score, link_reputation_score

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s/.:-]", " ", text) 
        return ' '.join(text.split())
    return ""

def extract_all_features(sms_text):
    has_link = contains_link(sms_text)
    sender_trust_score, link_reputation_score = simulate_trust_scores(None, has_link)
    ai_generated_score = detect_ai_generated_score(sms_text)
    data = {
        'message': preprocess_text(sms_text), 
        'contains_link': has_link, 
        'sender_trust_score': sender_trust_score, 
        'link_reputation_score': link_reputation_score, 
        'ai_generated_score': ai_generated_score, 
        'language_type': detect_language(sms_text), 
        'suspicious_words_found': find_suspicious_words(sms_text) 
    }
    return pd.DataFrame([data])

def get_explainability_features(df_features, X_vec):
    explanations = {}
    meta = df_features.iloc[0]
    explanations['top_text_features'] = meta['suspicious_words_found']
    
    if meta['contains_link'] == 1:
        reputation = meta['link_reputation_score']
        explanations['link_status'] = f"⚠️ Phishing Link (Reputation: {reputation:.2f})" if reputation < 0.3 else f"🟡 Medium Risk Link ({reputation:.2f})"
    
    if meta['sender_trust_score'] < 0.4: 
        explanations['sender_trust'] = f"🔴 Very Low Sender Trust Score ({meta['sender_trust_score']:.2f})"
    
    if meta['ai_generated_score'] > 0.8: 
        explanations['ai_template'] = f"🤖 High Likelihood of AI/Template Generation ({meta['ai_generated_score']:.2f})"
    
    if meta['language_type'] in ['Hinglish', 'Regional']: 
        explanations['language'] = f"🌐 Detected {meta['language_type']}"
    
    return explanations

def analyze_fraud_type_rule(text):
    text_lower = text.lower()
    fraud_scores = {}
    for fraud_type, data in FRAUD_PATTERNS.items():
        score = sum(1 for keyword in data['keywords'] if keyword in text_lower)
        if score > 0:
            fraud_scores[fraud_type] = (score, data['threat_level'])
    
    if fraud_scores:
        fraud_type = max(fraud_scores.keys(), key=lambda k: fraud_scores[k][0])
        return fraud_type, fraud_scores[fraud_type][1]
    
    return 'general_spam', 'LOW'

def extract_phone_numbers(text): 
    return re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b', text)

def extract_urls(text): 
    return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

def check_blocked_entities(text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT entity FROM blocked_entities')
    blocked = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    blocked_found = []
    for entity in extract_phone_numbers(text) + extract_urls(text):
        if entity in blocked: 
            blocked_found.append(('phone' if re.match(r'\b\d', entity) else 'url', entity))
    return blocked_found

# --- UPDATED: 3-class prediction logic ---
def predict_three_class(sms_text):
    """Predict using 3-class system: spam, ham, promo"""
    df_features = extract_all_features(sms_text)
    preprocessor = binary_pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(df_features)
    
    # Get binary prediction first
    classifier = binary_pipeline.named_steps['classifier']
    binary_pred_int = classifier.predict(X_transformed)[0]
    binary_proba = classifier.predict_proba(X_transformed)[0]
    
    # Convert to 3-class system
    if binary_pred_int == 0:  # ham
        return 'ham', float(binary_proba[0]) * 100, None, 'LOW'
    
    else:  # spam/promo - use subtype classifier to distinguish
        subtype_classifier = subtype_pipeline.named_steps['classifier']
        subtype = subtype_classifier.predict(X_transformed)[0]
        
        # Determine if it's promo or spam based on subtype
        if subtype == 'promotional_offers':
            return 'promo', float(binary_proba[1]) * 100, subtype, 'LOW'
        else:
            fraud_type_rule, threat_level = analyze_fraud_type_rule(sms_text)
            return 'spam', float(binary_proba[1]) * 100, subtype, threat_level

# -----------------------------------------------------------
# 🔹 API Endpoints (Updated for 3-class system)
# -----------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running', 
        'message': '3-Class SMS Classifier (Spam/Ham/Promo)', 
        'binary_model_loaded': binary_pipeline is not None,
        'subtype_model_loaded': subtype_pipeline is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sms_text = data.get('sms_text', '')
        
        if not sms_text: 
            return jsonify({'error': 'No SMS text provided'}), 400
        
        if binary_pipeline is None or subtype_pipeline is None: 
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Check for blocked entities
        blocked = check_blocked_entities(sms_text)
        if blocked: 
            return jsonify({
                'sms_text': sms_text, 
                'prediction': 'spam', 
                'confidence': 100.0, 
                'threat_level': 'BLOCKED', 
                'fraud_type': 'blocked_entity', 
                'explainability': {'warning': 'Contained blocked entity.'}, 
                'timestamp': datetime.now().isoformat()
            })
        
        # Get 3-class prediction
        result, confidence, fraud_type, threat_level = predict_three_class(sms_text)
        
        # Extract features for explainability
        df_features = extract_all_features(sms_text)
        explanations = get_explainability_features(df_features, None)
        
        # ASYNCHRONOUS LOGGING
        threading.Thread(
            target=log_prediction_async, 
            args=(sms_text, result, confidence, threat_level, fraud_type)
        ).start()
        
        # Build response
        response_data = {
            'sms_text': sms_text, 
            'prediction': result, 
            'confidence': round(confidence, 2),
            'risk_score': round(confidence, 2),  # For frontend compatibility
            'timestamp': datetime.now().isoformat(),
            'top_features': explanations.get('top_text_features', [])[:5]  # Top 5 features
        }
        
        # Add spam-specific fields
        if result == 'spam':
            response_data.update({
                'fraud_type': fraud_type,
                'threat_level': threat_level,
                'extracted_phones': extract_phone_numbers(sms_text),
                'extracted_urls': extract_urls(sms_text),
                'explainability': explanations
            })
        elif result == 'promo':
            response_data.update({
                'fraud_type': fraud_type,
                'threat_level': threat_level
            })
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['POST'])
def report_spam():
    try:
        data = request.get_json()
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_reports (sms_text, reported_as, user_comment, phone_number, feedback_type, prediction_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data.get('sms_text', ''), 
            data.get('reported_as', 'spam'), 
            data.get('comment', ''), 
            data.get('phone_number', ''), 
            data.get('feedback_type', None), 
            data.get('prediction_id', None)
        ))
        conn.commit()
        conn.close()
        return jsonify({
            'message': 'Thank you for your report! This helps protect the community.', 
            'status': 'success'
        })
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/block', methods=['POST'])
def block_entity():
    try:
        data = request.get_json()
        entity = data.get('entity', '')
        entity_type = data.get('type', 'phone')
        reason = data.get('reason', 'User reported')
        
        if not entity: 
            return jsonify({'error': 'No entity provided'}), 400
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO blocked_entities (entity, entity_type, reason)
                VALUES (?, ?, ?)
            ''', (entity, entity_type, reason))
            conn.commit()
            message = f'{entity_type.title()} blocked successfully'
        except sqlite3.IntegrityError: 
            message = f'{entity_type.title()} already blocked'
        conn.close()
        
        return jsonify({'message': message, 'status': 'success'})
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/threats', methods=['GET'])
def get_threats():
    try:
        days = request.args.get('days', 7, type=int)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT fraud_type, threat_level, COUNT(*) as count FROM threat_patterns
            WHERE last_seen >= datetime('now', '-' || ? || ' days')
            GROUP BY fraud_type, threat_level ORDER BY count DESC
        ''', (days,))
        threats = [{'fraud_type': row[0], 'threat_level': row[1], 'count': row[2]} for row in cursor.fetchall()]
        conn.close()
        return jsonify({'threats': threats})
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/community-reports', methods=['GET'])
def get_community_reports():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        total_reports = cursor.execute('SELECT COUNT(*) FROM user_reports').fetchone()[0]
        reports_by_type = dict(cursor.execute('SELECT reported_as, COUNT(*) FROM user_reports GROUP BY reported_as').fetchall())
        recent_reports = cursor.execute("SELECT COUNT(*) FROM user_reports WHERE timestamp >= datetime('now', '-7 days')").fetchone()[0]
        conn.close()
        return jsonify({
            'total_reports': total_reports, 
            'reports_by_type': reports_by_type, 
            'recent_reports': recent_reports
        })
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        limit = request.args.get('limit', 100, type=int)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, sms_text, prediction, confidence, threat_level, fraud_type, timestamp 
            FROM predictions ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        history = [{
            'id': row[0], 
            'sms_text': row[1], 
            'prediction': row[2], 
            'confidence': row[3], 
            'threat_level': row[4], 
            'fraud_type': row[5], 
            'timestamp': row[6]
        } for row in cursor.fetchall()]
        conn.close()
        return jsonify({'history': history})
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get counts for all three categories
        total = cursor.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        spam_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='spam'").fetchone()[0]
        ham_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='ham'").fetchone()[0]
        promo_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='promo'").fetchone()[0]
        
        threat_distribution = dict(cursor.execute(
            "SELECT threat_level, COUNT(*) FROM predictions WHERE threat_level IS NOT NULL GROUP BY threat_level"
        ).fetchall())
        
        fraud_distribution = dict(cursor.execute(
            "SELECT fraud_type, COUNT(*) FROM predictions WHERE fraud_type IS NOT NULL GROUP BY fraud_type"
        ).fetchall())
        
        community_reports = cursor.execute('SELECT COUNT(*) FROM user_reports').fetchone()[0]
        blocked_count = cursor.execute('SELECT COUNT(*) FROM blocked_entities').fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_predictions': total,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'promo_count': promo_count,
            'spam_percentage': round((spam_count / total * 100), 2) if total > 0 else 0,
            'ham_percentage': round((ham_count / total * 100), 2) if total > 0 else 0,
            'promo_percentage': round((promo_count / total * 100), 2) if total > 0 else 0,
            'threat_distribution': threat_distribution,
            'fraud_distribution': fraud_distribution,
            'community_reports': community_reports,
            'blocked_entities': blocked_count
        })
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/education', methods=['GET'])
def get_education():
    education_content = {
        'financial_scams': {
            'title': 'Financial/Banking Fraud', 
            'description': 'Scammers pretending to be from your bank or financial institution', 
            'red_flags': ['Urgent requests to verify account information', 'Threats of account suspension', 'Requests for PIN or password', 'Unusual payment requests'], 
            'action': 'Never share banking details via SMS. Contact your bank directly.'
        },
        'prize_clickbait_scams': {
            'title': 'Prize/Lottery Scams', 
            'description': 'Fake notifications about winning prizes or lotteries', 
            'red_flags': ['Winning contests you never entered', 'Requests for fees to claim prizes', 'Urgent deadlines to claim', 'Too good to be true offers'], 
            'action': 'Legitimate prizes never require upfront payment.'
        },
        'phishing_urgency_scams': {
            'title': 'Phishing Attacks', 
            'description': 'Attempts to steal personal information through fake links', 
            'red_flags': ['Suspicious links or shortened URLs', 'Urgent action required', 'Poor grammar or spelling', 'Requests to click immediately'], 
            'action': 'Never click suspicious links. Verify sender identity first.'
        },
        'government_impersonation': {
            'title': 'Government/Authority Impersonation', 
            'description': 'Scammers pretending to be government officials or police', 
            'red_flags': ['Threats of arrest or legal action', 'Demands for immediate payment', 'Requests for personal information', 'Unusual payment methods (gift cards, crypto)'], 
            'action': 'Government agencies never demand immediate payment via SMS.'
        },
        'promotional_offers': {
            'title': 'Promotional Messages', 
            'description': 'Legitimate marketing messages from businesses', 
            'red_flags': ['Too good to be true discounts', 'Pressure to act immediately', 'Requests for personal information'], 
            'action': 'Verify the sender and check official websites before responding to offers.'
        }
    }
    return jsonify({'education': education_content})

@app.route('/blocked-entities', methods=['GET'])
def get_blocked_entities():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT entity, entity_type, reason, timestamp FROM blocked_entities ORDER BY timestamp DESC')
        entities = [{
            'entity': row[0],
            'type': row[1],
            'reason': row[2],
            'timestamp': row[3]
        } for row in cursor.fetchall()]
        conn.close()
        return jsonify({'blocked_entities': entities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting 3-Class SMS Classifier on port 6008...")
    app.run(debug=True, host='0.0.0.0', port=6006)