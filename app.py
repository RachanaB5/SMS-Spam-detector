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
import json

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
main_3class_pipeline = None
scam_subtype_pipeline = None
promo_subtype_pipeline = None

try:
    with open('main_3class_pipeline.pkl', 'rb') as f:
        main_3class_pipeline = pickle.load(f)
    print("✅ Main 3-class pipeline loaded successfully!")
    
    # Try to load subtype pipelines if they exist
    try:
        with open('scam_subtype_pipeline.pkl', 'rb') as f:
            scam_subtype_pipeline = pickle.load(f)
        print("✅ Scam subtype pipeline loaded successfully!")
    except FileNotFoundError:
        print("⚠️  Scam subtype pipeline not found")
    
    try:
        with open('promo_subtype_pipeline.pkl', 'rb') as f:
            promo_subtype_pipeline = pickle.load(f)
        print("✅ Promo subtype pipeline loaded successfully!")
    except FileNotFoundError:
        print("⚠️  Promo subtype pipeline not found")
        
except Exception as e:
    print(f"❌ ERROR: Model files not found. Please train the models first. Error: {e}")

# --- Feature Extraction and Rules ---
RISKY_WORDS = ["click", "offer", "sale", "free", "account", "verify", "password", "urgent", "congratulations", "win", "reward", "loan", "deposit", "limited"]

# Enhanced fraud patterns for better classification
FRAUD_PATTERNS = {
    'financial_scams': {'keywords': ['bank', 'account', 'credit card', 'verify', 'suspend', 'block', 'loan', 'kyc', 'password', 'upi', 'transaction'], 'threat_level': 'HIGH'},
    'prize_clickbait_scams': {'keywords': ['won', 'winner', 'prize', 'lottery', 'congratulations', 'claim', 'gift card', 'click here', 'reward', 'selected', 'lucky'], 'threat_level': 'MEDIUM'},
    'phishing_urgency_scams': {'keywords': ['update', 'expires', 'urgent', 'immediately', 'act now', 'confirm', 'security', 'alert', 'suspended', 'verify now'], 'threat_level': 'HIGH'},
    'promotional_offers': {'keywords': ['sale', 'offer', 'discount', 'free', 'limited stock', 'shop now', 'deal', 'promo', 'buy now', 'special offer', 'flash sale'], 'threat_level': 'LOW'},
    'government_impersonation': {'keywords': ['government', 'irs', 'police', 'court', 'legal action', 'arrest', 'warrant', 'aadhaar', 'pan', 'income tax'], 'threat_level': 'CRITICAL'},
    'relationship_emotional_scams': {'keywords': ['love', 'miss you', 'emergency', 'help me', 'family', 'friend in need', 'stuck', 'hospital'], 'threat_level': 'MEDIUM'},
    'employment_opportunity_scams': {'keywords': ['job', 'work from home', 'salary', 'interview', 'hiring', 'opportunity', 'earn money', 'weekly income'], 'threat_level': 'MEDIUM'},
    'tech_support': {'keywords': ['tech support', 'virus', 'antivirus', 'update software', 'system scan'], 'threat_level': 'HIGH'},
    'service_update': {'keywords': ['order', 'track', 'delivery', 'shipped', 'appointment', 'bill', 'payment'], 'threat_level': 'LOW'},
}

def contains_link(text): 
    return int(bool(re.search(r"http\S+|www\.\S+", text)))

def find_suspicious_words(text): 
    return [w for w in RISKY_WORDS if w in text.lower()]

def detect_language(text):
    # Enhanced language detection
    text_lower = text.lower()
    
    # Hindi detection
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    # Kannada detection
    elif re.search(r"[\u0C80-\u0CFF]", text):
        return "Kannada"
    # Hinglish detection (mix of English and Hindi words)
    elif re.search(r"\b(accha|theek|hai|nahi|kyun|kya|kal|aaj)\b", text_lower):
        return "Hinglish"
    else:
        return "English"

def detect_ai_generated_score(text):
    patterns = ["congratulations", "urgent", "limited offer", "click here", "act now", "dear user", "verify now", "winner", "prize", "claim now"]
    score = sum(1 for p in patterns if p in text.lower()) / 10.0
    return round(min(score + random.uniform(0.0, 0.2), 0.99), 2)

def simulate_trust_scores(category, has_link):
    """Enhanced trust score simulation based on category"""
    if category == 'ham':
        sender_trust_score = round(random.uniform(0.7, 1.0), 2)
    elif category == 'promo':
        sender_trust_score = round(random.uniform(0.4, 0.8), 2)
    else:  # spam
        sender_trust_score = round(random.uniform(0.0, 0.4), 2)
    
    # Link reputation based on category and presence of link
    if has_link:
        if category == 'spam':
            link_reputation_score = round(random.uniform(0.0, 0.3), 2)
        elif category == 'promo':
            link_reputation_score = round(random.uniform(0.6, 0.9), 2)
        else:  # ham
            link_reputation_score = round(random.uniform(0.8, 1.0), 2)
    else:
        link_reputation_score = 1.0
    
    return sender_trust_score, link_reputation_score

def preprocess_text(text):
    """Enhanced text preprocessing for multilingual support"""
    if isinstance(text, str):
        # Convert to lowercase but preserve multilingual characters
        text = text.lower()
        
        # Keep essential punctuation and multilingual characters
        text = re.sub(r"[^a-zA-Z0-9\s/.:\-@\u0900-\u097F\u0C80-\u0CFF]", " ", text)
        
        # Remove extra whitespace
        return ' '.join(text.split())
    return ""

def extract_all_features(sms_text):
    """Extract all features for prediction"""
    has_link = contains_link(sms_text)
    ai_generated_score = detect_ai_generated_score(sms_text)
    language_type = detect_language(sms_text)
    
    # Initial trust scores (will be updated after prediction)
    sender_trust_score, link_reputation_score = simulate_trust_scores(None, has_link)
    
    data = {
        'message': preprocess_text(sms_text), 
        'contains_link': has_link, 
        'sender_trust_score': sender_trust_score, 
        'link_reputation_score': link_reputation_score, 
        'ai_generated_score': ai_generated_score, 
        'language_type': language_type, 
        'suspicious_words_found': find_suspicious_words(sms_text) 
    }
    return pd.DataFrame([data])

def get_explainability_features(df_features, prediction):
    """Enhanced explainability features"""
    explanations = {}
    meta = df_features.iloc[0]
    explanations['top_text_features'] = meta['suspicious_words_found']
    
    if meta['contains_link'] == 1:
        reputation = meta['link_reputation_score']
        if reputation < 0.3:
            explanations['link_status'] = f"⚠️ Suspicious Link (Reputation: {reputation:.2f})"
        elif reputation < 0.7:
            explanations['link_status'] = f"🟡 Medium Risk Link ({reputation:.2f})"
        else:
            explanations['link_status'] = f"🟢 Safe Link ({reputation:.2f})"
    
    if meta['sender_trust_score'] < 0.4: 
        explanations['sender_trust'] = f"🔴 Very Low Sender Trust Score ({meta['sender_trust_score']:.2f})"
    elif meta['sender_trust_score'] < 0.7:
        explanations['sender_trust'] = f"🟡 Medium Sender Trust Score ({meta['sender_trust_score']:.2f})"
    else:
        explanations['sender_trust'] = f"🟢 High Sender Trust Score ({meta['sender_trust_score']:.2f})"
    
    if meta['ai_generated_score'] > 0.7: 
        explanations['ai_template'] = f"🤖 Likely AI/Template Generated ({meta['ai_generated_score']:.2f})"
    
    if meta['language_type'] in ['Hinglish', 'Hindi', 'Kannada']: 
        explanations['language'] = f"🌐 Detected {meta['language_type']}"
    
    return explanations

def analyze_fraud_type_rule(text):
    """Enhanced fraud type analysis with better pattern matching"""
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

# --- Enhanced 3-class prediction logic ---
def predict_three_class(sms_text):
    """Enhanced prediction using 3-class system: spam, ham, promo"""
    if main_3class_pipeline is None:
        raise Exception("Main 3-class model not loaded")
    
    # Extract features
    df_features = extract_all_features(sms_text)
    
    # Get main 3-class prediction
    prediction_int = main_3class_pipeline.predict(df_features)[0]
    prediction_proba = main_3class_pipeline.predict_proba(df_features)[0]
    
    # Map integer prediction to class name
    class_mapping = {0: 'ham', 1: 'spam', 2: 'promo'}
    result = class_mapping[prediction_int]
    confidence = float(prediction_proba[prediction_int]) * 100
    
    # Update trust scores based on prediction
    has_link = df_features['contains_link'].iloc[0]
    df_features['sender_trust_score'], df_features['link_reputation_score'] = simulate_trust_scores(result, has_link)
    
    # Determine fraud type and threat level
    fraud_type = None
    threat_level = 'LOW'
    
    if result == 'spam':
        fraud_type, threat_level = analyze_fraud_type_rule(sms_text)
        # Try to use scam subtype classifier if available
        if scam_subtype_pipeline is not None:
            try:
                fraud_type = scam_subtype_pipeline.predict(df_features)[0]
            except:
                pass  # Fall back to rule-based analysis
    
    elif result == 'promo':
        fraud_type = 'promotional_offers'
        threat_level = 'LOW'
        # Try to use promo subtype classifier if available
        if promo_subtype_pipeline is not None:
            try:
                fraud_type = promo_subtype_pipeline.predict(df_features)[0]
            except:
                pass  # Fall back to default
    else:  # ham
        fraud_type = 'personal'
        threat_level = 'LOW'
    
    return result, confidence, fraud_type, threat_level, df_features

# -----------------------------------------------------------
# 🔹 API Endpoints (Enhanced for 3-class system)
# -----------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running', 
        'message': 'Enhanced 3-Class SMS Classifier (Spam/Ham/Promo)', 
        'model_classes': ['ham', 'spam', 'promo'],
        'main_model_loaded': main_3class_pipeline is not None,
        'scam_subtype_loaded': scam_subtype_pipeline is not None,
        'promo_subtype_loaded': promo_subtype_pipeline is not None,
        'version': '2.0',
        'features': ['multilingual_support', 'enhanced_trust_scores', 'subtype_classification']
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sms_text = data.get('sms_text', '')
        
        if not sms_text: 
            return jsonify({'error': 'No SMS text provided'}), 400
        
        if main_3class_pipeline is None: 
            return jsonify({'error': 'Models not loaded. Please train the models first.'}), 500
        
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
        result, confidence, fraud_type, threat_level, df_features = predict_three_class(sms_text)
        
        # Extract features for explainability
        explanations = get_explainability_features(df_features, result)
        
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
            'risk_score': round(confidence, 2) if result == 'spam' else 0,
            'timestamp': datetime.now().isoformat(),
            'language': df_features['language_type'].iloc[0],
            'top_features': explanations.get('top_text_features', [])[:5]
        }
        
        # Add category-specific fields
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
                'threat_level': threat_level,
                'explainability': explanations
            })
        else:  # ham
            response_data.update({
                'fraud_type': fraud_type,
                'explainability': explanations
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
        
        # Calculate percentages
        spam_percentage = round((spam_count / total * 100), 2) if total > 0 else 0
        ham_percentage = round((ham_count / total * 100), 2) if total > 0 else 0
        promo_percentage = round((promo_count / total * 100), 2) if total > 0 else 0
        
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
            'spam_percentage': spam_percentage,
            'ham_percentage': ham_percentage,
            'promo_percentage': promo_percentage,
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

@app.route('/model-info', methods=['GET'])
def get_model_info():
    try:
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
    except:
        model_info = {'error': 'Model info not available'}
    
    return jsonify({
        'model_info': model_info,
        'loaded_models': {
            'main_3class': main_3class_pipeline is not None,
            'scam_subtype': scam_subtype_pipeline is not None,
            'promo_subtype': promo_subtype_pipeline is not None
        },
        'features_supported': ['multilingual', 'trust_scores', 'link_analysis', 'ai_detection']
    })

@app.route('/model-metrics', methods=['GET'])
def get_model_metrics():
    """Endpoint to get model performance metrics"""
    try:
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except FileNotFoundError:
        return jsonify({'error': 'Model metrics not available. Please train the model first.'}), 404

if __name__ == '__main__':
    print("🚀 Starting Enhanced 3-Class SMS Classifier on port 6008...")
    print("📊 Enhanced Model Status:")
    print(f"   - Main 3-class model: {'✅ Loaded' if main_3class_pipeline else '❌ Not Found'}")
    print(f"   - Scam subtype model: {'✅ Loaded' if scam_subtype_pipeline else '❌ Not Found'}")
    print(f"   - Promo subtype model: {'✅ Loaded' if promo_subtype_pipeline else '❌ Not Found'}")

# Global error handler to ensure all errors return JSON
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e)}), 500

    print("🌐 Features: Multilingual Support, Enhanced Trust Scores, Subtype Classification")
    app.run(debug=True, host='0.0.0.0', port=6004)
