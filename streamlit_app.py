import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re

# --- Configuration & Initialization ---
API_URL = "http://localhost:6004"

st.set_page_config(
    page_title="Enhanced SMS Spam Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Enhanced Custom Styling ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;900&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp {
        font-family: 'Roboto', 'Montserrat', Arial, sans-serif !important;
        background: linear-gradient(120deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* Light mode styles */
    body, .stApp {
        color: #222 !important;
    }
    .main-title, .sub-title, .category-badge, .language-badge, .spam-result, .metric-card, .stMetric, .stDataFrame, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stTextArea textarea {
        color: #222 !important;
    }
    .spam-result { background-color: #ffeaea; }
    .ham { background-color: #eaffea; }
    .promo { background-color: #fffbe5; }
    .stTabs [data-baseweb="tab"] { background: #fff; color: #B22222; }
    .stTextArea textarea { background: #fffbe5; border: 2px solid #FFD700; }
    .stMetric, .metric-card { background: #fffbe5; }
    .stDataFrame { background: #fff; }
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        html, body, .stApp {
            background: linear-gradient(120deg, #23272f 0%, #181a20 100%) !important;
        }
        body, .stApp {
            color: #fff !important;
        }
        .main-title, .sub-title, .category-badge, .language-badge, .spam-result, .metric-card, .stMetric, .stDataFrame, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stTextArea textarea {
            color: #fff !important;
        }
        .spam-result { background-color: #3a2323 !important; }
        .ham { background-color: #233a23 !important; }
        .promo { background-color: #3a3923 !important; }
        .stTabs [data-baseweb="tab"] { background: #23272f !important; color: #FFD700 !important; }
        .stTextArea textarea { background: #23272f !important; border: 2px solid #FFD700 !important; }
        .stMetric, .metric-card { background: #23272f !important; }
        .stDataFrame { background: #23272f !important; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #FFD700 !important; }
    }
    /* --- Existing styles --- */
    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        text-align: center;
        color: #B22222;
        margin-bottom: 0.25rem;
        padding-top: 1rem;
        letter-spacing: 1px;
        text-shadow: 1px 2px 8px #e0e0e0;
    }
    .sub-title {
        font-family: 'Roboto', sans-serif;
        text-align: center;
        color: #444;
        margin-bottom: 2rem;
        font-size: 1.25rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .spam-result {
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 30px 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        font-family: 'Montserrat', sans-serif;
        transition: box-shadow 0.2s;
    }
    .spam-result:hover {
        box-shadow: 0 10px 32px rgba(178,34,34,0.12);
    }
    .spam { background-color: #ffeaea; color: #8B0000; border: 3px solid #FF4500; }
    .ham { background-color: #eaffea; color: #006400; border: 3px solid #3CB371; }
    .promo { background-color: #fffbe5; color: #b8860b; border: 3px solid #FFD700; }
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: bold;
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(90deg,#B22222 0%,#FF8C00 100%);
        color: #fff;
        box-shadow: 0 2px 8px rgba(178,34,34,0.08);
        transition: background 0.2s, box-shadow 0.2s;
        border: none;
        padding: 0.7em 2em;
        font-size: 1.1rem;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg,#FF8C00 0%,#B22222 100%);
        box-shadow: 0 4px 16px rgba(178,34,34,0.18);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 18px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.18rem;
        font-weight: 700;
        padding: 12px 28px;
        border-radius: 12px 12px 0 0;
        font-family: 'Montserrat', sans-serif;
        background: #fff;
        color: #B22222;
        transition: background 0.2s, color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #ffeaea;
        color: #FF8C00;
    }
    .category-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 18px;
        font-weight: 700;
        font-size: 1.05rem;
        margin: 2px;
        font-family: 'Montserrat', sans-serif;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .language-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 14px;
        font-size: 0.92rem;
        background: #e0e0e0;
        margin-left: 10px;
        font-family: 'Roboto', sans-serif;
        color: #333;
    }
    .metric-card {
        background: #fff;
        padding: 22px;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,123,255,0.07);
        font-family: 'Montserrat', sans-serif;
    }
    .stMetric {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 1.15rem !important;
        color: #B22222 !important;
        background: #fffbe5 !important;
        border-radius: 10px !important;
        padding: 10px 0 !important;
        margin: 6px 0 !important;
        box-shadow: 0 2px 8px rgba(255,215,0,0.07);
    }
    .stDataFrame {
        font-family: 'Roboto', sans-serif !important;
        font-size: 1.05rem !important;
        background: #fff !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .stMarkdown h3, .stMarkdown h2, .stMarkdown h1 {
        font-family: 'Montserrat', sans-serif !important;
        color: #B22222 !important;
        font-weight: 900 !important;
        letter-spacing: 0.5px;
    }
    .stMarkdown h4 {
        font-family: 'Montserrat', sans-serif !important;
        color: #FF8C00 !important;
        font-weight: 700 !important;
    }
    .stTextArea textarea {
        font-family: 'Roboto', sans-serif !important;
        font-size: 1.08rem !important;
        background: #fffbe5 !important;
        border-radius: 10px !important;
        border: 2px solid #FFD700 !important;
        padding: 12px !important;
        color: #333 !important;
    }
    /* Custom scrollbar for tables */
    ::-webkit-scrollbar {
        width: 8px;
        background: #e0e0e0;
        border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #FFD700;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title"> Enhanced SMS Detective</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced 3-Class AI Classification | Spam • Ham • Promo | Multilingual Support</div>', unsafe_allow_html=True)

# --- Tabs ---
tab_home, tab_stats, tab_history, tab_visualizations = st.tabs(["🏠 Live Detector", "📊 Statistics", "📜 History", "📈 Analytics"])

# --- Helper Functions ---
@st.cache_resource
def load_assets():
    try:
        if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, vectorizer = load_assets()

def fetch_api(endpoint, params={}):
    try:
        response = requests.get(f"{API_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Backend connection failed! Ensure Flask API is running on {API_URL}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.status_code} - {e.response.json().get('error', 'Unknown HTTP error')}")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        return None

def calculate_model_metrics():
    """Calculate model performance metrics from training data"""
    try:
        # Load the enhanced_5000_dataset.csv
        path = 'enhanced_5000_dataset.csv'
        if not os.path.exists(path):
            st.error("enhanced_5000_dataset.csv not found for model evaluation")
            return None, None, None, None
        df = pd.read_csv(path)
        st.success(f"Loaded dataset: {path}")
        
        # Prepare data
        required_cols = ['message', 'category']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Required columns {required_cols} not found in dataset")
            return None, None, None, None
        
        df_clean = df[required_cols].copy()
        df_clean = df_clean.dropna()
        df_clean = df_clean[df_clean['message'].astype(str).str.strip() != '']
        
        # Standardize labels
        def map_label(label):
            label_str = str(label).lower().strip()
            if 'spam' in label_str or 'scam' in label_str:
                return 'spam'
            elif 'ham' in label_str or 'legit' in label_str:
                return 'ham'
            elif 'promo' in label_str or 'offer' in label_str:
                return 'promo'
            else:
                return label_str
        
        df_clean['label'] = df_clean['category'].apply(map_label)
        df_final = df_clean[df_clean['label'].isin(['spam', 'ham', 'promo'])].copy()
        
        if len(df_final) == 0:
            st.error("No valid spam/ham/promo messages found")
            return None, None, None, None
        
        X = df_final['message']
        y = df_final['label']
        
        # Convert labels to numeric
        label_mapping = {'spam': 0, 'ham': 1, 'promo': 2}
        y_numeric = y.map(label_mapping)
        
        # Check if we have the model and vectorizer
        if model is None or vectorizer is None:
            st.warning("Using fallback model for evaluation...")
            # Create a simple model for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
            )
            
            vectorizer_fallback = TfidfVectorizer(max_features=1000, stop_words='english')
            X_train_tfidf = vectorizer_fallback.fit_transform(X_train)
            X_test_tfidf = vectorizer_fallback.transform(X_test)
            
            model_fallback = MultinomialNB()
            model_fallback.fit(X_train_tfidf, y_train)
            
            y_pred = model_fallback.predict(X_test_tfidf)
            cm = confusion_matrix(y_test, y_pred)
            
        else:
            # Use the loaded model
            X_transformed = vectorizer.transform(X)
            y_pred = model.predict(X_transformed)
            cm = confusion_matrix(y_numeric, y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_numeric, y_pred)
        precision = precision_score(y_numeric, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_numeric, y_pred, average='weighted', zero_division=0)
        
        return accuracy, precision, recall, cm
        
    except Exception as e:
        st.error(f"Error calculating model metrics: {e}")
        return None, None, None, None

# Fraud type labels with enhanced descriptions
FRAUD_LABELS = {
    # Spam types
    'financial_scams': ('Financial Scam', '#B22222', 'Scams targeting bank accounts, loans, or financial information.'),
    'prize_clickbait_scams': ('Prize/Lottery Scam', '#8B008B', 'Fake prize notifications or lottery winnings.'),
    'phishing_urgency_scams': ('Phishing Scam', '#FF8C00', 'Urgent requests for verification or personal information.'),
    'government_impersonation': ('Govt Impersonation', '#4682B4', 'Scammers pretending to be government officials.'),
    'relationship_emotional_scams': ('Relationship Scam', '#C71585', 'Emotional manipulation for financial gain.'),
    'employment_opportunity_scams': ('Job Scam', '#228B22', 'Fake job offers or work-from-home opportunities.'),
    'tech_support': ('Tech Support Scam', '#2E8B57', 'Fake tech support or software update requests.'),
    'general_spam': ('General Spam', '#666666', 'Uncategorized spam messages.'),
    
    # Promo types
    'promotional_offers': ('Promotional Offer', '#FFD700', 'Legitimate marketing offers and discounts.'),
    'service_update': ('Service Update', '#32CD32', 'Order tracking, delivery updates, or service notifications.'),
    'flash_sale': ('Flash Sale', '#FF69B4', 'Limited time offers and flash sales.'),
    'general_promo': ('General Promotion', '#FFA500', 'Marketing promotions and advertisements.'),
    
    # Ham types
    'personal': ('Personal Message', '#006400', 'Personal communication between individuals.'),
    'service_update': ('Service Update', '#32CD32', 'Legitimate service notifications and updates.')
}

# --- Tab: Live Detector ---
with tab_home:
    st.markdown("### 💬 Real-time SMS Classification")

    if 'sms_input' not in st.session_state:
        st.session_state.sms_input = ""

    sms_text = st.text_area(
        "Type or paste your SMS message here:",
        height=150,
        placeholder="Example: Congratulations! You've won a ₹1000 prize. Text WIN to 87121.",
        key='sms_input_area',
        value=st.session_state.sms_input
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button("🚀 Classify Message", use_container_width=True, type="primary")

    if predict_button:
        if not sms_text.strip():
            st.error("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner("🔍 Analyzing SMS with enhanced classification..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"sms_text": sms_text}
                    )
                    response.raise_for_status()
                    result = response.json()
                    prediction = result.get('prediction')
                    confidence = result.get('confidence', 0.0)
                    risk_score = result.get('risk_score', confidence)
                    fraud_type = result.get('fraud_type', 'Unknown')
                    threat_level = result.get('threat_level', 'Unknown')
                    language = result.get('language', 'Unknown')

                    fraud_label, fraud_color, fraud_tip = FRAUD_LABELS.get(fraud_type, (fraud_type, '#666666', 'Unknown type'))

                    # Display result based on prediction category
                    if prediction == 'spam':
                        st.markdown(
                            f'<div style="margin-bottom:10px;text-align:center;">'
                            f'<span class="category-badge" style="background:{fraud_color};color:#fff;">{fraud_label}</span>'
                            f'<span class="language-badge">{language}</span>'
                            f'<div style="margin-top:6px;color:#444;font-size:0.98rem;">{fraud_tip}</div>'
                            f'</div>', unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="spam-result spam">'
                            f'🚨 <b>SPAM DETECTED</b><br>'
                            f'<span style="color:#8B0000;font-weight:600;">Threat Level: {threat_level}</span><br>'
                            f'Risk Score: {risk_score:.2f}%'
                            f'</div>', unsafe_allow_html=True
                        )
                        st.error("⚠️ **Action Required:** This message exhibits high-risk characteristics. Do NOT click any links.")
                    
                    elif prediction == 'promo':
                        st.markdown(
                            f'<div style="margin-bottom:10px;text-align:center;">'
                            f'<span class="category-badge" style="background:{fraud_color};color:#fff;">{fraud_label}</span>'
                            f'<span class="language-badge">{language}</span>'
                            f'<div style="margin-top:6px;color:#444;font-size:0.98rem;">{fraud_tip}</div>'
                            f'</div>', unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="spam-result promo">'
                            f'🎁 <b>PROMOTIONAL MESSAGE</b><br>'
                            f'<span style="color:#b8860b;font-weight:600;">Promo Type: {fraud_label}</span><br>'
                            f'Confidence: {confidence:.2f}%'
                            f'</div>', unsafe_allow_html=True
                        )
                        st.info("ℹ️ **Note:** This is a promotional message. Exercise caution before engaging with offers.")
                    
                    else:  # ham
                        st.markdown(
                            f'<div style="margin-bottom:10px;text-align:center;">'
                            f'<span class="category-badge" style="background:{fraud_color};color:#fff;">{fraud_label}</span>'
                            f'<span class="language-badge">{language}</span>'
                            f'</div>', unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="spam-result ham">'
                            f'✅ <b>LEGITIMATE MESSAGE</b><br>'
                            f'Confidence: {confidence:.2f}%'
                            f'</div>', unsafe_allow_html=True
                        )
                        st.success("✓ **Status:** This message appears safe and legitimate.")

                    # Enhanced Confidence Meter
                    st.markdown("### Confidence Breakdown")
                    col_gauge, col_explainer = st.columns([1,2])
                    with col_gauge:
                        # Dynamic gauge color based on prediction
                        if prediction == 'spam':
                            gauge_color = "#8B0000"
                            threshold = 75
                        elif prediction == 'promo':
                            gauge_color = "#FFD700"
                            threshold = 60
                        else:  # ham
                            gauge_color = "#006400"
                            threshold = 85

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence,
                            domain={'x':[0,1], 'y':[0,1]},
                            title={'text': f"{prediction.upper()} Confidence"},
                            gauge={
                                'axis': {'range':[0,100], 'tickwidth':1, 'tickcolor':"darkblue"},
                                'bar': {'color': gauge_color},
                                'steps':[
                                    {'range':[0,40],'color':"lightgreen"},
                                    {'range':[40,70],'color':"yellow"},
                                    {'range':[70,100],'color':"salmon"}
                                ],
                                'threshold': {
                                    'line':{'color':"red",'width':4}, 
                                    'thickness':0.75, 
                                    'value':threshold
                                }
                            }
                        ))
                        fig.update_layout(height=280, margin=dict(t=50,b=10,l=10,r=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with col_explainer:
                        st.markdown("#### 🧠 Why this prediction?")
                        top_features = result.get('top_features', [])
                        if top_features:
                            st.markdown(
                                '<div style="background:#f9f9f9;border-radius:8px;padding:12px;margin-bottom:8px;">'
                                '<b>Key indicators detected:</b> '
                                + ', '.join([f'<span style="color:#B22222;font-weight:600;">{t}</span>' for t in top_features]) +
                                '</div>', unsafe_allow_html=True
                            )
                        
                        explainability = result.get('explainability', {})
                        if explainability:
                            st.markdown("#### 📊 Analysis Details")
                            for key, value in explainability.items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Category-specific tips
                        if prediction == 'spam':
                            st.error("**🚨 Safety Tip:** Never share personal or financial information via SMS. Beware of urgent requests and suspicious links.")
                        elif prediction == 'promo':
                            st.warning("**💡 Promo Tip:** Verify the sender and check official websites before responding to promotional offers.")
                        else:
                            st.info("**👍 Good Practice:** Continue to be cautious with unsolicited messages.")
                        
                        st.markdown("[Report suspicious messages to authorities](https://reportfraud.ftc.gov/)")

                except Exception as e:
                    st.error(f"Prediction API error: {e}")

# --- Tab: Statistics ---
with tab_stats:
    
    # System Statistics Section
    st.markdown("---")
    st.markdown("### 📈 System Statistics")
    
    stats = fetch_api("stats")
    
    if stats:
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", stats.get('total_predictions', 0))
        col2.metric("Spam Detected", stats.get('spam_count', 0))
        col3.metric("Ham Detected", stats.get('ham_count', 0))
        col4.metric("Promo Detected", stats.get('promo_count', 0))
        
        # Percentages
        col5, col6, col7 = st.columns(3)
        col5.metric("Spam Rate", f"{stats.get('spam_percentage', 0):.1f}%")
        col6.metric("Ham Rate", f"{stats.get('ham_percentage', 0):.1f}%")
        col7.metric("Promo Rate", f"{stats.get('promo_percentage', 0):.1f}%")
        
        # Threat distribution
        threat_dist = stats.get('threat_distribution', {})
        if threat_dist:
            st.markdown("### Threat Level Distribution")
            threats_df = pd.DataFrame(list(threat_dist.items()), columns=['Threat Level', 'Count'])
            fig_threats = px.pie(threats_df, values='Count', names='Threat Level', 
                                title="Threat Level Distribution")
            st.plotly_chart(fig_threats, use_container_width=True)
        
        # Fraud type distribution
        fraud_dist = stats.get('fraud_distribution', {})
        if fraud_dist:
            st.markdown("### Fraud Type Distribution")
            fraud_df = pd.DataFrame(list(fraud_dist.items()), columns=['Fraud Type', 'Count'])
            fraud_df = fraud_df.sort_values('Count', ascending=False)
            fig_fraud = px.bar(fraud_df, x='Fraud Type', y='Count', 
                              title="Fraud Type Distribution",
                              color='Count')
            st.plotly_chart(fig_fraud, use_container_width=True)
    else:
        st.info("No system statistics available yet. Start classifying messages to see data!")

# --- Tab: History ---
with tab_history:
    st.markdown("## 📜 Prediction History")
    
    # Add select box for number of history records
    history_limit = st.selectbox(
        "Show how many recent predictions?",
        options=[10, 20, 50, 100, 200],
        index=2,
        help="Select the number of recent predictions to display"
    )
    
    # Fetch history data
    history_data = fetch_api("history", params={"limit": history_limit})
    
    if history_data and history_data.get('history'):
        df_pred = pd.DataFrame(history_data['history'])
        
        if not df_pred.empty:
            # Calculate metrics
            total_predictions = len(df_pred)
            spam_count = len(df_pred[df_pred['prediction'].astype(str).str.lower().str.strip() == 'spam'])
            ham_count = len(df_pred[df_pred['prediction'].astype(str).str.lower().str.strip() == 'ham'])
            promo_count = len(df_pred[df_pred['prediction'].astype(str).str.lower().str.strip() == 'promo'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", total_predictions)
            col2.metric("Spam", spam_count)
            col3.metric("Ham", ham_count)
            col4.metric("Promo", promo_count)
            
            # Recent predictions table
            st.markdown("### Recent Predictions")
            
            # Create display dataframe
            display_data = []
            for idx, row in df_pred.head(history_limit).iterrows():
                display_row = {
                    'Message': row.get('sms_text', 'N/A')[:80] + '...' if len(str(row.get('sms_text', ''))) > 80 else row.get('sms_text', 'N/A'),
                    'Type': row.get('prediction', 'N/A'),
                    'Fraud Type': row.get('fraud_type', 'N/A'),
                    'Confidence': f"{float(row.get('confidence', 0)):.1f}%" if row.get('confidence') else "N/A",
                    'Threat Level': row.get('threat_level', 'N/A')
                }
                if 'timestamp' in row:
                    try:
                        display_row['Time'] = pd.to_datetime(row['timestamp']).strftime('%H:%M')
                    except:
                        display_row['Time'] = str(row['timestamp'])[11:16]
                display_data.append(display_row)
            
            display_df = pd.DataFrame(display_data)
            
            # Style the table
            def style_prediction(val):
                if str(val).lower().strip() == 'spam':
                    return 'color: red; font-weight: bold;'
                elif str(val).lower().strip() == 'promo':
                    return 'color: orange; font-weight: bold;'
                else:  # ham
                    return 'color: green; font-weight: bold;'
            
            if 'Type' in display_df.columns:
                styled_df = display_df.style.applymap(style_prediction, subset=['Type'])
                st.dataframe(styled_df, use_container_width=True, height=400)
            else:
                st.dataframe(display_df, use_container_width=True, height=400)
            
        else:
            st.info("No prediction history data available yet.")
    else:
        st.info("No prediction history found. Classify some messages to see them here!")

# --- Tab: Analytics ---
with tab_visualizations:
    st.markdown("## 📈 Classification Analytics")
    
    # Fetch data for analytics
    stats = fetch_api("stats")
    history_data = fetch_api("history", params={"limit": 100})
    
    if stats and history_data and history_data.get('history'):
        df_pred = pd.DataFrame(history_data['history'])
        
        if not df_pred.empty:
            # Category distribution over time
            st.markdown("### Category Distribution Over Time")
            
            # Convert timestamp and prepare data
            df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
            df_pred['date'] = df_pred['timestamp'].dt.date
            df_pred['hour'] = df_pred['timestamp'].dt.hour
            
            # Daily counts
            daily_counts = df_pred.groupby(['date', 'prediction']).size().reset_index(name='count')
            fig_daily = px.line(daily_counts, x='date', y='count', color='prediction',
                               title="Daily Prediction Trends",
                               color_discrete_map={'spam': 'red', 'ham': 'green', 'promo': 'gold'})
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Hourly pattern
            st.markdown("### Hourly Activity Pattern")
            hourly_counts = df_pred.groupby(['hour', 'prediction']).size().reset_index(name='count')
            fig_hourly = px.line(hourly_counts, x='hour', y='count', color='prediction',
                                title="Hourly Prediction Pattern",
                                color_discrete_map={'spam': 'red', 'ham': 'green', 'promo': 'gold'})
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Removed Confidence Score Distribution section
            
        else:
            st.info("Not enough data for analytics yet. Keep using the classifier!")
    else:
        st.info("Analytics data will appear here as you classify more messages!")

# Add system status in sidebar
st.sidebar.markdown("### 🔧 System Status")
api_status = "✅ Connected" if fetch_api("") is not None else "❌ Disconnected"
st.sidebar.metric("API Status", api_status)

# Add multilingual support information
st.sidebar.markdown("### 🌐 Supported Languages")
st.sidebar.info("""
- ✅ English
- ✅ Hinglish (Hindi+English)
- ✅ Hindi  
- ✅ Kannada
""")
