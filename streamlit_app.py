import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re

# --- Configuration & Initialization ---
API_URL = "http://localhost:6006"

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp > header { display: none; }
    .main-title { font-size: 3rem; font-weight: 800; text-align: center; color: #B22222; margin-bottom: 0.25rem; padding-top: 1rem; }
    .sub-title { text-align: center; color: #666666; margin-bottom: 2rem; }
    .spam-result { padding: 25px; border-radius: 12px; text-align: center; font-size: 1.8rem; font-weight: bold; margin: 25px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .spam { background-color: #ffcccc; color: #8B0000; border: 3px solid #FF4500; }
    .ham { background-color: #ccffcc; color: #006400; border: 3px solid #3CB371; }
    div.stButton > button:first-child { border-radius: 8px; font-weight: bold; transition: all 0.2s; }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.15rem; font-weight: 600; padding: 10px 20px; border-radius: 8px 8px 0 0; }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">📱 SMS Spam Detective</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An Enhanced AI/ML Project Demo | Backend: Flask, Model: Naive Bayes</div>', unsafe_allow_html=True)

# --- Tabs ---
tab_home, tab_stats, tab_history, tab_visualizations = st.tabs(["🏠 Live Detector", "📊 Statistics & Model Insights", "📜 Prediction History", "📈 Advanced Visualizations"])

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

def safe_load_dataset(file_path):
    """Safely load and clean the dataset"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                st.success(f"Dataset loaded successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("Could not read the dataset with any encoding")
            return None
        
        # Clean the dataset
        st.write(f"Original dataset shape: {df.shape}")
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        st.write(f"After removing empty rows: {df.shape}")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def prepare_dataset_for_training(df):
    """Prepare dataset for model training and evaluation"""
    try:
        # Use only message and category columns
        required_cols = ['message', 'category']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"Required columns {required_cols} not found in dataset")
            st.write("Available columns:", df.columns.tolist())
            return None, None, None
        
        # Keep only the required columns
        df_clean = df[required_cols].copy()
        df_clean = df_clean.dropna()
        
        # Remove rows where message is empty
        df_clean = df_clean[df_clean['message'].astype(str).str.strip() != '']
        
        # Standardize label format
        df_clean['category'] = df_clean['category'].astype(str).str.lower().str.strip()
        
        # Map common label variations to spam/ham
        spam_keywords = ['spam', '1', 'true', 'yes', 'fraud', 'scam']
        ham_keywords = ['ham', '0', 'false', 'no', 'legit', 'normal', 'safe']
        
        def map_label(label):
            label_str = str(label).lower().strip()
            if any(keyword == label_str for keyword in spam_keywords):
                return 'spam'
            elif any(keyword == label_str for keyword in ham_keywords):
                return 'ham'
            else:
                return label_str
        
        df_clean['label'] = df_clean['category'].apply(map_label)
        
        # Filter only spam and ham
        df_final = df_clean[df_clean['label'].isin(['spam', 'ham'])].copy()
        
        st.write(f"Final dataset shape: {df_final.shape}")
        st.write("Label distribution:")
        st.write(df_final['label'].value_counts())
        
        if len(df_final) == 0:
            st.error("No valid spam/ham messages found after cleaning")
            return None, None, None
            
        return df_final, df_final['message'], df_final['label']
        
    except Exception as e:
        st.error(f"Error preparing dataset: {e}")
        return None, None, None

# --- Tab: Live Detector ---
with tab_home:
    st.markdown("### 💬 Real-time SMS Classification")

    if 'sms_input' not in st.session_state:
        st.session_state.sms_input = ""

    sms_text = st.text_area(
        "Type or paste your SMS message here:",
        height=150,
        placeholder="Example: Congratulations! You've won a £1000 prize. Text WIN to 87121.",
        key='sms_input_area'
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button("🚀 Classify Message", use_container_width=True, type="primary")

    if predict_button:
        if not sms_text.strip():
            st.error("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner("Analyzing SMS for Spam Score..."):
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
                    scam_type = result.get('fraud_type', 'Unknown')
                    threat_level = result.get('threat_level', 'Unknown')

                    scam_labels = {
                        'financial_scams': ('Financial Scam', '#B22222', 'Scams targeting your money or bank details.'),
                        'identity_theft_scams': ('Identity Theft', '#8B008B', 'Scams aiming to steal personal information.'),
                        'ecommerce_shopping_scams': ('E-commerce Scam', '#FF8C00', 'Fake shopping or delivery offers.'),
                        'tech_support_service_scams': ('Tech Support Scam', '#4682B4', 'Impersonation of tech support or service providers.'),
                        'relationship_emotional_scams': ('Relationship/Emotional Scam', '#C71585', 'Exploiting emotions or relationships for fraud.'),
                        'employment_opportunity_scams': ('Employment/Opportunity Scam', '#228B22', 'Fake job, internship, or survey offers.'),
                        'subscription_content_scams': ('Subscription/Content Scam', '#2E8B57', 'Traps with premium SMS or fake streaming links.'),
                        'business_crypto_scams': ('Business/Crypto Scam', '#DAA520', 'Ponzi, crypto, or business proposal scams.'),
                        'government_utility_scams': ('Government/Utility Scam', '#4169E1', 'Impersonation of government or utility agencies.'),
                        'hybrid_ai_scams': ('Hybrid/AI-driven Scam', '#000000', 'Modern scams using AI, deepfakes, or social engineering.')
                    }

                    scam_label, scam_color, scam_tip = scam_labels.get(scam_type, (scam_type, '#B22222', ''))

                    if prediction == 'spam':
                        st.markdown(
                            f'<div style="margin-bottom:10px;text-align:center;">'
                            f'<span style="display:inline-block;padding:4px 12px;border-radius:16px;background:{scam_color};color:#fff;font-weight:600;font-size:1.05rem;">{scam_label}</span>'
                            f'<span title="{scam_tip}" style="margin-left:8px;color:#666;font-size:0.95rem;">🛈</span>'
                            f'<div style="margin-top:6px;color:#444;font-size:0.98rem;">{scam_tip}</div>'
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
                    else:
                        st.markdown(
                            f'<div class="spam-result ham">✅ LEGITIMATE MESSAGE<br>Confidence: {risk_score:.2f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.success("✓ **Status:** This message appears safe.")

                    # Confidence Meter
                    st.markdown("### Confidence Breakdown")
                    col_gauge, col_explainer = st.columns([1,2])
                    with col_gauge:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_score,
                            domain={'x':[0,1], 'y':[0,1]},
                            title={'text': "Spam Risk Score"},
                            gauge={
                                'axis': {'range':[0,100], 'tickwidth':1, 'tickcolor':"darkblue"},
                                'bar': {'color': "#8B0000" if prediction=='spam' else "#006400"},
                                'steps':[{'range':[0,40],'color':"lightgreen"},{'range':[40,70],'color':"yellow"},{'range':[70,100],'color':"salmon"}],
                                'threshold': {'line':{'color':"red",'width':4}, 'thickness':0.75, 'value':75}
                            }
                        ))
                        fig.update_layout(height=280, margin=dict(t=50,b=10,l=10,r=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with col_explainer:
                        st.markdown("#### Why this prediction?")
                        top_features = result.get('top_features', [])
                        if top_features:
                            st.markdown(
                                '<div style="background:#f9f9f9;border-radius:8px;padding:12px;margin-bottom:8px;">'
                                '<b>Top tokens influencing spam detection:</b> '
                                + ', '.join([f'<span style="color:#B22222;font-weight:600;">{t}</span>' for t in top_features]) +
                                '</div>', unsafe_allow_html=True
                            )
                        else:
                            st.write("Model explainability not available.")
                        st.info("**Safety Tip:** Never share personal or financial information via SMS. Beware of urgent requests and suspicious links.")
                        st.markdown("[Report suspicious messages](https://reportfraud.ftc.gov/)")

                except Exception as e:
                    st.error(f"Prediction API error: {e}")

# --- Tab: Statistics & Model Insights ---
with tab_stats:
    st.markdown("## 📊 Model & Usage Statistics")
    stats = fetch_api("stats")

    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total API Calls", stats.get('total_predictions', 0))
        col2.metric("Spam Identified", stats.get('spam_count', 0))
        col3.metric("Ham Identified", stats.get('ham_count', 0))
        col4.metric("Spam Rate", f"{stats.get('spam_percentage',0):.2f}%")

        st.markdown("---")
        st.markdown("### Model Performance Metrics (On Training Data)")

        try:
            # Load and prepare dataset
            df_raw = safe_load_dataset('enhanced_scam_dataset.csv')
            if df_raw is not None:
                df_clean, X, y = prepare_dataset_for_training(df_raw)
                
                if df_clean is not None and model and vectorizer:
                    # Convert labels to numeric
                    y_numeric = y.map({'spam': 1, 'ham': 0})
                    
                    # Check for NaN values
                    if y_numeric.isna().any():
                        st.warning("NaN values found in labels after conversion. Removing them.")
                        valid_indices = y_numeric.notna()
                        X = X[valid_indices]
                        y_numeric = y_numeric[valid_indices]
                    
                    # Transform messages using the loaded vectorizer
                    X_transformed = vectorizer.transform(X)
                    
                    # Make predictions
                    y_pred = model.predict(X_transformed)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_numeric, y_pred)
                    precision = precision_score(y_numeric, y_pred, zero_division=0)
                    recall = recall_score(y_numeric, y_pred, zero_division=0)

                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Accuracy", f"{accuracy:.2%}")
                    metrics_col2.metric("Precision", f"{precision:.2%}")
                    metrics_col3.metric("Recall", f"{recall:.2%}")

                    # Confusion matrix
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_numeric, y_pred)
                    fig_cm = px.imshow(cm, labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                       x=['Ham (0)','Spam (1)'], y=['Ham (0)','Spam (1)'],
                                       color_continuous_scale='Reds')
                    for i in range(len(cm)):
                        for j in range(len(cm[i])):
                            fig_cm.add_annotation(x=j, y=i, text=str(cm[i][j]), showarrow=False,
                                                  font=dict(color="black", size=16))
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                else:
                    st.warning("Could not prepare data for model evaluation")
            else:
                st.warning("Dataset not available for model evaluation")

        except FileNotFoundError:
            st.warning("⚠️ enhanced_scam_dataset.csv not found. Cannot calculate model metrics.")
        except Exception as e:
            st.error(f"Error calculating model metrics: {e}")

# --- Tab: Prediction History ---
with tab_history:
    st.markdown("## 📜 Prediction History & Performance")
    
    # Fetch history data
    history_data = fetch_api("history", params={"limit": 1000})
    
    if history_data and history_data.get('history'):
        df_pred = pd.DataFrame(history_data['history'])
        
        if not df_pred.empty:
            # Display metrics
            st.markdown("### User Prediction Metrics")
            
            # Calculate metrics
            total_predictions = len(df_pred)
            
            # Count spam predictions correctly
            spam_count = 0
            ham_count = 0
            
            if 'prediction' in df_pred.columns:
                spam_count = len(df_pred[df_pred['prediction'].astype(str).str.lower().str.strip() == 'spam'])
                ham_count = len(df_pred[df_pred['prediction'].astype(str).str.lower().str.strip() == 'ham'])
            
            spam_rate = (spam_count / total_predictions) * 100 if total_predictions > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Predictions", total_predictions)
            col2.metric("Spam Detected", spam_count)
            col3.metric("Ham Detected", ham_count)
            col4.metric("Spam Rate", f"{spam_rate:.2f}%")
            
            # Recent predictions table
            st.markdown("### Recent Predictions")
            
            # Create display dataframe with available columns
            display_data = []
            for idx, row in df_pred.head(20).iterrows():
                display_row = {
                    'sms_text': row.get('sms_text', 'N/A')[:100] + '...' if len(str(row.get('sms_text', ''))) > 100 else row.get('sms_text', 'N/A'),
                    'prediction': row.get('prediction', 'N/A'),
                    'confidence': f"{float(row.get('confidence', 0)):.2f}%" if row.get('confidence') else "N/A"
                }
                if 'timestamp' in row:
                    try:
                        display_row['timestamp'] = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        display_row['timestamp'] = str(row['timestamp'])
                display_data.append(display_row)
            
            display_df = pd.DataFrame(display_data)
            
            # Style the table
            def style_prediction(val):
                if str(val).lower().strip() == 'spam':
                    return 'color: red; font-weight: bold;'
                else:
                    return 'color: green; font-weight: bold;'
            
            if 'prediction' in display_df.columns:
                st.dataframe(
                    display_df.style.applymap(style_prediction, subset=['prediction']),
                    use_container_width=True,
                    height=400
                )
            else:
                st.dataframe(display_df, use_container_width=True, height=400)
            
        else:
            st.info("No prediction history data available yet.")
    else:
        st.warning("No prediction history found. Predictions will appear here as users interact with the app.")

# --- Tab: Advanced Visualizations ---
with tab_visualizations:
    st.markdown("## 📈 Advanced Model Visualizations")
    
    try:
        # Load dataset for visualizations
        df_raw = safe_load_dataset("enhanced_scam_dataset.csv")
        
        if df_raw is not None:
            df_clean, X, y = prepare_dataset_for_training(df_raw)
            
            if df_clean is not None:
                st.markdown("### Dataset Overview")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Messages", len(df_clean))
                spam_count = len(df_clean[df_clean['label'] == 'spam'])
                ham_count = len(df_clean[df_clean['label'] == 'ham'])
                col2.metric("Spam Count", spam_count)
                col3.metric("Ham Count", ham_count)
                
                # Visualization options
                viz_option = st.selectbox(
                    "Choose Visualization:",
                    [
                        "Confusion Matrix Heatmap",
                        "Message Length Distribution",
                        "Spam vs Ham Distribution",
                        "Word Frequency Analysis",
                        "Model Performance Comparison"
                    ]
                )
                
                if viz_option == "Confusion Matrix Heatmap":
                    st.markdown("### Confusion Matrix Heatmap")
                    
                    # Prepare data for model training
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Vectorize text
                    vectorizer_viz = TfidfVectorizer(max_features=1000, stop_words="english")
                    X_train_tfidf = vectorizer_viz.fit_transform(X_train)
                    X_test_tfidf = vectorizer_viz.transform(X_test)
                    
                    # Train model
                    model_viz = LogisticRegression(max_iter=1000)
                    model_viz.fit(X_train_tfidf, y_train.map({'spam': 1, 'ham': 0}))
                    
                    # Predictions
                    y_pred = model_viz.predict(X_test_tfidf)
                    y_test_numeric = y_test.map({'spam': 1, 'ham': 0})
                    
                    # Create confusion matrix
                    cm = confusion_matrix(y_test_numeric, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                               xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
                    plt.title("Confusion Matrix")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(plt)
                    
                    # Show metrics
                    accuracy = accuracy_score(y_test_numeric, y_pred)
                    precision = precision_score(y_test_numeric, y_pred, zero_division=0)
                    recall = recall_score(y_test_numeric, y_pred, zero_division=0)
                    
                    st.markdown(f"""
                    **Model Performance Metrics:**
                    - **Accuracy:** {accuracy:.3f}
                    - **Precision:** {precision:.3f}  
                    - **Recall:** {recall:.3f}
                    """)
                    
                elif viz_option == "Message Length Distribution":
                    st.markdown("### Message Length Distribution")
                    
                    df_clean['message_length'] = df_clean['message'].str.len()
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Histogram
                    sns.histplot(data=df_clean, x='message_length', hue='label', bins=50, ax=ax1)
                    ax1.set_title('Message Length Distribution by Category')
                    ax1.set_xlabel('Message Length')
                    ax1.set_ylabel('Frequency')
                    
                    # Box plot
                    sns.boxplot(data=df_clean, x='label', y='message_length', ax=ax2)
                    ax2.set_title('Message Length by Category')
                    ax2.set_xlabel('Category')
                    ax2.set_ylabel('Message Length')
                    
                    plt.tight_layout()
                    st.pyplot(plt)
                    
                elif viz_option == "Spam vs Ham Distribution":
                    st.markdown("### Spam vs Ham Distribution")
                    
                    category_counts = df_clean['label'].value_counts()
                    
                    fig = px.pie(values=category_counts.values, 
                                names=category_counts.index,
                                title="Spam vs Ham Distribution",
                                color=category_counts.index,
                                color_discrete_map={'spam': 'red', 'ham': 'green'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif viz_option == "Word Frequency Analysis":
                    st.markdown("### Word Frequency Analysis")
                    
                    from collections import Counter
                    
                    # Extract words from spam messages
                    spam_words = ' '.join(df_clean[df_clean['label'] == 'spam']['message']).lower()
                    spam_words = re.findall(r'\b[a-z]+\b', spam_words)
                    spam_word_freq = Counter(spam_words)
                    
                    # Extract words from ham messages
                    ham_words = ' '.join(df_clean[df_clean['label'] == 'ham']['message']).lower()
                    ham_words = re.findall(r'\b[a-z]+\b', ham_words)
                    ham_word_freq = Counter(ham_words)
                    
                    # Get top words (exclude common short words)
                    common_words = {'the', 'and', 'to', 'a', 'i', 'you', 'is', 'in', 'it', 'for', 'of', 'on', 'that', 'with', 'my', 'your', 'me', 'are', 'so', 'but', 'be', 'at', 'if', 'or', 'as', 'will', 'have', 'has', 'had', 'this', 'not'}
                    spam_word_freq_filtered = {word: count for word, count in spam_word_freq.items() if word not in common_words and len(word) > 2}
                    ham_word_freq_filtered = {word: count for word, count in ham_word_freq.items() if word not in common_words and len(word) > 2}
                    
                    top_spam_words = dict(Counter(spam_word_freq_filtered).most_common(15))
                    top_ham_words = dict(Counter(ham_word_freq_filtered).most_common(15))
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Spam words
                    if top_spam_words:
                        ax1.barh(list(top_spam_words.keys()), list(top_spam_words.values()), color='red')
                        ax1.set_title('Top 15 Words in Spam Messages')
                        ax1.set_xlabel('Frequency')
                    else:
                        ax1.text(0.5, 0.5, 'No spam words found', ha='center', va='center')
                    
                    # Ham words
                    if top_ham_words:
                        ax2.barh(list(top_ham_words.keys()), list(top_ham_words.values()), color='green')
                        ax2.set_title('Top 15 Words in Ham Messages')
                        ax2.set_xlabel('Frequency')
                    else:
                        ax2.text(0.5, 0.5, 'No ham words found', ha='center', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(plt)
                    
                elif viz_option == "Model Performance Comparison":
                    st.markdown("### Model Performance Comparison")
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    vectorizer_comp = TfidfVectorizer(max_features=1000, stop_words="english")
                    X_train_tfidf = vectorizer_comp.fit_transform(X_train)
                    X_test_tfidf = vectorizer_comp.transform(X_test)
                    
                    y_train_numeric = y_train.map({'spam': 1, 'ham': 0})
                    y_test_numeric = y_test.map({'spam': 1, 'ham': 0})
                    
                    performance_data = []
                    
                    # Logistic Regression
                    lr_model = LogisticRegression(max_iter=1000)
                    lr_model.fit(X_train_tfidf, y_train_numeric)
                    y_pred_lr = lr_model.predict(X_test_tfidf)
                    
                    performance_data.append({
                        'Model': 'Logistic Regression',
                        'Accuracy': accuracy_score(y_test_numeric, y_pred_lr),
                        'Precision': precision_score(y_test_numeric, y_pred_lr, zero_division=0),
                        'Recall': recall_score(y_test_numeric, y_pred_lr, zero_division=0)
                    })
                    
                    # Naive Bayes
                    nb_model = MultinomialNB()
                    nb_model.fit(X_train_tfidf, y_train_numeric)
                    y_pred_nb = nb_model.predict(X_test_tfidf)
                    
                    performance_data.append({
                        'Model': 'Naive Bayes',
                        'Accuracy': accuracy_score(y_test_numeric, y_pred_nb),
                        'Precision': precision_score(y_test_numeric, y_pred_nb, zero_division=0),
                        'Recall': recall_score(y_test_numeric, y_pred_nb, zero_division=0)
                    })
                    
                    # Create comparison chart
                    perf_df = pd.DataFrame(performance_data)
                    perf_melted = perf_df.melt(id_vars=['Model'], 
                                              value_vars=['Accuracy', 'Precision', 'Recall'],
                                              var_name='Metric', value_name='Score')
                    
                    fig = px.bar(perf_melted, x='Model', y='Score', color='Metric',
                                barmode='group', title='Model Performance Comparison',
                                color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not prepare dataset for visualizations")
        else:
            st.error("Dataset not available for visualizations")
            
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")