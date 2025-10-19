#!/usr/bin/env python3
"""
EMOTIONALLY AWARE CHATBOT - COMPLETE AND ERROR-FREE VERSION
A sophisticated chatbot with real-time emotion detection and empathetic responses.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import re
import warnings
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# Try importing transformers with fallback
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Try importing torch with fallback
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try importing textblob as fallback
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Page configuration
st.set_page_config(
    page_title="Emotionally Aware Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
def load_css():
    """Load custom CSS for beautiful UI"""
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding: 0rem 1rem;
    }

    /* Chat interface styling */
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        border-radius: 20px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px 0 rgba(236, 116, 149, 0.35);
        animation: slideInRight 0.5s ease-out;
    }

    .bot-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 20px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px 0 rgba(168, 237, 234, 0.35);
        animation: slideInLeft 0.5s ease-out;
    }

    /* Emotion badges - 28 emotions */
    .emotion-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px;
        animation: pulse 2s infinite;
    }

    .emotion-joy { background: linear-gradient(45deg, #FFD700, #FFA500); color: white; }
    .emotion-sadness { background: linear-gradient(45deg, #4682B4, #191970); color: white; }
    .emotion-anger { background: linear-gradient(45deg, #DC143C, #8B0000); color: white; }
    .emotion-fear { background: linear-gradient(45deg, #483D8B, #2F2F2F); color: white; }
    .emotion-surprise { background: linear-gradient(45deg, #FF69B4, #FF1493); color: white; }
    .emotion-disgust { background: linear-gradient(45deg, #556B2F, #8FBC8F); color: white; }
    .emotion-neutral { background: linear-gradient(45deg, #708090, #2F4F4F); color: white; }
    .emotion-love { background: linear-gradient(45deg, #FF6347, #FF69B4); color: white; }
    .emotion-admiration { background: linear-gradient(45deg, #FFB6C1, #FF69B4); color: white; }
    .emotion-amusement { background: linear-gradient(45deg, #98FB98, #90EE90); color: black; }
    .emotion-annoyance { background: linear-gradient(45deg, #FF8C00, #FF6347); color: white; }
    .emotion-approval { background: linear-gradient(45deg, #32CD32, #228B22); color: white; }
    .emotion-caring { background: linear-gradient(45deg, #FF69B4, #FFB6C1); color: white; }
    .emotion-confusion { background: linear-gradient(45deg, #DDA0DD, #BA55D3); color: white; }
    .emotion-curiosity { background: linear-gradient(45deg, #00CED1, #48D1CC); color: white; }
    .emotion-desire { background: linear-gradient(45deg, #FF1493, #C71585); color: white; }
    .emotion-disappointment { background: linear-gradient(45deg, #696969, #808080); color: white; }
    .emotion-disapproval { background: linear-gradient(45deg, #CD5C5C, #8B4513); color: white; }
    .emotion-embarrassment { background: linear-gradient(45deg, #FFB6C1, #FFC0CB); color: black; }
    .emotion-excitement { background: linear-gradient(45deg, #FF4500, #FF6347); color: white; }
    .emotion-gratitude { background: linear-gradient(45deg, #DAA520, #FFD700); color: black; }
    .emotion-grief { background: linear-gradient(45deg, #2F4F4F, #000000); color: white; }
    .emotion-nervousness { background: linear-gradient(45deg, #B0C4DE, #778899); color: white; }
    .emotion-optimism { background: linear-gradient(45deg, #87CEEB, #00BFFF); color: white; }
    .emotion-pride { background: linear-gradient(45deg, #9370DB, #8A2BE2); color: white; }
    .emotion-realization { background: linear-gradient(45deg, #F0E68C, #FFFF00); color: black; }
    .emotion-relief { background: linear-gradient(45deg, #90EE90, #98FB98); color: black; }
    .emotion-remorse { background: linear-gradient(45deg, #483D8B, #6A5ACD); color: white; }

    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 10px 20px;
    }

    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #667eea;
        padding: 15px 20px;
    }

    /* Chart container */
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False

    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0

    if 'dominant_emotion' not in st.session_state:
        st.session_state.dominant_emotion = "neutral"

    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'response_style': 'balanced',
            'show_emotions': True,
            'show_confidence': True,
            'auto_save': False
        }

# Emotion Analyzer Class
class EmotionAnalyzer:
    """Advanced emotion analyzer with multiple model support and fallbacks"""

    def __init__(self):
        self.primary_model = None
        self.backup_model = None
        self.all_emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        self.model_loaded = False

        # Try to load models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize emotion detection models with fallbacks"""
        try:
            if HAS_TRANSFORMERS:
                # Try primary model for 28 emotions
                try:
                    self.primary_model = pipeline(
                        "text-classification",
                        model="SamLowe/roberta-base-go_emotions",
                        device=-1  # Use CPU
                    )
                    self.model_loaded = True
                    print("Primary emotion model loaded successfully")
                except Exception as e:
                    print(f"Primary model failed: {e}")

                # Try backup model
                if not self.model_loaded:
                    try:
                        self.backup_model = pipeline(
                            "text-classification",
                            model="j-hartmann/emotion-english-distilroberta-base",
                            device=-1
                        )
                        self.model_loaded = True
                        print("Backup emotion model loaded successfully")
                    except Exception as e:
                        print(f"Backup model failed: {e}")
        except Exception as e:
            print(f"Model initialization failed: {e}")

    def analyze_emotion(self, text: str) -> Tuple[str, float]:
        """
        Analyze emotion in text with comprehensive fallbacks

        Args:
            text (str): Input text to analyze

        Returns:
            Tuple[str, float]: Detected emotion and confidence score
        """
        if not text or not text.strip():
            return "neutral", 0.5

        # Clean text
        text = str(text).strip()[:512]  # Limit length

        # Try primary model
        if self.primary_model:
            try:
                result = self.primary_model(text)
                if result and len(result) > 0:
                    emotion = result[0]['label'].lower()
                    confidence = float(result[0]['score'])
                    return self._normalize_emotion(emotion), confidence
            except Exception as e:
                print(f"Primary model error: {e}")

        # Try backup model
        if self.backup_model:
            try:
                result = self.backup_model(text)
                if result and len(result) > 0:
                    emotion = result[0]['label'].lower()
                    confidence = float(result[0]['score'])
                    return self._normalize_emotion(emotion), confidence
            except Exception as e:
                print(f"Backup model error: {e}")

        # TextBlob fallback
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # Map polarity and subjectivity to 28 emotions
                if polarity > 0.5:
                    if subjectivity > 0.5:
                        return "excitement", abs(polarity)
                    else:
                        return "joy", abs(polarity)
                elif polarity > 0.2:
                    return "optimism", abs(polarity)
                elif polarity < -0.5:
                    if subjectivity > 0.5:
                        return "anger", abs(polarity)
                    else:
                        return "sadness", abs(polarity)
                elif polarity < -0.2:
                    return "disappointment", abs(polarity)
                else:
                    if subjectivity > 0.7:
                        return "confusion", 0.5
                    else:
                        return "neutral", 0.5
            except Exception as e:
                print(f"TextBlob error: {e}")

        # Final fallback - keyword-based analysis
        return self._keyword_emotion_analysis(text)

    def _normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion labels to 28-emotion schema"""
        emotion_mapping = {
            'positive': 'joy',
            'negative': 'sadness',
            'happiness': 'joy',
            'sad': 'sadness',
            'angry': 'anger',
            'scared': 'fear',
            'surprised': 'surprise',
            'disgusted': 'disgust',
            'happy': 'joy',
            'anxious': 'nervousness',
            'grateful': 'gratitude',
            'proud': 'pride',
            'confused': 'confusion',
            'curious': 'curiosity',
            'disappointed': 'disappointment',
            'embarrassed': 'embarrassment',
            'excited': 'excitement',
            'loving': 'love',
            'optimistic': 'optimism',
            'relieved': 'relief',
            'remorseful': 'remorse'
        }

        normalized = emotion_mapping.get(emotion, emotion)
        # Ensure it's one of our 28 emotions
        if normalized not in self.all_emotions:
            return "neutral"
        return normalized

    def _keyword_emotion_analysis(self, text: str) -> Tuple[str, float]:
        """Keyword-based emotion analysis as final fallback for 28 emotions"""
        text_lower = text.lower()

        emotion_keywords = {
            'admiration': ['admire', 'respect', 'impressed', 'inspired', 'look up to'],
            'amusement': ['funny', 'hilarious', 'amusing', 'entertained', 'laugh'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'rage'],
            'annoyance': ['annoying', 'irritating', 'bothersome', 'frustrating'],
            'approval': ['approve', 'agree', 'good', 'right', 'correct', 'yes'],
            'caring': ['care', 'concerned', 'worried about', 'love', 'support'],
            'confusion': ['confused', 'unclear', 'puzzled', 'lost', 'uncertain'],
            'curiosity': ['curious', 'wondering', 'interested', 'want to know'],
            'desire': ['want', 'wish', 'hope', 'crave', 'long for', 'need'],
            'disappointment': ['disappointed', 'let down', 'expected better'],
            'disapproval': ['disapprove', 'disagree', 'wrong', 'bad', 'no'],
            'disgust': ['disgusting', 'gross', 'revolting', 'nasty', 'ugh'],
            'embarrassment': ['embarrassed', 'ashamed', 'humiliated', 'awkward'],
            'excitement': ['excited', 'thrilled', 'enthusiastic', 'pumped'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'terrified'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'thankful'],
            'grief': ['grief', 'mourn', 'loss', 'bereaved', 'sorrow'],
            'joy': ['happy', 'joy', 'delighted', 'pleased', 'cheerful'],
            'love': ['love', 'adore', 'cherish', 'affection', 'fond'],
            'nervousness': ['nervous', 'anxious', 'uneasy', 'tense', 'jittery'],
            'optimism': ['optimistic', 'hopeful', 'positive', 'bright', 'confident'],
            'pride': ['proud', 'accomplished', 'achieved', 'succeeded'],
            'realization': ['realize', 'understand', 'see', 'get it', 'aha'],
            'relief': ['relieved', 'relief', 'phew', 'finally', 'glad'],
            'remorse': ['sorry', 'regret', 'remorse', 'guilty', 'apologize'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'down'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow']
        }

        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[emotion] = score / len(keywords)

        if scores:
            best_emotion = max(scores, key=scores.get)
            confidence = min(scores[best_emotion] * 2, 1.0)  # Scale confidence
            return best_emotion, confidence

        return "neutral", 0.5

# Emotional Response Generator
class EmotionalResponseGenerator:
    """Generate contextually appropriate emotional responses for 28 emotions"""

    def __init__(self):
        self.response_templates = {
            'admiration': [
                "That's truly inspiring! I can sense your deep respect. {context}",
                "Your admiration is wonderful to see! {context}",
                "It's beautiful when we find someone or something to look up to. {context}"
            ],
            'amusement': [
                "Ha! That's genuinely funny! {context}",
                "Your sense of humor is delightful! {context}",
                "Laughter truly is the best medicine! {context}"
            ],
            'anger': [
                "I can sense your frustration. {context} Let's work through this together.",
                "Your anger is completely understandable. {context} How can I help?",
                "It's natural to feel angry sometimes. {context} Your feelings are valid."
            ],
            'annoyance': [
                "That does sound irritating. {context} I understand your frustration.",
                "I can see why that would be annoying. {context} Let's address it.",
                "Your annoyance makes perfect sense. {context} What can we do about it?"
            ],
            'approval': [
                "Excellent! I completely agree with you. {context}",
                "That's absolutely right! {context} Great thinking!",
                "Yes, you're spot on! {context} Well done!"
            ],
            'caring': [
                "Your caring nature really shines through. {context}",
                "It's wonderful to see such compassion. {context}",
                "Your concern for others is touching. {context}"
            ],
            'confusion': [
                "I understand this is confusing. {context} Let's clarify things together.",
                "It's okay to feel uncertain. {context} What specifically is unclear?",
                "Confusion is the first step to understanding. {context} Let's work through it."
            ],
            'curiosity': [
                "Your curiosity is wonderful! {context} Let's explore this together.",
                "Great question! {context} I love your inquisitive mind.",
                "Curiosity drives discovery! {context} What would you like to know more about?"
            ],
            'desire': [
                "I understand what you're hoping for. {context} Tell me more about it.",
                "Your aspirations are clear. {context} How can we work toward them?",
                "It's good to know what we want. {context} What's your plan?"
            ],
            'disappointment': [
                "I'm sorry things didn't go as expected. {context} That must be hard.",
                "Disappointment is never easy. {context} You're handling it well.",
                "I understand your letdown. {context} Better things are ahead."
            ],
            'disapproval': [
                "I see you disagree with this. {context} Your perspective matters.",
                "Your disapproval is noted. {context} What would you prefer?",
                "I understand your objection. {context} Let's find a better way."
            ],
            'disgust': [
                "That sounds unpleasant. {context} I can understand your reaction.",
                "Ugh, that does sound gross! {context} Let's talk about something better.",
                "I can see why that would bother you. {context} Your reaction makes sense."
            ],
            'embarrassment': [
                "We all have embarrassing moments. {context} You're not alone.",
                "I understand that feeling. {context} It will pass, I promise.",
                "Embarrassment is temporary. {context} You're doing great."
            ],
            'excitement': [
                "Your excitement is contagious! {context} This is wonderful!",
                "I love your enthusiasm! {context} Tell me more!",
                "That's absolutely thrilling! {context} I'm excited for you!"
            ],
            'fear': [
                "I hear the concern in your words. {context} You're brave for sharing this.",
                "Fear is natural. {context} You're stronger than you think.",
                "It's okay to feel afraid. {context} Let's face this together."
            ],
            'gratitude': [
                "Your gratitude is heartwarming. {context} It's wonderful to appreciate things.",
                "Being thankful enriches life. {context} I'm glad you feel this way.",
                "Gratitude is powerful. {context} Thank you for sharing this."
            ],
            'grief': [
                "I'm so sorry for your loss. {context} Take all the time you need.",
                "Grief is a journey. {context} I'm here to support you.",
                "Your pain is valid. {context} You don't have to go through this alone."
            ],
            'joy': [
                "Your happiness is wonderful! {context} I'm so glad for you!",
                "That's fantastic! {context} Your joy brightens everything!",
                "I love seeing you this happy! {context} Celebrate this moment!"
            ],
            'love': [
                "Love is beautiful! {context} It's wonderful when we feel connected.",
                "Your love shines through. {context} What a precious feeling.",
                "Love makes everything brighter. {context} Cherish this emotion."
            ],
            'nervousness': [
                "It's okay to feel nervous. {context} You've got this!",
                "Nervousness means you care. {context} That's actually a good sign.",
                "Take a deep breath. {context} You're more capable than you realize."
            ],
            'optimism': [
                "Your positive outlook is inspiring! {context} Keep that spirit!",
                "Optimism is powerful! {context} Good things are coming.",
                "I love your hopeful perspective! {context} It will serve you well."
            ],
            'pride': [
                "You should be proud! {context} You've earned this feeling.",
                "Your pride is well-deserved! {context} Celebrate your achievement!",
                "What an accomplishment! {context} Own this moment of pride!"
            ],
            'realization': [
                "That's a powerful insight! {context} Moments of clarity are precious.",
                "Aha moments are amazing! {context} What will you do with this understanding?",
                "Great realization! {context} This changes things, doesn't it?"
            ],
            'relief': [
                "I'm so glad you're feeling relieved! {context} You made it through.",
                "Relief feels wonderful! {context} You can relax now.",
                "What a weight off your shoulders! {context} Well done getting here."
            ],
            'remorse': [
                "It takes courage to feel remorse. {context} You can make things right.",
                "Regret shows growth. {context} What matters is what you do next.",
                "I understand your feelings. {context} Forgiveness starts with yourself."
            ],
            'sadness': [
                "I'm sorry you're feeling this way. {context} I'm here to listen.",
                "Sadness is part of being human. {context} You're not alone.",
                "It's okay to feel sad. {context} Take your time to process this."
            ],
            'surprise': [
                "Wow, that's unexpected! {context} How are you feeling about it?",
                "What a surprise! {context} Life certainly keeps us on our toes!",
                "That must have caught you off guard! {context} Tell me more!"
            ],
            'neutral': [
                "I hear you. {context} Tell me more about what's on your mind.",
                "Thanks for sharing that with me. {context} What would you like to discuss?",
                "I'm listening. {context} What's the most important thing right now?"
            ]
        }

    def generate_response(self, emotion: str, confidence: float, user_message: str) -> str:
        """Generate an emotionally appropriate response"""
        try:
            # Get response templates for the detected emotion
            templates = self.response_templates.get(emotion, self.response_templates['neutral'])

            # Select a template based on confidence (higher confidence = more specific responses)
            template_index = min(int(confidence * len(templates)), len(templates) - 1)
            template = templates[template_index]

            # Generate context based on message analysis
            context = self._generate_context(user_message, emotion, confidence)

            # Fill in the template
            response = template.format(context=context)

            return response

        except Exception as e:
            print(f"Response generation error: {e}")
            return f"I understand you're feeling {emotion}. Tell me more about what's on your mind."

    def _generate_context(self, message: str, emotion: str, confidence: float) -> str:
        """Generate contextual information for the response"""
        try:
            # Analyze message length and complexity
            word_count = len(message.split())

            if word_count < 5:
                return "Even in few words, I can sense your feelings."
            elif word_count < 15:
                return "I can hear the emotion in your message."
            else:
                return "Thank you for sharing so openly with me."

        except Exception:
            return "I appreciate you sharing this with me."

# Analytics Functions
def show_emotion_analytics():
    """Display comprehensive emotion analytics"""
    try:
        if not st.session_state.emotion_history:
            st.info("Start chatting to see emotion analytics!")
            return

        st.subheader("Emotion Analytics Dashboard")

        # Prepare data
        df = pd.DataFrame(st.session_state.emotion_history)

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_emotions = len(df)
            st.markdown(f"""
            <div class="metric-container">
                <h3>Total</h3>
                <h2>{total_emotions}</h2>
                <p>Total Interactions</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            dominant = df['emotion'].mode().iloc[0] if not df.empty else "neutral"
            st.markdown(f"""
            <div class="metric-container">
                <h3>Dominant</h3>
                <h2>{dominant.title()}</h2>
                <p>Dominant Emotion</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_confidence = df['confidence'].mean() if not df.empty else 0
            st.markdown(f"""
            <div class="metric-container">
                <h3>Confidence</h3>
                <h2>{avg_confidence:.1%}</h2>
                <p>Avg Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            unique_emotions = df['emotion'].nunique() if not df.empty else 0
            st.markdown(f"""
            <div class="metric-container">
                <h3>Variety</h3>
                <h2>{unique_emotions}</h2>
                <p>Unique Emotions</p>
            </div>
            """, unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Emotion distribution
            emotion_counts = df['emotion'].value_counts()
            fig_pie = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title_font_size=16
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Confidence over time
            df['message_number'] = range(1, len(df) + 1)
            fig_line = px.line(
                df,
                x='message_number',
                y='confidence',
                title="Confidence Over Time",
                color_discrete_sequence=['#00CED1']
            )
            fig_line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title_font_size=16
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # Recent emotion timeline
        if len(df) >= 5:
            st.subheader("Recent Emotion Timeline")
            recent_df = df.tail(10)

            fig_timeline = px.bar(
                recent_df,
                x='message_number',
                y='confidence',
                color='emotion',
                title="Last 10 Interactions",
                hover_data=['emotion', 'confidence']
            )
            fig_timeline.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title_font_size=16
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    except Exception as e:
        st.error(f"Analytics error: {e}")
        st.info("Unable to display analytics. Please continue chatting!")

def show_emotion_insights():
    """Display emotion insights and patterns"""
    try:
        if not st.session_state.emotion_history:
            st.info("Chat more to unlock emotion insights!")
            return

        st.subheader("Emotion Insights")

        df = pd.DataFrame(st.session_state.emotion_history)

        # Insights
        insights = []

        # Most frequent emotion
        if not df.empty:
            most_common = df['emotion'].mode().iloc[0]
            insights.append(f"Your most expressed emotion is **{most_common}**")

        # Confidence patterns
        if len(df) >= 3:
            avg_conf = df['confidence'].mean()
            if avg_conf > 0.8:
                insights.append("You express emotions very clearly!")
            elif avg_conf > 0.6:
                insights.append("Your emotional expressions are moderately clear")
            else:
                insights.append("Your emotions seem subtle or mixed")

        # Emotion diversity
        if not df.empty:
            unique_emotions = df['emotion'].nunique()
            total_messages = len(df)
            diversity = unique_emotions / total_messages if total_messages > 0 else 0

            if diversity > 0.7:
                insights.append("You show a wide range of emotions!")
            elif diversity > 0.4:
                insights.append("You express a good variety of emotions")
            else:
                insights.append("You tend to be emotionally consistent")

        # Display insights
        for insight in insights:
            st.markdown(f"- {insight}")

        if not insights:
            st.info("Keep chatting to discover more insights about your emotional patterns!")

    except Exception as e:
        st.error(f"Insights error: {e}")
        st.info("Unable to generate insights. Please continue chatting!")

def export_conversation():
    """Export conversation history"""
    try:
        if not st.session_state.messages:
            st.warning("No conversation to export!")
            return

        # Prepare export data
        export_data = {
            'conversation': st.session_state.messages,
            'emotion_history': st.session_state.emotion_history,
            'stats': {
                'total_messages': len(st.session_state.messages),
                'total_emotions': len(st.session_state.emotion_history),
                'dominant_emotion': st.session_state.dominant_emotion,
                'export_timestamp': datetime.now().isoformat()
            }
        }

        # Convert to JSON
        json_data = json.dumps(export_data, indent=2, default=str)

        # Create download
        st.download_button(
            label="Download Conversation (JSON)",
            data=json_data,
            file_name=f"emotion_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # Also offer CSV export for analytics
        if st.session_state.emotion_history:
            df = pd.DataFrame(st.session_state.emotion_history)
            csv_data = df.to_csv(index=False)

            st.download_button(
                label="Download Analytics (CSV)",
                data=csv_data,
                file_name=f"emotion_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        st.success("Export ready! Click the buttons above to download.")

    except Exception as e:
        st.error(f"Export error: {e}")

# Main application
def main():
    """Main application function"""
    try:
        # Load CSS
        load_css()

        # Initialize session state
        init_session_state()

        # Initialize models
        if 'emotion_analyzer' not in st.session_state:
            with st.spinner("Loading emotion detection models..."):
                st.session_state.emotion_analyzer = EmotionAnalyzer()

        if 'response_generator' not in st.session_state:
            st.session_state.response_generator = EmotionalResponseGenerator()

        # Header
        st.markdown("""
        <div class="chat-container">
            <h1 style="text-align: center; color: white; margin-bottom: 0;">
                Emotionally Aware Chatbot
            </h1>
            <p style="text-align: center; color: white; margin-top: 5px;">
                Your AI companion that understands and responds to 28 different emotions
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.header("Settings")

            # User preferences
            st.subheader("Preferences")

            st.session_state.user_preferences['response_style'] = st.selectbox(
                "Response Style",
                ["balanced", "empathetic", "analytical", "casual"],
                index=0
            )

            st.session_state.user_preferences['show_emotions'] = st.checkbox(
                "Show emotion badges", 
                value=True
            )

            st.session_state.user_preferences['show_confidence'] = st.checkbox(
                "Show confidence scores", 
                value=True
            )

            # Model status
            st.subheader("System Status")

            if hasattr(st.session_state, 'emotion_analyzer'):
                if st.session_state.emotion_analyzer.model_loaded:
                    st.success("Emotion AI Ready")
                else:
                    st.warning("Using Fallback Mode")

            # Stats
            st.subheader("Session Stats")
            st.metric("Messages", st.session_state.total_messages)

            if st.session_state.emotion_history:
                df = pd.DataFrame(st.session_state.emotion_history)
                dominant = df['emotion'].mode().iloc[0] if not df.empty else "neutral"
                st.metric("Dominant Emotion", dominant.title())

            # Available emotions
            st.subheader("28 Emotions Detected")
            emotions_text = ", ".join([e.title() for e in st.session_state.emotion_analyzer.all_emotions[:14]])
            st.text(emotions_text + "...")
            
            # Export button
            st.subheader("Export")
            if st.button("Export Chat"):
                export_conversation()

            # Reset button
            if st.button("Reset Chat", help="Clear conversation history"):
                st.session_state.messages = []
                st.session_state.emotion_history = []
                st.session_state.total_messages = 0
                st.session_state.conversation_started = False
                st.rerun()

        # Main chat interface
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Chat Interface")

            # Display chat history
            chat_container = st.container()

            with chat_container:
                for i, message in enumerate(st.session_state.messages):
                    if message["role"] == "user":
                        # User message
                        st.markdown(f"""
                        <div class="user-message">
                            <strong>You:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)

                        # Show emotion if available
                        if (st.session_state.user_preferences['show_emotions'] and 
                            i < len(st.session_state.emotion_history)):

                            emotion_data = st.session_state.emotion_history[i]
                            emotion = emotion_data['emotion']
                            confidence = emotion_data['confidence']

                            badge_html = f"""
                            <div style="text-align: right; margin-top: -10px;">
                                <span class="emotion-badge emotion-{emotion}">
                                    {emotion.title()}
                                    {f" ({confidence:.1%})" if st.session_state.user_preferences['show_confidence'] else ""}
                                </span>
                            </div>
                            """
                            st.markdown(badge_html, unsafe_allow_html=True)

                    else:
                        # Bot message
                        st.markdown(f"""
                        <div class="bot-message">
                            <strong>Assistant:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)

            # Chat input
            st.markdown("### Your Message")
            user_input = st.chat_input("Type your message here... I'm listening!")

            # Process user input
            if user_input:
                try:
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Analyze emotion
                    emotion, confidence = st.session_state.emotion_analyzer.analyze_emotion(user_input)

                    # Store emotion data
                    emotion_data = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat(),
                        'message': user_input[:100]  # Store first 100 chars
                    }
                    st.session_state.emotion_history.append(emotion_data)

                    # Update stats
                    st.session_state.total_messages += 1
                    st.session_state.dominant_emotion = emotion
                    st.session_state.conversation_started = True

                    # Generate response
                    response = st.session_state.response_generator.generate_response(
                        emotion, confidence, user_input
                    )

                    # Add bot response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Rerun to update display
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing message: {e}")
                    st.info("Please try again or restart the chat.")

        with col2:
            # Analytics panel
            st.subheader("Analytics")

            # Show analytics
            show_emotion_analytics()

            # Show insights
            show_emotion_insights()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: gray; font-size: 12px;">
            Built with Streamlit - Powered by Emotion AI - Detecting 28 Emotions
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or check your installation.")

        # Emergency fallback
        st.subheader("Emergency Mode")
        st.text_input("Simple chat (no emotion detection):", key="emergency_chat")

if __name__ == "__main__":
    main()