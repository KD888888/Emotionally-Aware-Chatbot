import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import html
import io
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Emotionally Aware Chatbot - Reddit Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .progress-text {
        font-weight: bold;
        color: #27ae60;
    }
    .error-text {
        color: #e74c3c;
        font-weight: bold;
    }
    .success-text {
        color: #27ae60;
        font-weight: bold;
    }
    .warning-text {
        color: #f39c12;
        font-weight: bold;
    }

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }

    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background-color: #3498db;
        color: white;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'emotion_results' not in st.session_state:
    st.session_state.emotion_results = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Text preprocessing functions
def clean_reddit_text(text):
    """Clean Reddit text data to handle encoding issues and artifacts"""
    if pd.isna(text) or text == '':
        return ""

    text = str(text)
    # Fix common encoding issues
    text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    text = text.replace('â€"', '--').replace('â€"', '-')
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def comprehensive_text_preprocessing(text):
    """Complete text preprocessing pipeline for Reddit comments"""
    if pd.isna(text) or text == '':
        return ""

    # Clean encoding issues first
    text = clean_reddit_text(text)

    # Convert to lowercase
    text = text.lower()

    # Remove Reddit-specific patterns
    text = re.sub(r'\[NAME\]', 'person', text)
    text = re.sub(r'\[RELIGION\]', 'religion', text)
    text = re.sub(r'/u/\w+', 'user', text)
    text = re.sub(r'/r/\w+', 'subreddit', text)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove extra punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Load emotion classification models
@st.cache_resource
def load_emotion_models():
    """Load the 28 Go Emotions classification models"""
    try:
        # SamLowe model
        samlowe_model = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )

        # Alternative model for comparison
        alt_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )

        st.success("✅ Emotion models loaded successfully!")
        return samlowe_model, alt_model

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# Main application header
st.markdown('<div class="main-header">🤖 Emotionally Aware Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Reddit Comment Analysis with 28 Go Emotions</div>', unsafe_allow_html=True)

# Sidebar for controls and info
with st.sidebar:
    st.header("📊 Control Panel")

    # Model loading status
    if st.button("🔄 Load Emotion Models", use_container_width=True):
        with st.spinner("Loading models..."):
            samlowe_model, alt_model = load_emotion_models()
            if samlowe_model is not None:
                st.session_state.models_loaded = True
                st.session_state.samlowe_model = samlowe_model
                st.session_state.alt_model = alt_model

    # File upload section
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Reddit CSV File",
        type=['csv'],
        help="Upload your Reddit comments CSV file with columns: text, id, author, subreddit, etc."
    )

    # Quick actions
    st.header("⚡ Quick Actions")
    if st.button("🏃‍♂️ Run Full Pipeline", use_container_width=True):
        st.session_state.run_full_pipeline = True

    # Progress indicator
    st.header("📈 Progress")
    progress_steps = [
        "Data Upload",
        "Text Preprocessing", 
        "Emotion Analysis",
        "Clustering",
        "Visualization"
    ]

    current_step = 0
    if uploaded_file: current_step = 1
    if st.session_state.processed_data is not None: current_step = 2
    if st.session_state.emotion_results is not None: current_step = 3
    if st.session_state.clustering_results is not None: current_step = 4

    progress = current_step / len(progress_steps)
    st.progress(progress)
    st.write(f"Step {current_step}/{len(progress_steps)}: {progress_steps[min(current_step, len(progress_steps)-1)]}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data Processing", 
    "🧠 Emotion Analysis", 
    "🔍 Clustering Analysis", 
    "📊 Visualizations", 
    "💬 Interactive Chat"
])

with tab1:
    st.header("📁 Data Processing & Preprocessing")

    if uploaded_file is not None:
        try:
            # Load and display data
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.success(f"✅ Successfully loaded {len(df)} Reddit comments!")

            # Data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Comments", f"{len(df):,}")
            with col2:
                st.metric("Unique Authors", df['author'].nunique())
            with col3:
                st.metric("Unique Subreddits", df['subreddit'].nunique())
            with col4:
                st.metric("Data Columns", len(df.columns))

            # Show sample data
            st.subheader("📋 Data Sample")
            st.dataframe(df.head(10))

            # Text preprocessing
            if st.button("🔧 Process Text Data", use_container_width=True):
                with st.spinner("Processing text data..."):
                    # Clean and preprocess text
                    df['cleaned_text'] = df['text'].apply(clean_reddit_text)
                    df['processed_text'] = df['text'].apply(comprehensive_text_preprocessing)

                    # Remove empty texts
                    df = df[df['processed_text'].str.len() > 0]

                    # Store in session state
                    st.session_state.processed_data = df

                    st.success("✅ Text preprocessing completed!")

                    # Show before/after comparison
                    st.subheader("🔄 Before vs After Processing")
                    comparison_df = pd.DataFrame({
                        'Original': df['text'].head(3).values,
                        'Cleaned': df['cleaned_text'].head(3).values,
                        'Processed': df['processed_text'].head(3).values
                    })
                    st.dataframe(comparison_df)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.info("👆 Please upload a Reddit CSV file to begin analysis")

        # Show expected format
        st.subheader("📋 Expected CSV Format")
        sample_data = pd.DataFrame({
            'text': ['That game hurt.', 'Man I love reddit.', 'You do right!'],
            'id': ['eew5j0j', 'eeibobj', 'ed2mah1'],
            'author': ['User1', 'User2', 'User3'],
            'subreddit': ['gaming', 'general', 'advice'],
            'created_utc': [1548381039, 1547965054, 1546427744],
            'rater_id': [1, 18, 37]
        })
        st.dataframe(sample_data)

with tab2:
    st.header("🧠 28 Go Emotions Analysis")

    if st.session_state.processed_data is not None and st.session_state.models_loaded:
        df = st.session_state.processed_data

        # Emotion analysis controls
        st.subheader("🎯 Analysis Configuration")
        col1, col2 = st.columns(2)

        with col1:
            sample_size = st.slider("Sample Size for Analysis", 100, min(len(df), 10000), 1000)
            analysis_column = st.selectbox("Text Column to Analyze", ['cleaned_text', 'processed_text'])

        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
            model_choice = st.selectbox("Model Selection", ["SamLowe GoEmotions", "Both Models"])

        # Run emotion analysis
        if st.button("🔍 Analyze Emotions", use_container_width=True):
            with st.spinner(f"Analyzing emotions for {sample_size} comments..."):
                try:
                    # Sample data for analysis
                    sample_df = df.sample(n=min(sample_size, len(df)))

                    # Initialize results lists
                    emotions_list = []
                    confidence_scores = []

                    # Progress bar for emotion analysis
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, (i, row) in enumerate(sample_df.iterrows()):
                        # Update progress
                        progress = (idx + 1) / len(sample_df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing comment {idx + 1}/{len(sample_df)}")

                        text = row[analysis_column]
                        if len(text) > 0:
                            try:
                                # Get emotion predictions
                                results = st.session_state.samlowe_model(text[:512])  # Limit text length

                                # Find highest confidence emotion
                                best_emotion = max(results[0], key=lambda x: x['score'])

                                if best_emotion['score'] >= confidence_threshold:
                                    emotions_list.append(best_emotion['label'])
                                    confidence_scores.append(best_emotion['score'])
                                else:
                                    emotions_list.append('neutral')
                                    confidence_scores.append(best_emotion['score'])

                            except Exception as e:
                                emotions_list.append('error')
                                confidence_scores.append(0.0)
                        else:
                            emotions_list.append('empty')
                            confidence_scores.append(0.0)

                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Store results
                    sample_df['predicted_emotion'] = emotions_list
                    sample_df['confidence'] = confidence_scores

                    st.session_state.emotion_results = sample_df

                    st.success(f"✅ Emotion analysis completed for {len(sample_df)} comments!")

                    # Show emotion distribution
                    st.subheader("📊 Emotion Distribution")
                    emotion_counts = pd.Series(emotions_list).value_counts()

                    col1, col2 = st.columns(2)

                    with col1:
                        # Pie chart
                        fig_pie = px.pie(
                            values=emotion_counts.values, 
                            names=emotion_counts.index,
                            title="Emotion Distribution"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        # Bar chart
                        fig_bar = px.bar(
                            x=emotion_counts.index, 
                            y=emotion_counts.values,
                            title="Emotion Counts",
                            labels={'x': 'Emotion', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # Show sample results
                    st.subheader("📋 Sample Results")
                    result_display = sample_df[['text', 'predicted_emotion', 'confidence']].head(10)
                    st.dataframe(result_display)

                except Exception as e:
                    st.error(f"❌ Error during emotion analysis: {str(e)}")

    elif st.session_state.processed_data is None:
        st.warning("⚠️ Please process your data in the Data Processing tab first")

    elif not st.session_state.models_loaded:
        st.warning("⚠️ Please load the emotion models first using the sidebar")

    else:
        st.info("🔄 Ready for emotion analysis!")

with tab3:
    st.header("🔍 Advanced Clustering Analysis")

    if st.session_state.emotion_results is not None:
        df_emotions = st.session_state.emotion_results

        # Clustering configuration
        st.subheader("⚙️ Clustering Configuration")
        col1, col2 = st.columns(2)

        with col1:
            clustering_method = st.selectbox("Clustering Method", ["KMeans", "DBSCAN"])
            n_clusters = st.slider("Number of Clusters (KMeans)", 2, 20, 5)

        with col2:
            vectorizer_method = st.selectbox("Text Vectorization", ["Glove", "TF-IDF"])
            dimension_reduction = st.checkbox("Apply PCA Reduction", value=True)

        # Run clustering analysis
        if st.button("🎯 Run Clustering Analysis", use_container_width=True):
            with st.spinner("Performing clustering analysis..."):
                try:
                    # Prepare features for clustering
                    if vectorizer_method == "Glove":
                        # TF-IDF vectorization
                        vectorizer = TfidfVectorizer(
                            max_features=1000,
                            stop_words='english',
                            ngram_range=(1, 2)
                        )
                        X = vectorizer.fit_transform(df_emotions['processed_text'].fillna(''))
                        X = X.toarray()
                    else:
                        # Emotion-based features (dummy encoding)
                        emotion_dummies = pd.get_dummies(df_emotions['predicted_emotion'])
                        confidence_features = df_emotions[['confidence']].values
                        X = np.concatenate([emotion_dummies.values, confidence_features], axis=1)

                    # Apply PCA if requested
                    if dimension_reduction and X.shape[1] > 50:
                        pca = PCA(n_components=50)
                        X = pca.fit_transform(X)
                        explained_variance = pca.explained_variance_ratio_.sum()
                        st.info(f"PCA applied: {explained_variance:.2%} variance explained")

                    # Perform clustering
                    if clustering_method == "KMeans":
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = clusterer.fit_predict(X)
                    else:  # DBSCAN
                        clusterer = DBSCAN(eps=0.5, min_samples=5)
                        cluster_labels = clusterer.fit_predict(X)

                    # Store clustering results
                    df_emotions['cluster'] = cluster_labels
                    st.session_state.clustering_results = df_emotions

                    # Display clustering results
                    st.success("✅ Clustering analysis completed!")

                    # Cluster statistics
                    cluster_stats = df_emotions.groupby('cluster').agg({
                        'predicted_emotion': lambda x: x.mode().iloc[0],
                        'confidence': 'mean',
                        'text': 'count'
                    }).round(3)
                    cluster_stats.columns = ['Dominant_Emotion', 'Avg_Confidence', 'Count']

                    st.subheader("📊 Cluster Statistics")
                    st.dataframe(cluster_stats)

                    # Visualization with PCA for 2D plot
                    if X.shape[1] > 2:
                        pca_viz = PCA(n_components=2)
                        X_2d = pca_viz.fit_transform(X)
                    else:
                        X_2d = X

                    # Scatter plot of clusters
                    fig_cluster = px.scatter(
                        x=X_2d[:, 0], 
                        y=X_2d[:, 1],
                        color=cluster_labels.astype(str),
                        title="Cluster Visualization (PCA 2D)",
                        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error during clustering: {str(e)}")
    else:
        st.warning("⚠️ Please complete emotion analysis first")

with tab4:
    st.header("📊 Advanced Visualizations")

    if st.session_state.emotion_results is not None:
        df_viz = st.session_state.emotion_results

        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization Type",
            [
                "Emotion Timeline",
                "Subreddit Emotion Heatmap", 
                "Confidence Distribution",
                "Author Emotion Patterns",
                "Interactive Dashboard"
            ]
        )

        if viz_type == "Emotion Timeline":
            st.subheader("📈 Emotion Timeline Analysis")

            # Convert timestamp to datetime
            df_viz['datetime'] = pd.to_datetime(df_viz['created_utc'], unit='s')
            df_viz['date'] = df_viz['datetime'].dt.date

            # Aggregate by date and emotion
            timeline_data = df_viz.groupby(['date', 'predicted_emotion']).size().unstack(fill_value=0)

            # Create timeline plot
            fig_timeline = go.Figure()

            for emotion in timeline_data.columns:
                fig_timeline.add_trace(
                    go.Scatter(
                        x=timeline_data.index,
                        y=timeline_data[emotion],
                        mode='lines+markers',
                        name=emotion,
                        line=dict(width=2)
                    )
                )

            fig_timeline.update_layout(
                title="Emotion Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Comment Count",
                hovermode='x unified'
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

        elif viz_type == "Subreddit Emotion Heatmap":
            st.subheader("🔥 Subreddit Emotion Heatmap")

            # Create subreddit-emotion matrix
            heatmap_data = df_viz.groupby(['subreddit', 'predicted_emotion']).size().unstack(fill_value=0)

            # Select top subreddits by activity
            top_subreddits = df_viz['subreddit'].value_counts().head(15).index
            heatmap_data = heatmap_data.loc[heatmap_data.index.intersection(top_subreddits)]

            # Create heatmap
            fig_heatmap = px.imshow(
                heatmap_data,
                labels=dict(x="Emotion", y="Subreddit", color="Count"),
                title="Emotion Distribution Across Subreddits"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

        elif viz_type == "Confidence Distribution":
            st.subheader("📊 Model Confidence Analysis")

            # Confidence distribution by emotion
            fig_conf = px.box(
                df_viz,
                x='predicted_emotion',
                y='confidence',
                title="Confidence Distribution by Emotion"
            )
            fig_conf.update_xaxes(tickangle=45)

            st.plotly_chart(fig_conf, use_container_width=True)

            # Confidence histogram
            fig_hist = px.histogram(
                df_viz,
                x='confidence',
                nbins=30,
                title="Overall Confidence Distribution"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        elif viz_type == "Author Emotion Patterns":
            st.subheader("👤 Author Emotion Patterns")

            # Select top authors
            top_authors = df_viz['author'].value_counts().head(10).index
            author_emotions = df_viz[df_viz['author'].isin(top_authors)]

            # Create author emotion matrix
            author_matrix = author_emotions.groupby(['author', 'predicted_emotion']).size().unstack(fill_value=0)

            # Normalize by author total posts
            author_matrix_norm = author_matrix.div(author_matrix.sum(axis=1), axis=0)

            # Create heatmap
            fig_author = px.imshow(
                author_matrix_norm,
                labels=dict(x="Emotion", y="Author", color="Proportion"),
                title="Emotion Patterns by Top Authors (Normalized)"
            )

            st.plotly_chart(fig_author, use_container_width=True)

        elif viz_type == "Interactive Dashboard":
            st.subheader("🎛️ Interactive Analytics Dashboard")

            # Create dashboard with multiple metrics
            col1, col2 = st.columns(2)

            with col1:
                # Top emotions
                emotion_counts = df_viz['predicted_emotion'].value_counts()
                fig_top_emotions = px.bar(
                    x=emotion_counts.values[:10],
                    y=emotion_counts.index[:10],
                    orientation='h',
                    title="Top 10 Emotions"
                )
                st.plotly_chart(fig_top_emotions, use_container_width=True)

                # Average confidence by emotion
                avg_confidence = df_viz.groupby('predicted_emotion')['confidence'].mean().sort_values(ascending=False)
                fig_confidence = px.bar(
                    x=avg_confidence.index[:10],
                    y=avg_confidence.values[:10],
                    title="Average Confidence by Emotion"
                )
                st.plotly_chart(fig_confidence, use_container_width=True)

            with col2:
                # Top subreddits
                subreddit_counts = df_viz['subreddit'].value_counts().head(10)
                fig_subreddits = px.pie(
                    values=subreddit_counts.values,
                    names=subreddit_counts.index,
                    title="Top Subreddits"
                )
                st.plotly_chart(fig_subreddits, use_container_width=True)

                # Emotion diversity by subreddit
                subreddit_diversity = df_viz.groupby('subreddit')['predicted_emotion'].nunique().sort_values(ascending=False).head(10)
                fig_diversity = px.bar(
                    x=subreddit_diversity.values,
                    y=subreddit_diversity.index,
                    orientation='h',
                    title="Emotion Diversity by Subreddit"
                )
                st.plotly_chart(fig_diversity, use_container_width=True)

    else:
        st.warning("⚠️ Please complete emotion analysis to view visualizations")

with tab5:
    st.header("💬 Interactive Emotion Chat")

    if st.session_state.models_loaded:
        st.subheader("🤖 Real-time Emotion Analysis")

        # Text input for real-time analysis
        user_input = st.text_area(
            "Enter text to analyze emotions:",
            placeholder="Type your message here...",
            height=100
        )

        if st.button("🔍 Analyze This Text", use_container_width=True) and user_input:
            with st.spinner("Analyzing emotions..."):
                try:
                    # Clean and analyze the input
                    cleaned_input = clean_reddit_text(user_input)

                    # Get emotion predictions
                    results = st.session_state.samlowe_model(cleaned_input[:512])

                    # Display results
                    st.subheader("📊 Emotion Analysis Results")

                    # Create results dataframe
                    results_df = pd.DataFrame(results[0])
                    results_df = results_df.sort_values('score', ascending=False)

                    # Top 5 emotions
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Top 5 Detected Emotions:**")
                        for idx, row in results_df.head(5).iterrows():
                            confidence_pct = row['score'] * 100
                            st.markdown(
                                f'<span class="emotion-badge">{row["label"]}: {confidence_pct:.1f}%</span>',
                                unsafe_allow_html=True
                            )

                    with col2:
                        # Emotion bar chart
                        fig_emotions = px.bar(
                            results_df.head(8),
                            x='score',
                            y='label',
                            orientation='h',
                            title="Emotion Confidence Scores"
                        )
                        st.plotly_chart(fig_emotions, use_container_width=True)

                    # Detailed results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)

                except Exception as e:
                    st.error(f"❌ Error analyzing text: {str(e)}")

        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if user_input and st.button("💾 Save to Chat History"):
            try:
                results = st.session_state.samlowe_model(clean_reddit_text(user_input)[:512])
                top_emotion = max(results[0], key=lambda x: x['score'])

                st.session_state.chat_history.append({
                    'text': user_input,
                    'emotion': top_emotion['label'],
                    'confidence': top_emotion['score'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                st.success("💾 Added to chat history!")

            except Exception as e:
                st.error(f"❌ Error saving to history: {str(e)}")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📜 Chat Analysis History")

            for idx, entry in enumerate(reversed(st.session_state.chat_history[-10:])):
                with st.expander(f"Entry {len(st.session_state.chat_history) - idx}: {entry['emotion']} ({entry['confidence']:.2%})"):
                    st.write(f"**Time:** {entry['timestamp']}")
                    st.write(f"**Text:** {entry['text']}")
                    st.write(f"**Detected Emotion:** {entry['emotion']}")
                    st.write(f"**Confidence:** {entry['confidence']:.2%}")

    else:
        st.warning("⚠️ Please load the emotion models first using the sidebar")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">'
    '🤖 Emotionally Aware Chatbot | Advanced Reddit Analysis with 28 Go Emotions | '
    'Built with Streamlit & Transformers'
    '</div>',
    unsafe_allow_html=True
)
