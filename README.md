# Emotionally-Aware-Chatbot
Emotionally aware chatbot built in Streamlit using advanced NLP, clustering, and visualization pipelines. Analyzes Reddit comments via transformer-based emotion models, text mining, and interactive dashboards for real-time emotional intelligence insights.

**Overview**

This project showcases a data-driven, emotionally intelligent chatbot built using Streamlit, HuggingFace Transformers, and advanced text mining.
It combines real-time NLP, emotion classification, clustering analytics, and interactive dashboards into a unified emotional intelligence system for unstructured social data.

**Product Objective**

Create a platform that understands how people feel, not just what they say.
It analyzes large-scale Reddit discussions to surface emotional patterns, behavioral trends, and empathy signals — valuable for brands, researchers, and community managers.

**Key Capabilities**

Multi-model Emotion Detection	Uses 28-class GoEmotions (RoBERTa & DistilRoBERTa) for nuanced emotional tagging.
Text Mining & Cleaning	Comprehensive Reddit-specific preprocessing pipeline for noise reduction.
Interactive Analytics	Live dashboards for emotion distribution, subreddit heatmaps, author patterns, and timelines.
Unsupervised Clustering	K-Means / DBSCAN to discover emotion-driven community clusters.
Emotion-Aware Chat Interface	Real-time inference with sentiment confidence scores and historical tracking.

**Architecture Overview**
Reddit CSV → Preprocessing → Emotion Models → Clustering → Visualization → Chat Interface

**Modules:**

clean_reddit_text() – Fixes Reddit encoding and artifacts

load_emotion_models() – Loads GoEmotions pipelines from HuggingFace

emotion_analysis() – Applies model inference with confidence thresholds

clustering_module() – KMeans / DBSCAN over TF-IDF or emotion vectors

visualization_layer() – Streamlit dashboard using Plotly and Seaborn

**Tech Stack**

Frontend / UI: Streamlit

ML Frameworks: HuggingFace Transformers, PyTorch

Data Science: Scikit-Learn, Pandas, NumPy

Visualization: Plotly, Seaborn, Matplotlib

Models: SamLowe/roberta-base-go_emotions, j-hartmann/emotion-english-distilroberta-base

**How to Run**
1. Clone the repo
   git clone https://github.com/yourusername/emotion-aware-chatbot.git
   cd emotion-aware-chatbot
2. Create virtual environment
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
3. Install dependencies
  pip install -r requirements.txt
4. Run the Streamlit app
   streamlit run main_backend.py

**Business Use Case**

Emotionally aware NLP systems are foundational in:

Customer Experience (CX): Detecting frustration or delight in feedback streams.

Community Management: Tracking emotional health across online communities.

Brand Monitoring: Quantifying audience sentiment across user groups.

Therapeutic AI: Enabling empathic conversational agents that adapt tone and context.

This prototype demonstrates how data-centric empathy can be operationalized into scalable digital products.


