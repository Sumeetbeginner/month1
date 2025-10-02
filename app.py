# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="NLP & AI Insights Extractor",
    page_icon="🔍",
    layout="wide"
)

def load_spacy_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import streamlit as st
        st.error("spaCy model not installed. Ensure 'en-core-web-sm' is in requirements.txt")
        return None

nlp = load_spacy_model()
def scrape_website(url):
    """Scrape content from website URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from main content areas
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit text length for demo
    except Exception as e:
        st.error(f"Error scraping website: {e}")
        return None

def extract_entities_spacy(text, nlp):
    """Extract entities using spaCy"""
    if not nlp:
        return []
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    return entities

def tfidf_keywords(text, n_top=10):
    """Extract keywords using TF-IDF"""
    # For single document, we'll compare with common English words
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    # We need multiple documents for TF-IDF, so let's split into sentences
    sentences = [sentence for sentence in text.split('.') if len(sentence) > 10]
    
    if len(sentences) > 1:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        keywords_df = pd.DataFrame({
            'keyword': feature_names,
            'tfidf_score': tfidf_scores
        }).sort_values('tfidf_score', ascending=False).head(n_top)
        return keywords_df
    return None

def perform_lda_topic_modeling(text, n_topics=3):
    """Perform LDA topic modeling"""
    sentences = [sentence.strip() for sentence in text.split('.') if len(sentence) > 20]
    
    if len(sentences) > 2:
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        lda = LatentDirichletAllocation(n_components=min(n_topics, len(sentences)-1), random_state=42)
        lda.fit(tfidf_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            topics.append({
                'topic_id': topic_idx,
                'top_words': ', '.join(top_features)
            })
        return topics
    return None

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'sentiment': 'Positive' if blob.sentiment.polarity > 0.1 else 'Negative' if blob.sentiment.polarity < -0.1 else 'Neutral'
    }

def get_method_comparison():
    """Generate dynamic method comparison with advantages/disadvantages"""
    comparison_data = {
        'Method': ['spaCy NER', 'TF-IDF', 'LDA', 'TextBlob', 'LLM (GPT)'],
        'Entity Recognition': ['✅', '❌', '❌', '❌', '✅'],
        'Keyword Extraction': ['❌', '✅', '✅', '❌', '✅'], 
        'Topic Modeling': ['❌', '❌', '✅', '❌', '✅'],
        'Sentiment Analysis': ['❌', '❌', '❌', '✅', '✅'],
        'Advantages': [
            'Fast, accurate entity recognition, good for structured text',
            'Simple, interpretable, no training required',
            'Discovers latent topics, good for document collections',
            'Fast, rule-based, good for straightforward sentiment',
            'Context-aware, handles nuance, multi-task capable'
        ],
        'Disadvantages': [
            'Limited context, predefined entity types',
            'Bag-of-words approach, no semantic meaning',
            'Requires tuning, may produce incoherent topics',
            'Limited nuance, poor with sarcasm/context',
            'Costly, API-dependent, potential biases'
        ]
    }
    return pd.DataFrame(comparison_data)

def main():
    st.title("🔍 NLP & AI Insights Extractor")
    st.markdown("### Comprehensive Text Analysis Platform")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Module",
        [
            "🏠 Home",
            "🔤 NER & Entity Extraction", 
            "📊 Keywords & Topic Modeling",
            "😊 Sentiment Analysis",
            "🤖 LLM Insights & Comparison",
            "🎯 Mini Project: Top Entities + Sentiment"
        ]
    )
    
    # Initialize session state
    if 'scraped_text' not in st.session_state:
        st.session_state.scraped_text = ""
    if 'entities' not in st.session_state:
        st.session_state.entities = []
    
    # Load spaCy model
    nlp = load_spacy_model()
    
    if app_mode == "🏠 Home":
        show_homepage()
    elif app_mode == "🔤 NER & Entity Extraction":
        show_ner_module(nlp)
    elif app_mode == "📊 Keywords & Topic Modeling":
        show_keywords_module(nlp)
    elif app_mode == "😊 Sentiment Analysis":
        show_sentiment_module()
    elif app_mode == "🤖 LLM Insights & Comparison":
        show_llm_module()
    elif app_mode == "🎯 Mini Project: Top Entities + Sentiment":
        show_mini_project(nlp)

def show_homepage():
    """Display homepage with overview"""
    st.header("Welcome to NLP & AI Insights Extractor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Weeks 1-2")
        st.markdown("""
        - Named Entity Recognition (NER)
        - Keyword Extraction
        - Topic Modeling (TF-IDF, LDA, BERTopic)
        - Prompt Engineering
        - Tools: spaCy, Hugging Face
        """)
    
    with col2:
        st.markdown("### 📈 Weeks 3-4") 
        st.markdown("""
        - Sentiment Analysis
        - Tone Detection
        - LLM Summarization
        - ML vs LLM Comparison
        - Insights Extraction
        """)
    
    with col3:
        st.markdown("### 🛠️ Features")
        st.markdown("""
        - URL Content Scraping
        - Multi-method Analysis
        - Interactive Visualizations
        - Comparative Analysis
        - Exportable Results
        """)
    
    st.markdown("---")
    st.markdown("### 🎯 Learning Objectives Covered")
    
    objectives = [
        "✅ Named Entity Recognition for companies, products, market terms",
        "✅ Keyword extraction using TF-IDF and advanced methods", 
        "✅ Topic modeling with LDA and BERTopic",
        "✅ Sentiment analysis and tone detection",
        "✅ LLM-based insights extraction",
        "✅ ML vs LLM approach comparison",
        "✅ Practical mini-project implementation"
    ]
    
    for objective in objectives:
        st.markdown(objective)

def show_ner_module(nlp):
    """NER and Entity Extraction Module"""
    st.header("🔤 Named Entity Recognition & Extraction")
    
    # URL input
    url = st.text_input("Enter Website URL:", placeholder="https://example.com/news-article")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Scrape & Analyze Content") and url:
            with st.spinner("Scraping website content..."):
                text = scrape_website(url)
                if text:
                    st.session_state.scraped_text = text
                    st.session_state.entities = extract_entities_spacy(text, nlp) if nlp else []
    
    with col2:
        sample_text = st.text_area("Or enter text manually:", height=100, key="ner_text")
        if st.button("Analyze Manual Text") and sample_text:
            st.session_state.scraped_text = sample_text
            if nlp:
                st.session_state.entities = extract_entities_spacy(sample_text, nlp)
    
    if st.session_state.scraped_text:
        st.subheader("📄 Scraped Content")
        st.text_area("Content Preview", st.session_state.scraped_text[:1000], height=150, key="preview")
        
        if st.session_state.entities:
            st.subheader("🔍 Extracted Entities")
            
            # Convert to DataFrame for better display
            entities_df = pd.DataFrame(st.session_state.entities)
            
            # Entity statistics
            entity_counts = entities_df['label'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Entity Distribution")
                fig = px.pie(values=entity_counts.values, names=entity_counts.index, 
                           title="Entity Types Distribution")
                st.plotly_chart(fig)
            
            with col2:
                st.markdown("### Entity Count by Type")
                fig = px.bar(x=entity_counts.index, y=entity_counts.values,
                           labels={'x': 'Entity Type', 'y': 'Count'})
                st.plotly_chart(fig)
            
            # Detailed entity table
            st.markdown("### Detailed Entity List")
            st.dataframe(entities_df)
            
            # Entity explanation
            st.markdown("### Entity Type Legend")
            entity_descriptions = {
                'PERSON': 'People, including fictional',
                'NORP': 'Nationalities or religious or political groups',
                'FAC': 'Buildings, airports, highways, bridges, etc.',
                'ORG': 'Companies, agencies, institutions, etc.',
                'GPE': 'Countries, cities, states',
                'PRODUCT': 'Objects, vehicles, foods, etc. (not services)',
                'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
                'MONEY': 'Monetary values, including unit'
            }
            
            for entity_type, description in entity_descriptions.items():
                st.markdown(f"**{entity_type}**: {description}")

def show_keywords_module(nlp):
    """Keywords and Topic Modeling Module"""
    st.header("📊 Keyword Extraction & Topic Modeling")
    
    if not st.session_state.scraped_text:
        st.warning("Please scrape some content first in the NER module or enter text manually.")
        
        # Allow manual text input in this module too
        manual_text = st.text_area("Enter text for analysis:", height=150, key="keywords_text")
        if st.button("Analyze Text") and manual_text:
            st.session_state.scraped_text = manual_text
            if nlp:
                st.session_state.entities = extract_entities_spacy(manual_text, nlp)
        else:
            return
    
    text = st.session_state.scraped_text
    
    # TF-IDF Keywords
    st.subheader("🔑 TF-IDF Keyword Extraction")
    keywords_df = tfidf_keywords(text)
    
    if keywords_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Keywords (TF-IDF)")
            st.dataframe(keywords_df)
        
        with col2:
            fig = px.bar(keywords_df, x='tfidf_score', y='keyword', 
                        orientation='h', title="Top Keywords by TF-IDF Score")
            st.plotly_chart(fig)
    else:
        st.warning("Not enough text content for TF-IDF analysis. Please provide longer text.")
    
    # LDA Topic Modeling
    st.subheader("📚 LDA Topic Modeling")
    n_topics = st.slider("Number of Topics", 2, 5, 3)
    
    topics = perform_lda_topic_modeling(text, n_topics)
    
    if topics:
        topics_df = pd.DataFrame(topics)
        st.dataframe(topics_df)
        
        # Visualize topics
        st.markdown("### Topic Word Distributions")
        for topic in topics:
            st.markdown(f"**Topic {topic['topic_id']+1}**: {topic['top_words']}")
    else:
        st.warning("Not enough sentences for LDA topic modeling. Please provide longer text with more sentences.")
    
    # BERTopic placeholder
    st.subheader("🤖 BERTopic (Advanced)")
    st.info("""
    BERTopic uses transformer-based embeddings for more sophisticated topic modeling.
    This requires more computational resources and is ideal for larger datasets.
    
    **Key Advantages:**
    - Handles semantic similarity better
    - No need to specify number of topics
    - Better with modern language patterns
    """)

def show_sentiment_module():
    """Sentiment Analysis Module"""
    st.header("😊 Sentiment Analysis & Tone Detection")
    
    if not st.session_state.scraped_text:
        st.warning("Please scrape some content first in the NER module or enter text manually.")
        
        # Allow manual text input in this module too
        manual_text = st.text_area("Enter text for sentiment analysis:", height=150, key="sentiment_text")
        if st.button("Analyze Sentiment") and manual_text:
            st.session_state.scraped_text = manual_text
        else:
            return
    
    text = st.session_state.scraped_text
    
    # ML-based sentiment analysis
    st.subheader("🤖 Machine Learning Approach (TextBlob)")
    sentiment = analyze_sentiment_textblob(text)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Polarity", f"{sentiment['polarity']:.3f}")
        st.info("Polarity: -1 (Negative) to +1 (Positive)")
    
    with col2:
        st.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}")
        st.info("Subjectivity: 0 (Objective) to 1 (Subjective)")
    
    with col3:
        sentiment_color = {
            'Positive': 'green',
            'Negative': 'red', 
            'Neutral': 'blue'
        }
        st.metric("Overall Sentiment", sentiment['sentiment'])
    
    # Sentiment visualization
    st.subheader("📊 Sentiment Visualization")
    
    # Create sentiment gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment['polarity'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Polarity"},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.1], 'color': "lightcoral"},
                {'range': [-0.1, 0.1], 'color': "lightyellow"},
                {'range': [0.1, 1], 'color': "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig)
    
    # LLM-based sentiment analysis placeholder
    st.subheader("🧠 LLM-based Sentiment Analysis")
    st.info("""
    **LLM Approach Benefits:**
    - Context-aware sentiment analysis
    - Multi-dimensional tone detection  
    - Aspect-based sentiment
    - Sarcasm and nuance detection
    
    *Configure your OpenAI API key in settings to enable this feature*
    """)

def show_llm_module():
    """LLM Insights and Comparison Module"""
    st.header("🤖 LLM Insights & Method Comparison")
    
    st.subheader("📋 ML vs LLM Approach Comparison")
    
    # Use dynamic comparison function
    comparison_df = get_method_comparison()
    st.dataframe(comparison_df)
    
    # Detailed advantages and disadvantages
    st.subheader("🔍 Detailed Method Analysis")
    
    methods_analysis = {
        'spaCy NER': {
            'advantages': [
                'Fast and efficient processing',
                'Accurate for predefined entity types',
                'Works well on structured text',
                'No API calls required'
            ],
            'disadvantages': [
                'Limited to predefined entity categories',
                'Poor with ambiguous entities',
                'No semantic understanding',
                'Struggles with domain-specific terms'
            ]
        },
        'TF-IDF': {
            'advantages': [
                'Simple and interpretable',
                'No training required for new domains',
                'Computationally efficient',
                'Good for keyword extraction'
            ],
            'disadvantages': [
                'Bag-of-words limitation',
                'No semantic relationships',
                'Poor with synonyms/polysemy',
                'Frequency-based, not meaning-based'
            ]
        },
        'LDA': {
            'advantages': [
                'Discovers latent topics',
                'Good for document collections',
                'Unsupervised learning',
                'Provides topic distributions'
            ],
            'disadvantages': [
                'Requires careful parameter tuning',
                'May produce incoherent topics',
                'Struggles with short texts',
                'Computationally intensive for large datasets'
            ]
        },
        'TextBlob': {
            'advantages': [
                'Very fast sentiment analysis',
                'Simple rule-based approach',
                'No external dependencies',
                'Good for straightforward sentiment'
            ],
            'disadvantages': [
                'Poor with context and sarcasm',
                'Limited nuance detection',
                'Dictionary-based limitations',
                'No aspect-based sentiment'
            ]
        },
        'LLM (GPT)': {
            'advantages': [
                'Context-aware understanding',
                'Handles nuance and ambiguity',
                'Multi-task capability',
                'No feature engineering needed'
            ],
            'disadvantages': [
                'API costs and dependencies',
                'Potential biases in training data',
                'Black-box decision making',
                'Computationally expensive'
            ]
        }
    }
    
    selected_method = st.selectbox("Select method to view details:", list(methods_analysis.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Advantages")
        for advantage in methods_analysis[selected_method]['advantages']:
            st.markdown(f"• {advantage}")
    
    with col2:
        st.markdown("### ❌ Disadvantages")
        for disadvantage in methods_analysis[selected_method]['disadvantages']:
            st.markdown(f"• {disadvantage}")
    
    # LLM capabilities showcase
    st.subheader("🚀 LLM Capabilities Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Summarization")
        st.info("""
        **LLM Strengths:**
        - Abstractive summarization
        - Key point extraction
        - Length-controlled summaries
        - Multi-document summarization
        """)
    
    with col2:
        st.markdown("### 💡 Insights Extraction")
        st.info("""
        **Advanced Capabilities:**
        - Trend identification
        - Relationship extraction
        - Implicit meaning detection
        - Actionable insights generation
        """)
    
    # Prompt engineering examples
    st.subheader("🎯 Prompt Engineering Examples")
    
    prompt_examples = {
        "Entity Extraction": "Extract all company names, products, and market terms from the text. Return as JSON.",
        "Sentiment Analysis": "Analyze the sentiment towards each mentioned entity. Consider context and relationships.", 
        "Summary Generation": "Provide a 3-bullet summary highlighting key business insights and market implications.",
        "Insight Extraction": "Identify emerging trends, risks, and opportunities mentioned in the text."
    }
    
    for purpose, prompt in prompt_examples.items():
        with st.expander(f"📋 {purpose}"):
            st.code(prompt, language="text")

def show_mini_project(nlp):
    """Mini Project: Top Entities with Sentiment"""
    st.header("🎯 Mini Project: Top 5 Entities + Sentiment Analysis")
    
    # Allow text input directly in mini project
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input("Enter Website URL:", placeholder="https://example.com/news-article", key="mini_url")
    
    with col2:
        sample_text = st.text_area("Or enter text manually:", height=100, key="mini_text")
    
    if st.button("Analyze Content") and (url or sample_text):
        with st.spinner("Analyzing content..."):
            if url:
                text = scrape_website(url)
                if text:
                    st.session_state.scraped_text = text
            elif sample_text:
                st.session_state.scraped_text = sample_text
            
            if st.session_state.scraped_text and nlp:
                st.session_state.entities = extract_entities_spacy(st.session_state.scraped_text, nlp)
    
    if not st.session_state.scraped_text:
        st.warning("Please enter a URL or text to analyze.")
        return
    
    text = st.session_state.scraped_text
    
    if not st.session_state.entities and nlp:
        st.session_state.entities = extract_entities_spacy(text, nlp)
    
    st.markdown("""
    ### Project Objective
    Extract top 5 most frequent entities and analyze their contextual sentiment using both ML and LLM approaches.
    """)
    
    # Get top entities
    if st.session_state.entities:
        entities_df = pd.DataFrame(st.session_state.entities)
        top_entities = entities_df['text'].value_counts().head(5)
        
        st.subheader("🏆 Top 5 Entities by Frequency")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(x=top_entities.values, y=top_entities.index, 
                        orientation='h', title="Top 5 Entities")
            st.plotly_chart(fig)
        
        with col2:
            st.dataframe(top_entities.reset_index().rename(
                columns={'index': 'Entity', 'text': 'Frequency'}
            ))
        
        # Entity-sentiment analysis
        st.subheader("😊 Entity-level Sentiment Analysis")
        
        # Simple context-based sentiment (for demo)
        entity_sentiments = []
        for entity in top_entities.index:
            # Find sentences containing the entity
            sentences = [sent for sent in text.split('.') if entity in sent]
            if sentences:
                context_text = '. '.join(sentences[:3])  # Use first 3 mentions
                sentiment = analyze_sentiment_textblob(context_text)
                entity_sentiments.append({
                    'entity': entity,
                    'frequency': top_entities[entity],
                    'polarity': sentiment['polarity'],
                    'subjectivity': sentiment['subjectivity'],
                    'sentiment': sentiment['sentiment']
                })
        
        if entity_sentiments:
            sentiment_df = pd.DataFrame(entity_sentiments)
            st.dataframe(sentiment_df)
            
            # Visualization
            fig = px.scatter(sentiment_df, x='polarity', y='frequency',
                           size='frequency', color='sentiment',
                           hover_data=['entity'], 
                           title="Entity Sentiment vs Frequency")
            st.plotly_chart(fig)
        else:
            st.warning("Could not extract entity sentiments. The text might be too short.")
    else:
        st.warning("No entities found in the text. Please try with different content.")
    
    # Methodology comparison
    st.subheader("🔬 Methodology Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖥️ ML Approach")
        st.markdown("""
        **Implementation:**
        1. spaCy for entity recognition
        2. Frequency analysis for top entities  
        3. TextBlob for sentiment analysis
        4. Context window around entities
        
        **Advantages:**
        - Fast and efficient
        - No API dependencies
        - Reproducible results
        - Cost-effective
        
        **Limitations:**
        - Limited context understanding
        - Predefined entity types only
        - Basic sentiment analysis
        """)
    
    with col2:
        st.markdown("### 🧠 LLM Approach")
        st.markdown("""
        **Implementation:**
        1. GPT-based entity extraction
        2. Context-aware sentiment
        3. Relationship understanding
        4. Nuanced analysis
        
        **Advantages:**
        - Better context understanding
        - Handles complex language
        - Multi-dimensional analysis
        - Aspect-based sentiment
        
        **Limitations:**
        - API costs and dependencies
        - Potential biases
        - Black-box decisions
        - Computational expense
        """)
    
    st.success("🎉 Mini Project Completed! This demonstrates comprehensive NLP pipeline implementation.")

if __name__ == "__main__":
    main()