import streamlit as st
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
    <style>
        .stApp {
            background-color: #1a1a2e;
        }
        
        h1 {
            color: #f5b8c9 !important;
            font-size: 2.8rem !important;
            letter-spacing: 2px;
        }

        h2 {
            color: #e8d5e8 !important;
            font-weight: 300 !important;
            font-size: 1.1rem !important;
            letter-spacing: 1px;
        }

        .stTextArea label {
            color: #e8d5e8 !important;
            font-size: 0.9rem !important;
            letter-spacing: 1px;
        }  

        div[data-testid="stTextArea"] {
            text-align: center;
        }
        .stTextArea textarea:focus {
            border: 1px solid #f5b8c9 !important;
            box-shadow: 0 0 8px rgba(245, 184, 201, 0.3) !important;
        }

        button[kind="primary"], .stButton > button {
            background-color: #f5b8c9 !important;
            color: #1a1a2e !important;
            border: none !important;
            border-radius: 20px !important;
            font-weight: 600 !important;
        }

        .stButton button:hover {
            background-color: #e8a0b4 !important;
            color: #1a1a2e !important;
        }
        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
            margin-top: 12px;
        }

        .card {
            background-color: #2a1f2e;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 16px;
            border-left: 3px solid #f5b8c9;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(245, 184, 201, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

df=pd.read_csv("afterglow_dataset.csv")

df["combined"]=(df["mood_tags"]+" "+ df["description"]+ " " + df["after_watch_suggests"] + " " + df["emotional_tone"])
# custom_stops={"watch", "something", "like", "equally", "feel", "want"}
# all_stops=ENGLISH_STOP_WORDS | custom_stops
# vectorizer=TfidfVectorizer(stop_words=list(all_stops))
# tf_idf_matrix=vectorizer.fit_transform(df["combined"])
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["combined"].tolist(), show_progress_bar=True)

# recommendation function
def recommend(mood, top_n=5):
    mood_embedding=model.encode([mood])
    similarities = cosine_similarity(mood_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[::-1]
    
    results = []
    for idx in top_indices:
        title = df.iloc[idx]["title"]
        entry_type = df.iloc[idx]["type"]
        # skip if title is mentioned in mood description
        if title.lower() not in mood.lower():
            if filter_type=='all' or entry_type==filter_type:
                results.append(idx)
        if len(results) == top_n:
            break
    
    return df.iloc[results][["title", "type", "mood_tags", "emotional_tone"]].reset_index(drop=True)


st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="color:#f5b8c9; font-size:3rem; letter-spacing:3px; margin-bottom:0;">
            ⋆｡°✩ AfterGlow
        </h1>
        <p style="color:#e8d5e8; font-size:1rem; font-weight:300; margin-top:8px; letter-spacing:1px;">
            for when you finish something and don't know what to feel
        </p>
        <div style="width:60px; height:2px; background:#f5b8c9; margin:16px auto; border-radius:2px; opacity:0.6;"></div>
        <p style="color:#c9a8c9; font-size:0.85rem; font-weight:300; letter-spacing:1px;">
            tell us how you feel. we'll find your next obsession ✦
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#e8d5e8; font-size:0.9rem; letter-spacing:1px;'>describe your mood or what you've finished watching.</p>", unsafe_allow_html=True)
mood = st.text_area("", placeholder="i just finished attack on titan and i feel empty...")

filter_type=st.selectbox(
    "filter by type(optional)", 
    ["all", "anime", "web series", "book", "movie"]
)
col1, col2, col3 = st.columns([2, 2, 2])
with col2:
    clicked = st.button("find my new obsession ✦")

if clicked:
    if mood.strip()=="":
        st.markdown("""
    <div style="background-color:#2a1f2e; border-left: 3px solid #f5b8c9; 
    padding:12px 20px; border-radius:12px; margin-bottom:16px;">
        <p style="color:#f5b8c9; margin:0;">tell us how you're feeling first ♥︎ </p>
    </div>
""", unsafe_allow_html=True)
    else:
        results=recommend(mood)
        st.markdown("""
    <div style="background-color:#2a1f2e; border-left: 3px solid #f5b8c9; 
    padding:12px 20px; border-radius:12px; margin-bottom:16px;">
        <p style="color:#f5b8c9; margin:0; font-size:1rem;">
            here's what we think you'll love next ❤︎₊ ⊹
        </p>
    </div>
""", unsafe_allow_html=True)
        for i, row in results.iterrows():
            st.markdown(f"""
            <div class="card">
                <h4 style="color:#f5b8c9; margin:0">{i+1}. {row['title']} <span style="font-size:0.8rem; color:#e8d5e8;">— {row['type']}</span></h4>
                <p style="color:#c9a8c9; margin:8px 0 4px 0; font-size:0.85rem;">✦ mood: {row['mood_tags']}</p>
                <p style="color:#c9a8c9; margin:0; font-size:0.85rem;">✦ tone: {row['emotional_tone']}</p>
            </div>
    """, unsafe_allow_html=True)