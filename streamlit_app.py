# Streamlit app entry point
import os
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_authenticator as stauth
from wordcloud import WordCloud

# ── 1. Authentication ──────────────────────────────────────────────────────────
# Using the new API format - let the library auto-hash the password
credentials = {
    "usernames": {
        "admin": {
            "name": "Admin",
            "password": "*6996@QweQu#"  # This will be hashed automatically
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="galamsey_cookie", 
    key="galamsey_key",
    cookie_expiry_days=1
)

# Use the new login method with proper location parameter
try:
    authenticator.login(location="main")
except Exception as e:
    st.error(f"Login error: {e}")

# Check authentication status using session state
if st.session_state.get("authentication_status") is False:
    st.error("❌ Username/password incorrect")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")
    st.stop()

# If authenticated, show logout and welcome message
if st.session_state.get("authentication_status"):
    authenticator.logout(location="sidebar")
    st.sidebar.write(f"Hello, **{st.session_state.get('name')}**!")

    # ── 2. Load Data & Last‑Updated Badge ──────────────────────────────────────────
    DATA_PATH = "data/processed/twitter_with_emotions.csv"
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}")
        st.info("Please upload the required data file to continue.")
        st.stop()
    
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    mod_time = datetime.fromtimestamp(os.path.getmtime(DATA_PATH))
    st.sidebar.markdown(f"**Last updated:** {mod_time:%Y-%m-%d %H:%M:%S}")

    # ── 3. Sidebar: Interactive Filters ────────────────────────────────────────────
    st.sidebar.subheader("Filters")

    # Date range selector
    start, end = st.sidebar.date_input(
        "Date range",
        value=[df["date"].min(), df["date"].max()]
    )
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    filtered = df.loc[mask]

    # Emotion multiselect
    emotions = st.sidebar.multiselect(
        "Emotions",
        options=sorted(df["dominant_emotion"].unique()),
        default=sorted(df["dominant_emotion"].unique())
    )
    filtered = filtered[filtered["dominant_emotion"].isin(emotions)]

    # ── 4. Main Dashboard ─────────────────────────────────────────────────────────
    st.title("Galamsey Sentiment & Emotion Dashboard")

    # Sentiment Distribution
    st.subheader("RoBERTa Sentiment Labels")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="roberta_label", data=filtered, ax=ax1)
    ax1.set_xlabel("Sentiment")
    st.pyplot(fig1)

    # Emotion Distribution
    st.subheader("Dominant Emotions in Tweets")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="dominant_emotion", data=filtered, ax=ax2)
    ax2.set_xlabel("Emotion")
    st.pyplot(fig2)

    # VADER Time‑Series
    st.subheader("VADER Sentiment Over Time")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.lineplot(x="date", y="vader_score", data=filtered, ax=ax3)
    ax3.set_ylabel("VADER Compound Score")
    st.pyplot(fig3)

    # Word Cloud
    st.subheader("Tweet Word Cloud")
    text = " ".join(filtered["content"].dropna())
    if text.strip():  # Only create wordcloud if there's text
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.imshow(wordcloud, interpolation="bilinear")
        ax4.axis("off")
        st.pyplot(fig4)
    else:
        st.info("No text data available for word cloud")

    # ── 5. Tweet Explorer ──────────────────────────────────────────────────────────
    st.subheader("Tweet Explorer")
    st.dataframe(
        filtered[["date", "username", "roberta_label", "dominant_emotion", "content"]],
        height=300
    )