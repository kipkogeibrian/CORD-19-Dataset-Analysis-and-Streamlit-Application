# Create a new file: app.py
# This will be a separate Streamlit application

# Save the following code as app.py and run it with: streamlit run app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import numpy as np

# Set page config
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("CORD-19 Data Explorer")
st.write("Explore metadata from COVID-19 research papers")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('metadata.csv')
        st.success("Successfully loaded metadata.csv")
    except FileNotFoundError:
        st.error("metadata.csv not found. Using sample data for demonstration.")
        # Create sample data
        titles = [
            "COVID-19 transmission dynamics and control measures",
            "Clinical characteristics of coronavirus patients",
            "Vaccine development for SARS-CoV-2",
            "Impact of pandemic on mental health",
            "Economic consequences of COVID-19 lockdowns",
            "Diagnostic testing methods for COVID-19",
            "Treatment options for severe COVID-19 cases",
            "Genomic analysis of SARS-CoV-2 variants",
            "Public health response to COVID-19 pandemic",
            "Modeling the spread of infectious diseases"
        ]
        
        journals = ["The Lancet", "Nature", "Science", "JAMA", "NEJM", "BMJ", "PLOS One", "Cell", "PNAS", "Elsevier"]
        
        sample_data = []
        for i in range(500):
            # Create data across multiple years (2020-2023)
            year = np.random.choice([2020, 2021, 2022, 2023], p=[0.4, 0.3, 0.2, 0.1])
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            publish_date = f"{year}-{month:02d}-{day:02d}"
            
            sample_data.append({
                'title': np.random.choice(titles),
                'journal': np.random.choice(journals + [None], p=[0.9, 0.1]),
                'publish_time': publish_date,
                'abstract': f"Abstract text for paper {i} about COVID-19 research" if np.random.random() > 0.2 else None,
            })
        
        df = pd.DataFrame(sample_data)
    
    # Clean data
    df = df.dropna(subset=['title'])
    df['abstract'] = df['abstract'].fillna('')
    df['journal'] = df['journal'].fillna('Unknown')
    
    # Handle date conversion with error checking
    try:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['publish_time'])
        df['year'] = df['publish_time'].dt.year
    except Exception as e:
        st.warning(f"Date conversion failed: {e}. Using default year 2020.")
        df['year'] = 2020
    
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
    
    return df

df = load_data()

# Sidebar with filters
st.sidebar.header("Filters")

# Handle year range slider - ensure min and max are different
min_year = int(df['year'].min())
max_year = int(df['year'].max())

if min_year == max_year:
    # If only one year exists, create a range around it
    year_range = st.sidebar.slider(
        "Select year range",
        min_year - 1,  # Extend range by 1 year below
        max_year + 1,  # Extend range by 1 year above
        (min_year, max_year)
    )
    # Filter to only include the actual year if it's the only one
    filtered_df = df[df['year'] == min_year]
else:
    year_range = st.sidebar.slider(
        "Select year range",
        min_year,
        max_year,
        (min_year, max_year)
    )
    # Filter data based on year selection
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]

# Journal selection
selected_journals = st.sidebar.multiselect(
    "Select journals",
    options=df['journal'].unique(),
    default=df['journal'].unique()[:5] if len(df['journal'].unique()) > 5 else df['journal'].unique()
)

# Apply journal filter
if selected_journals:
    filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]

# Display dataset info
st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Papers", len(df))
col2.metric("Filtered Papers", len(filtered_df))
col3.metric("Unique Journals", df['journal'].nunique())
col4.metric("Year Range", f"{min_year} - {max_year}")

# Show warning if no data after filtering
if len(filtered_df) == 0:
    st.warning("No data matches your filters. Please adjust your selection.")
    st.stop()

# Visualizations
st.header("Visualizations")

# Create two columns for charts
col1, col2 = st.columns(2)

# Publications by year
with col1:
    st.subheader("Publications by Year")
    year_counts = filtered_df['year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    year_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Number of Publications by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    plt.xticks(rotation=0)
    st.pyplot(fig)

# Top journals
with col2:
    st.subheader("Top Journals")
    top_journals = filtered_df['journal'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    top_journals.plot(kind='bar', ax=ax, color='lightcoral')
    ax.set_title('Top Journals by Number of Publications')
    ax.set_xlabel('Journal')
    ax.set_ylabel('Number of Publications')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# Word cloud
st.subheader("Word Cloud of Paper Titles")
all_titles = ' '.join(filtered_df['title'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Word Cloud of Paper Titles')
st.pyplot(fig)

# Most common words
st.subheader("Most Common Words in Titles")

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    stopwords = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'by', 'an', 'at', 'is', 'are', 'as', 'from', 'that', 'this', 'was', 'were', 'be', 'been', 'being'}
    words = [word for word in text.split() if word not in stopwords and len(word) > 2]
    return words

all_words = []
for title in filtered_df['title']:
    all_words.extend(clean_text(title))

word_freq = Counter(all_words)
common_words = word_freq.most_common(15)

words, counts = zip(*common_words)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(words, counts, color='lightgreen')
ax.set_title('Top 15 Most Frequent Words in Paper Titles')
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Abstract length distribution
st.subheader("Abstract Length Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(filtered_df['abstract_word_count'], bins=20, color='purple', alpha=0.7)
ax.set_title('Distribution of Abstract Word Counts')
ax.set_xlabel('Word Count')
ax.set_ylabel('Number of Papers')
ax.axvline(filtered_df['abstract_word_count'].mean(), color='red', linestyle='dashed', linewidth=1, 
           label=f'Mean: {filtered_df["abstract_word_count"].mean():.1f}')
ax.legend()
st.pyplot(fig)

# Show sample data
st.header("Sample Data")
st.dataframe(filtered_df[['title', 'journal', 'year', 'abstract_word_count']].head(10))

# Footer
st.markdown("---")
st.markdown("CORD-19 Data Explorer | Built with Streamlit")