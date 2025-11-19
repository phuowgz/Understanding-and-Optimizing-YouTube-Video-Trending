import pandas as pd
import json
from collections import Counter
from tabulate import tabulate
import os
import spacy
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import unidecode
from spacy.lang.es import Spanish
import warnings
warnings.filterwarnings('ignore', category=Warning)

def analyze_tags(country_code):
    """
    Analyze popular tags and title keywords across categories and analyze keyword sentiment.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').

    Returns:
        pd.DataFrame: DataFrame containing tag analysis results.
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load spaCy model with optimized settings
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.max_length = 2000000
        if country_code == 'MX':
            nlp_es = Spanish()
            stopwords_es = nlp_es.Defaults.stop_words
        else:
            stopwords_es = set()
    except Exception as e:
        print(f"⚠️ Failed to load spaCy model: {e}")
        print("⚠️ Install with: python -m spacy download en_core_web_sm")
        nlp = None
        stopwords_es = set()

    # Load data
    try:
        df = pd.read_csv(f'{country_code}_youtube_trending_data.csv')
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {country_code}_youtube_trending_data.csv not found")

    # Load category mapping
    try:
        with open(f'{country_code}_category_id_to_name.json', 'r') as f:
            category_names = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Category mapping file {country_code}_category_id_to_name.json not found")

    # Convert categoryId to int for comparison
    df['categoryId'] = df['categoryId'].astype(int)

    # Filter out invalid categoryIds (those not in category_names)
    valid_category_ids = set(map(int, category_names.keys()))
    df = df[df['categoryId'].isin(valid_category_ids)]

    # Now proceed with tag and keyword analysis
    category_names = {str(k): v for k, v in category_names.items()}

    # Initialize columns
    df['tags_processed'] = ''
    df['title_keywords'] = ''
    df['keyword_sentiment'] = 0

    # Optimize tag processing
    def process_tags(tag_str):
        if pd.isna(tag_str) or tag_str == '[none]':
            return []
        tags = [tag.strip().lower() for tag in tag_str.replace('"', '').split('|') if tag.strip()]
        return [unidecode.unidecode(tag) for tag in tags]  # Normalize characters

    df['tags_processed'] = df['tags'].apply(process_tags)

    # Tag counting
    all_tags = df['tags_processed'].explode().dropna()
    tag_counts = Counter(all_tags)
    top_n = 30
    top_tags = tag_counts.most_common(top_n)
    tags_df = pd.DataFrame(top_tags, columns=['Tag', 'Count'])
    tags_df['Percentage'] = (tags_df['Count'] / len(df) * 100).round(2)

    # Display tag results
    print(f"=== Top {top_n} Most Popular Tags Across All Categories ({country_code}) ===")
    print(tabulate(tags_df, headers='keys', tablefmt='pretty', showindex=False))

    # Analyze tags by category
    print(f"\n=== Tag Popularity by Category ({country_code}) ===")
    category_tags_list = []
    for category_id, category_name in category_names.items():
        category_videos = df[df['categoryId'] == int(category_id)]
        cat_tags = category_videos['tags_processed'].explode().dropna()
        cat_tags = cat_tags[cat_tags != '[none]']
        cat_tag_counts = Counter(cat_tags).most_common(5)
        for tag, count in cat_tag_counts:
            if tag != '[none]':
                category_tags_list.append({'Category': category_name, 'Tag': tag, 'Count': count})

    category_tags_df = pd.DataFrame(category_tags_list)
    print(tabulate(category_tags_df, headers='keys', tablefmt='pretty', showindex=False))
    category_tags_df.to_csv(f'{output_dir}/{country_code}_most_popular_tags_by_category.csv', index=False)

    # Process titles with spaCy
    if nlp:
        def extract_keywords(titles):
            docs = list(nlp.pipe(titles, batch_size=1000))
            keywords = []
            for doc in docs:
                words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
                if country_code == 'MX':
                    words = [w for w in words if w not in stopwords_es]
                keywords.append(','.join(words))
            return keywords

        # Keyword extraction by category
        keywords_by_category = []
        for category_id, category_name in category_names.items():
            category_videos = df[df['categoryId'] == int(category_id)].copy()
            valid_titles = category_videos['title'].dropna().replace('', np.nan).dropna()
            if not valid_titles.empty:
                keywords = extract_keywords(valid_titles)
                category_videos.loc[valid_titles.index, 'title_keywords'] = keywords
                df.loc[category_videos.index, 'title_keywords'] = category_videos['title_keywords']

                # Keyword counting
                all_keywords = category_videos['title_keywords'].str.split(',').explode().dropna()
                all_keywords = all_keywords[all_keywords != '']
                if not all_keywords.empty:
                    keyword_counts = Counter(all_keywords).most_common(10)  # 10 keywords per category
                    keywords_df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Count'])
                    keywords_df['Category'] = category_name
                    keywords_by_category.append(keywords_df)
                else:
                    print(f"⚠️ No keywords extracted for category {category_name} ({country_code})")

        if keywords_by_category:
            keywords_all_df = pd.concat(keywords_by_category, ignore_index=True)
            keywords_all_df.to_csv(f'{output_dir}/{country_code}_top_title_keywords_by_category.csv', index=False)
            print(f"\n=== Top 10 Title Keywords by Category ({country_code}) ===")
            print(tabulate(keywords_all_df, headers='keys', tablefmt='pretty', showindex=False))
        else:
            print(f"⚠️ No title keywords extracted for {country_code}")

        # Calculate keyword sentiment
        df['title_keywords'] = df['title_keywords'].fillna('')
        df['keyword_sentiment'] = df['title_keywords'].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)

        # Analyze sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['keyword_sentiment'], bins=20, kde=True)
        plt.title(f'Distribution of Title Keyword Sentiment ({country_code})')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Count')
        plt.savefig(f'{output_dir}/{country_code}_keyword_sentiment.png')
        plt.close()

        # Global keyword counting
        all_keywords = df['title_keywords'].str.split(',').explode().dropna()
        all_keywords = all_keywords[all_keywords != '']
        if not all_keywords.empty:
            keyword_counts = Counter(all_keywords).most_common(20)
            keywords_df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Count'])
            keywords_df.to_csv(f'{output_dir}/{country_code}_top_title_keywords.csv', index=False)
            print(f"\n=== Top 20 Title Keywords (Global) ({country_code}) ===")
            print(tabulate(keywords_df, headers='keys', tablefmt='pretty', showindex=False))
        else:
            print(f"⚠️ No title keywords extracted for {country_code}")
    else:
        print(f"⚠️ Skipping title keyword extraction due to missing spaCy model")
        df['title_keywords'] = ''
        df['keyword_sentiment'] = 0

    print(f"\n✅ Saved results to '{output_dir}/{country_code}_most_popular_tags_by_category.csv'")
    if nlp and 'title_keywords' in df.columns and df['title_keywords'].str.len().sum() > 0:
        print(f"✅ Saved title keywords to '{output_dir}/{country_code}_top_title_keywords_by_category.csv'")
        print(f"✅ Saved global title keywords to '{output_dir}/{country_code}_top_title_keywords.csv'")
    print(f"✅ Saved sentiment plot to '{output_dir}/{country_code}_keyword_sentiment.png'")

    return category_tags_df

if __name__ == "__main__":
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    for country in countries:
        analyze_tags(country)