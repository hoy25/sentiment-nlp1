import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns

def set_clean_style(ax, title=None, xlabel=None, ylabel=None, fontsize=12, grid=True):

    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    if grid:
        ax.grid(True, linestyle='--', alpha=0.5)

    ax.tick_params(axis='both', labelsize=fontsize - 2)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))

def apply_style_hist(ax, color='skyblue', alpha=0.85, edgecolor='black'):

    for patch in ax.patches:
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor(edgecolor)

def apply_sns_style():

    sns.set_style("whitegrid")
    sns.set_palette("pastel")
    sns.set_context("notebook", font_scale=1.1)

def plot_top_raw_brands(df_raw, brand_col='Brand Name', top_k=15):
    raw_counts = df_raw[brand_col].value_counts().head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))
    raw_counts.sort_values().plot(kind='barh', ax=ax, color='orangered')
    ax.set_title(f"Raw Top {top_k} brand (unnormalized)")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Raw Brand Name")
    plt.tight_layout()
    return fig


def plot_top_cleaned_brands(df_cleaned, brand_col='Brand Name', top_k=15):
    cleaned_counts = df_cleaned[brand_col].value_counts().head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))
    cleaned_counts.sort_values().plot(kind='barh', ax=ax, color='seagreen')
    ax.set_title(f"Cleaned Top {top_k} brands (normalized)")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Cleaned Brand Name")
    plt.tight_layout()
    return fig

def compare_brand_cleaning(df_raw, df_cleaned, brand_col='Brand Name',top_k=15):
    raw_counts = df_raw['Brand Name'].value_counts().rename('Raw Reviews')
    cleaned_counts = df_cleaned['Brand Name'].value_counts().rename('Cleaned Reviews')
    merged = pd.concat([raw_counts, cleaned_counts], axis=1).fillna(0).astype(int)
    merged['Total'] = merged['Raw Reviews'] + merged['Cleaned Reviews']
    top_brands = merged.sort_values(by='Total', ascending=False).head(top_k)

    #merge compare his
    fig1, ax = plt.subplots(figsize=(10, 8))
    top_brands[['Raw Reviews', 'Cleaned Reviews']].plot(kind='barh', ax=ax)
    ax.set_title(f"Top {top_k} Brands by Reviews Count (Raw vs Cleaned)")
    ax.set_xlabel("Number of Reviews")
    ax.invert_yaxis()  
    plt.tight_layout()

    return fig1

## product name for each brand
def k_mean_cluster_product(df, brand_col='Brand Name', product_col='Product Name', n_clusters=3):
    results = []

    for brand in df[brand_col].unique():
        brand_df = df[df[brand_col] == brand].copy()
        product_names = brand_df[product_col].dropna().astype(str).tolist()

        if len(product_names) < n_clusters:
            continue 

        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        X = vectorizer.fit_transform(product_names)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        brand_df = brand_df.loc[brand_df[product_col].notna()].copy()
        brand_df['Cluster'] = labels
        results.append(brand_df)

    if results:
        return pd.concat(results)
    else:
        return pd.DataFrame()
    
def plot_price_hist(df,selected_brand):
    fig, ax = plt.subplots(figsize=(12, 6))
    brand_df = df[df["Brand Name"]== selected_brand]
    brand_df['Price'].dropna().hist(bins=30, ax=ax)
    ax.set_title("Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    return fig

def plot_rating_hist(df,selected_brand):
    fig, ax = plt.subplots(figsize=(12, 6))
    brand_df = df[df["Brand Name"]== selected_brand]
    df['Rating'].dropna().hist(bins=20, ax=ax, color='skyblue')
    ax.set_title("Rating Distribution")
    return fig

def plot_price_vs_rating_scatter(df,price_col='Price', rating_col='Rating'):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind='scatter', x=price_col, y=rating_col, alpha=0.4, ax=ax)
    ax.set_title("Price vs Rating")
    ax.set_xlabel("Price")
    ax.set_ylabel("Rating")
    plt.tight_layout()
    return fig

def plot_price_box_by_brand(df, brand_col='Brand Name', price_col='Price', top_n=10):
    fig, ax = plt.subplots(figsize=(12, 6))
    top_brands = df[brand_col].value_counts().nlargest(top_n).index
    sns.boxplot(data=df[df[brand_col].isin(top_brands)],
                x=brand_col, y=price_col, ax=ax)
    ax.set_title("Price Distribution by Top Brands")
    ax.set_xlabel("Brand")
    ax.set_ylabel("Price")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    return fig

def plot_rating_box_by_brand(df,brand_col='Brand Name', rating_col='Rating', top_n=10):
    fig, ax = plt.subplots(figsize=(12, 6))
    top_brands = df[brand_col].value_counts().nlargest(top_n).index
    sns.boxplot(data=df[df[brand_col].isin(top_brands)],
                x=brand_col, y=rating_col, ax=ax)
    ax.set_title("Rating Distribution by Top Brands")
    ax.set_xlabel("Brand")
    ax.set_ylabel("Rating")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    return fig



def review_length(df_cleaned, quantile=0.99, bins=50):
    df_cleaned['review_length'] = df_cleaned['Reviews'].dropna().astype(str).apply(lambda x: len(x.split()))
    max_val = df_cleaned['review_length'].quantile(quantile)

    fig, ax = plt.subplots(figsize=(12, 6))
    df_cleaned[df_cleaned['review_length'] <= max_val]['review_length'].hist(
        bins=bins,
        ax=ax,
        color='skyblue',
        edgecolor='black',
        alpha=0.85
    )

    ax.set_title("Review Length Distribution (filtered)", fontsize=16)
    ax.set_xlabel("Number of Words", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=10)

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))

    plt.tight_layout()
    return fig

def review_WordCloud(df_cleaned):
    text = " ".join(df_cleaned['Reviews'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords='english').generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)

def rate_reviews(df_cleaned):
    fig, ax = plt.subplots(figsize=(12, 6))
    df_cleaned['Rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
    set_clean_style(ax, title = "Number of Reviews per Rating Level",xlabel="Rating", ylabel="Number of Reviews")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    apply_style_hist(ax)

def product_reviews(df_cleaned):
    top_products = df_cleaned['Product Name'].value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_cleaned[df_cleaned['Product Name'].isin(top_products)],
                x='Product Name', y='Price', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    set_clean_style(ax,title="Product Price",xlabel="Product Name",ylabel='Price')
