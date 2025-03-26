import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import classification_report


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
    ax.set_title(f"Top {top_k} Brands by Review Count (Raw vs Cleaned)")
    ax.set_xlabel("Number of Reviews")
    ax.invert_yaxis()  
    plt.tight_layout()

    return fig1