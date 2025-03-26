import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#import functions from eda
from eda import (compare_brand_cleaning,
                 plot_top_raw_brands,
                 plot_top_cleaned_brands,
                 k_mean_cluster_product,
                   plot_price_hist, plot_rating_hist, 
                   plot_price_vs_rating_scatter, 
                   plot_price_box_by_brand,
                   plot_rating_box_by_brand,
                   review_length,
                   review_WordCloud,
                   rate_reviews,
                   product_reviews
                   )

st.set_page_config(layout="wide")
page = st.sidebar.radio("Pages", ["Data Cleaning", "EDA on Cleaned Data","Reviews' EDA"])
if page == "Data Cleaning":
    st.title("Visualization Dashboard")

    st.header("Data Cleaning")
    #upload
    raw_file = st.file_uploader("up load the original dataset", type="csv", key="raw")
    cleaned_file = st.file_uploader("up load normalized dataset", type="csv", key="cleaned")

    if raw_file and cleaned_file:
        df_raw = pd.read_csv(raw_file)
        df_cleaned = pd.read_csv(cleaned_file)

        st.session_state.df_raw = df_raw
        st.session_state.df_cleaned = df_cleaned
        #### raw data
        st.subheader("Raw data")
        st.dataframe(df_raw.head(),use_container_width=True)
        st.markdown("### Missing value: raw data")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if not missing.empty:
            st.dataframe(missing.to_frame(name="Missing Count"))
        else:
            st.success("There is no missing value")
        #### explain data cleaning 
        st.markdown("---")
        st.markdown("### Data Cleaning: Brand and product normalization strategy")
        st.markdown("""
    To improve data quality and enhance the value of analysis, applied the following cleaning and normalization steps to the brand and product name columns:

    1. **Brand Normalization**  
    - All brand names were converted to lowercase and stripped of extra spaces;  
    - Ranked brands by number of reviews and selected the top 13 as *Known Brands*;  
    - Other brands were matched using fuzzy string matching based on known brand keywords;  
    - Brands that did not match or had a low number (default < 500)of reviews were grouped under the `others` category.

    2. **Product Name Standardization**  
    - For known brands, used **regular expressions** to extract core product models (e.g., *galaxy s21*, *iphone 13*);  
    - For unknown or `others` brands, extracted the first few keywords to form a simplified product name;  
    - A new column `Product Name (core)` created for downstream tasks such as rating analysis, pricing comparison, and sentiment classification.

    These steps significantly reduced inconsistencies in brand and product naming, making the dataset more structured and suitable for further analysis. ✅
    """)
        #########compare
        st.markdown("### Top brands")
        top_k = st.slider("Choose top brands", min_value=5, max_value=30, value=15, step=1)

        fig = compare_brand_cleaning(df_raw, df_cleaned, brand_col='Brand Name', top_k=15)

        if st.button("reviews of raw brand"):
            st.markdown("### raw brand reviews distribution")
            fig1 = plot_top_raw_brands(df_raw, top_k=top_k)
            st.pyplot(fig1)
            st.markdown("""
        ## From the plot, 
        - There are a large number of similar/duplicate/niche brands in the original brand, with confusing naming
        - After cleaning, the brand structure is clearer, which is conducive to analysis, modeling and visualization \n
        ## Top.14 brand: LG Electronics is actually Top.4 brand: LG.
        - Make top 13 brands as known brand list to classify similar/duplicate/niche brands into the original brand
            """)

        #### cleaned data 
        st.subheader("Cleaned data")
        st.dataframe(df_cleaned.head(),use_container_width=True)
        st.markdown("### Missing value: cleaned data")
        missing = df_cleaned.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if not missing.empty:
            st.dataframe(missing.to_frame(name="Missing Count"))
        else:
            st.success("There is no missing value")
            
        if st.button("reviews of normalized brand"):
            st.markdown("### normalized reviews distribution")
            fig2 = plot_top_cleaned_brands(df_cleaned, top_k=top_k)
            st.pyplot(fig2)

        ##merge
        st.subheader("Compare the raw data and cleaned data")
        if st.button("merged reviews distribution"):
            fig = compare_brand_cleaning(df_raw, df_cleaned, brand_col='Brand Name', top_k=top_k)
            st.pyplot(fig, use_container_width=True)
        
    else:
        st.info("upload raw & normalized data")

elif page == "EDA on Cleaned Data":
    st.title("General Exploratory Data Analysis")
    if "df_cleaned" in st.session_state:
        df_cleaned = st.session_state.df_cleaned
        #### show product under each brand
        st.header("Product Distribution")
        st.markdown("Product name is classified by regularization. \n"
        "For more accurate identification, may need web script from gsmarena dataset.\n"
        "For analyzing product name, I can also use FuzzyWuzzy to keep popular brands which contains reviews more than 30.")
        brand_list = sorted(df_cleaned['Brand Name'].unique())
        selected_brand = st.selectbox("Choose Brand", brand_list)
        brand_df = df_cleaned[df_cleaned['Brand Name'] == selected_brand]

        product_counts = brand_df['Product Name'].value_counts().sort_values(ascending=False)

        st.write(f"** Brand `{selected_brand}` have {len(product_counts)} product(s))**")

        if not product_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            product_counts.plot(kind='bar', ax=ax)
            ax.set_title(f"Product Distribution - {selected_brand}")
            ax.set_xlabel("Product Name")
            ax.set_ylabel("Quantity")
            st.pyplot(fig, use_container_width=True)

            if st.checkbox("Show Price Distribution"):
                fig = plot_price_hist(df_cleaned,selected_brand)
                st.pyplot(fig, use_container_width=True)

            if st.checkbox("Show Rating Distribution"):
                fig = plot_rating_hist(df_cleaned,selected_brand)
                st.pyplot(fig, use_container_width=True)

        
        else:
            st.warning("This brand has no product recorded")

        st.header("Price vs Rating Scatter")
        fig = plot_price_vs_rating_scatter(df_cleaned)
        st.pyplot(fig, use_container_width=True)
        st.header("Price Boxplot by Brand")
        fig = plot_price_box_by_brand(df_cleaned)
        st.pyplot(fig, use_container_width=True)
        st.header("Rating Boxplot by Brand")
        fig = plot_rating_box_by_brand(df_cleaned)
        st.pyplot(fig, use_container_width=True)
        
    else:
        st.warning("upload cleaned dataset")


elif page == "Reviews' EDA":
    st.title("Reviews Exploratory Data Analysis")
    if "df_cleaned" in st.session_state:
        df_cleaned = st.session_state.df_cleaned
        if st.button("Reviews length distribution"):
            st.markdown("### Reviews length distribution")
            fig = review_length(df_cleaned)
            st.pyplot(fig, use_container_width=True)

        if st.button("Reviews wordCloud"):
            st.markdown("### Reviews wordCloud")
            fig = review_WordCloud(df_cleaned)
            st.pyplot(fig, use_container_width=True)
        if st.button("Rate vs Reviews"):
            st.markdown("### Rate vs Reviews")
            fig = rate_reviews(df_cleaned)
            st.pyplot(fig, use_container_width=True)
        if st.button("Product's Review distribution"):
            st.markdown("### Rate vs Reviews")
            fig = product_reviews(df_cleaned)
            st.pyplot(fig, use_container_width=True)
        

    else:
        st.warning("upload cleaned dataset")