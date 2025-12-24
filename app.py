
################################################

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from recommender import ProductRecommender

recommender = ProductRecommender(r'C:\Users\Anjali\Downloads\AP final\AP\data\shopping_trends.csv')
df = recommender.df

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'explanation_df' not in st.session_state:
    st.session_state.explanation_df = None

st.set_page_config(page_title="🛍️ Product Recommendation System", layout="wide")
st.title('🛍️ Trendify : A Product Recommendation System')

st.sidebar.header('🔎 Filter Preferences')

with st.sidebar.container():
    category = st.selectbox('Select Category', ['All'] + sorted(df['Category'].unique()))
    gender = st.selectbox('Select Gender', ['All'] + sorted(df['Gender'].unique()))
    age_group = st.selectbox('Select Age Group', ['All'] + sorted(df['Age Group'].dropna().unique()))
    price_range = st.slider('Select Price Range', int(df['Price'].min()), int(df['Price'].max()),
                            (int(df['Price'].min()), int(df['Price'].max())))
    season = st.selectbox('Select Season', ['All'] + sorted(df['Season'].unique()))
    rating = st.slider('Minimum Review Rating', 0.0, 5.0, 0.0)
    top_n = st.slider('Number of Recommendations', 1, 20, 5)

st.sidebar.markdown("""---""")
recommend_button = st.sidebar.button('🚀 Recommend')

st.sidebar.markdown("---")
st.sidebar.header("🔥 Trending Picks")

selected_season = st.sidebar.selectbox('Choose a Season', ['Summer', 'Winter', 'Fall','Spring'])
selected_gender = st.sidebar.selectbox('Choose a Gender', ['Male', 'Female'])
top_n_trending = st.sidebar.slider('Top N Products', 1, 10, 5)

if st.sidebar.button('Show Trending Picks'):
    recommender.top_trending_products(
        season=selected_season,
        gender=selected_gender,
        top_n=top_n_trending
    )

if recommend_button:
    filtered_df = df.copy()

    if category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category]
    if gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == gender]
    if age_group != 'All':
        filtered_df = filtered_df[filtered_df['Age Group'] == age_group]
    if season != 'All':
        filtered_df = filtered_df[filtered_df['Season'] == season]

    filtered_df = filtered_df[
        (filtered_df['Price'] >= price_range[0]) &
        (filtered_df['Price'] <= price_range[1]) &
        (filtered_df['Review Rating'] >= rating)
    ]

    if filtered_df.empty:
        st.error('No products found based on selected criteria.')
    else:
        dummy_user_id = 1
        recommendations, explanation_df = recommender.hybrid_recommendations(dummy_user_id, filtered_df, top_n=top_n)

        st.session_state.recommendations = recommendations
        st.session_state.explanation_df = explanation_df

if st.session_state.recommendations is not None:
    st.success('✅ Here are your Recommended Products:')
    st.dataframe(st.session_state.recommendations.reset_index(drop=True))

#     st.markdown("### 🔍 Why These Recommendations?")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.barplot(x='Item Purchased', y='Final Score', data=st.session_state.explanation_df, palette='viridis', ax=ax)
#     ax.set_title('Final Recommendation Scores')
#     plt.xticks(rotation=45)
#     st.pyplot(fig)

#     with st.expander("See detailed recommendation explanation"):
#         st.dataframe(st.session_state.explanation_df)

if st.button('Show Similarity Heatmap'):
    if st.session_state.recommendations is not None:
        recommender.plot_similarity_heatmap(top_n=top_n)
    else:
        st.warning('⚠️ Please generate recommendations first by clicking "Recommend"')





