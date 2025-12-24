


#############################################################
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

class ProductRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        self.train_models()

    def prepare_data(self):
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        self.df = self.df.dropna(subset=['Price'])

        self.df['Age Group'] = pd.cut(
            self.df['Age'],
            bins=[0, 17, 25, 35, 50, 100],
            labels=['0-17', '18-25', '26-35', '36-50', '51+']
        )

        self.df['content_features'] = (
            self.df['Category'].astype(str) + ' ' +
            self.df['Color'].astype(str) + ' ' +
            self.df['Season'].astype(str) + ' ' +
            self.df['Size'].astype(str) + ' ' +
            self.df['Gender'].astype(str)
        )

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['content_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        if {'Customer ID', 'Item Purchased', 'Review Rating'}.issubset(self.df.columns):
            reader = Reader(rating_scale=(1, 5))
            self.collab_data = Dataset.load_from_df(self.df[['Customer ID', 'Item Purchased', 'Review Rating']], reader)
        else:
            self.collab_data = None

    def train_models(self):
        if self.collab_data:
            trainset, _ = train_test_split(self.collab_data, test_size=0.2, random_state=42)
            self.svd = SVD()
            self.svd.fit(trainset)
        else:
            self.svd = None

    def top_trending_products(self, season=None, gender=None, top_n=5):
        filtered_df = self.df.copy()

        if season:
            filtered_df = filtered_df[filtered_df['Season'].str.lower() == season.lower()]
        if gender:
            filtered_df = filtered_df[filtered_df['Gender'].str.lower() == gender.lower()]

        if filtered_df.empty:
            st.warning(f"No products found for Season: {season} and Gender: {gender}")
            return

        top_items = filtered_df['Item Purchased'].value_counts().head(top_n)

        st.markdown(f"### 🛍️ Top {top_n} {season.title() if season else ''} Picks for {gender.title() if gender else ''}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_items.values, y=top_items.index, palette="coolwarm", ax=ax)
        ax.set_xlabel("Number of Purchases")
        ax.set_ylabel("Product")
        st.pyplot(fig)

        with st.expander("See raw list"):
            st.dataframe(top_items)

    def plot_similarity_heatmap(self, top_n=20):
        top_items = self.df.head(top_n)
        filtered_cosine_sim = self.cosine_sim[:top_n, :top_n]

        fig = go.Figure(data=go.Heatmap(
            z=filtered_cosine_sim,
            x=top_items['Item Purchased'],
            y=top_items['Item Purchased'],
            colorscale='Viridis',
            colorbar=dict(title="Cosine Similarity"),
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Cosine Similarity: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Top {top_n} Item Similarity Heatmap",
            xaxis_title="Item Purchased",
            yaxis_title="Item Purchased",
            xaxis=dict(tickangle=45, tickmode='array'),
            yaxis=dict(tickangle=0),
            autosize=True,
            margin=dict(l=100, r=100, b=100, t=100),
            template="plotly_dark"
        )

        st.plotly_chart(fig)

    def hybrid_recommendations(self, user_id, filtered_df, top_n=5):
        if filtered_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        indices = filtered_df.index.tolist()
        scores = []

        for idx in indices:
            row = filtered_df.loc[idx]
            item_id = row['Item Purchased']

            content_score = np.mean(self.cosine_sim[idx])

            if self.svd:
                try:
                    user_cf_score = self.svd.predict(user_id, item_id).est
                except:
                    user_cf_score = 0.0
            else:
                user_cf_score = 0.0

            # Item-based collaborative filtering score
            item_idx = self.df[self.df['Item Purchased'] == item_id].index
            if not item_idx.empty:
                sim_scores = self.cosine_sim[item_idx[0]]
                item_cf_score = np.mean(sim_scores)
            else:
                item_cf_score = 0.0

            final_score = 0.5 * content_score + 0.25 * user_cf_score + 0.25 * item_cf_score
            scores.append((idx, content_score, user_cf_score, item_cf_score, final_score))

        sorted_scores = sorted(scores, key=lambda x: x[4], reverse=True)
        top_indices = [x[0] for x in sorted_scores[:top_n]]

        recommendations = filtered_df.loc[top_indices][[
            'Item Purchased', 'Category', 'Price', 'Gender', 'Age Group', 'Season', 'Review Rating'
        ]].drop_duplicates()

        explanations = [{
            'Item Purchased': filtered_df.loc[idx]['Item Purchased'],
            'Content Score': round(content, 3),
            'User CF Score': round(user_cf, 3),
            'Item CF Score': round(item_cf, 3),
            'Final Score': round(final, 3)
        } for idx, content, user_cf, item_cf, final in sorted_scores[:top_n]]

        explanation_df = pd.DataFrame(explanations)

        return recommendations, explanation_df



