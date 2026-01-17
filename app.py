import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("cleaned_products.csv")
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

# Similarity calculation
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(product_name, num_recommendations=5):
    idx = df[df['product_name'] == product_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))[1:]

    weighted_scores = []
    for i, score in scores:
        rating = df.iloc[i]['rating']
        weighted_scores.append((i, score * rating))

    weighted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    top = weighted_scores[:num_recommendations]

    return [df.iloc[i[0]]['product_name'] for i in top]

# ---------------- UI ----------------
st.title("🛒 AI-Enabled Recommendation Engine")

st.write("Select a product to get AI-based recommendations")

product = st.selectbox("Choose a product", df['product_name'])

if st.button("Get Recommendations"):
    recommendations = recommend(product)
    st.subheader("Recommended Products:")
    for r in recommendations:
        st.write("👉", r)