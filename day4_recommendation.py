import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("cleaned_products.csv")
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(product_name, num_recommendations=5):
    if product_name not in df['product_name'].values:
        return "Product not found!"

    idx = df[df['product_name'] == product_name].index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    # Exclude the product itself
    similarity_scores = similarity_scores[1:]

    # Multiply similarity with rating
    weighted_scores = []
    for i, score in similarity_scores:
        rating = df.iloc[i]['rating']
        weighted_scores.append((i, score * rating))

    # Sort by weighted score
    weighted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

    top_products = weighted_scores[:num_recommendations]

    return [df.iloc[i[0]]['product_name'] for i in top_products]

# TEST
product = "Apple iPhone 13"
print("Because you viewed:", product)
print("Recommended products:")
print(recommend(product))
