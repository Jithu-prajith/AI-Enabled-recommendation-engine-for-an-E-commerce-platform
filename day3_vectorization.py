import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv("cleaned_products.csv")

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(tfidf_matrix, open("tfidf_matrix.pkl", "wb"))

print("Vectorizer and matrix saved successfully!")