import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time

# ================= LOAD DATA =================
df = pd.read_csv("cleaned_products.csv")
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ================= UNIQUE IMAGES =================
product_images = {
    "Apple iPhone 13": "https://images.unsplash.com/photo-1632661674596-df8be070a5c5",
    "Samsung Galaxy S21": "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf",
    "OnePlus 9": "https://images.unsplash.com/photo-1620799139834-6b8f844fbe61",
    "Redmi Note 12": "https://images.unsplash.com/photo-1603899123005-19c9f56f14b2",
    "MacBook Air M1": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8",
    "Dell Inspiron 15": "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed",
    "HP Pavilion": "https://images.unsplash.com/photo-1611078489935-0cb964de46d6",
    "Boat Rockerz 450": "https://images.unsplash.com/photo-1585386959984-a4155224a1ad",
    "Sony WH-1000XM4": "https://images.unsplash.com/photo-1546435770-a3e426bf472b",
    "Logitech Wireless Mouse": "https://images.unsplash.com/photo-1586816001966-79b736744398",
    "Apple Watch Series 8": "https://images.unsplash.com/photo-1551817958-20204d6ab8df",
    "Mi Band 7": "https://images.unsplash.com/photo-1629992101753-56d196c8aabb",
    "Nike Running Shoes": "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
    "Adidas Sports Shoes": "https://images.unsplash.com/photo-1600180758895-7c1f1b41c61b",
    "Puma Casual Sneakers": "https://images.unsplash.com/photo-1528701800489-20be3c8f1f95"
}

def get_image(name):
    return product_images.get(name, "https://images.unsplash.com/photo-1523275335684-37898b6baf30")

# ================= RECOMMEND =================
def recommend(product, top_n=5):
    idx = df[df['product_name'] == product].index[0]
    scores = list(enumerate(cosine_sim[idx]))[1:]

    weighted = []
    for i, score in scores:
        weighted.append((i, score * df.iloc[i]['rating']))

    weighted.sort(key=lambda x: x[1], reverse=True)
    return [df.iloc[i[0]]['product_name'] for i in weighted[:top_n]]

# ================= EVALUATION =================
def evaluate_model(product, recs):
    category = df[df['product_name'] == product]['category'].values[0]
    relevant = df[df['category'] == category]['product_name'].tolist()

    if product in relevant:
        relevant.remove(product)

    tp = len(set(recs) & set(relevant))

    precision = tp / len(recs) if recs else 0
    recall = tp / len(relevant) if relevant else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

# ================= PAGE CONFIG =================
st.set_page_config("AI Recommendation Engine", "🛒", layout="wide")

# ================= STYLES =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #2b0f3f, #3c1361);
}
h1 { color: #f5c16c; text-align: center; }
h2 { color: #f1e6d2; }

.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 28px;
    margin-top: 20px;
}

.product-card {
    background: #f8f5ef;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    transition: all 0.4s ease;
}
.product-card:hover {
    transform: translateY(-10px) scale(1.04);
}

.product-card img {
    width: 100%;
    height: 170px;
    object-fit: cover;
    transition: transform 0.5s ease;
}
.product-card:hover img {
    transform: scale(1.1);
}

.product-body {
    padding: 16px;
}
.product-title {
    font-size: 17px;
    font-weight: 700;
    color: #2b0f3f;
}
.product-meta {
    font-size: 14px;
    color: #555;
}
.price {
    font-weight: bold;
    color: #7b2cbf;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
}
.metric-card h3, .metric-card h2 {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ================= UI =================
st.title("🛒 AI-Enabled Recommendation Engine")
product = st.selectbox("🔍 Choose a Product", df['product_name'])

if st.button("🚀 Get Recommendations"):
    with st.spinner("Analyzing products..."):
        time.sleep(0.8)
        recommendations = recommend(product)

    st.markdown("## 🔥 Customers Also Viewed")
    st.markdown('<div class="product-grid">', unsafe_allow_html=True)

    for item in recommendations:
        row = df[df['product_name'] == item].iloc[0]
        st.markdown(f"""
        <div class="product-card">
            <img src="{get_image(item)}">
            <div class="product-body">
                <div class="product-title">{item}</div>
                <div class="product-meta">⭐ {row['rating']} | {row['category']}</div>
                <div class="price">₹ {row['price']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    precision, recall, f1 = evaluate_model(product, recommendations)

    st.markdown("## 📊 Recommendation Quality")
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='metric-card'><h3>Precision</h3><h2>{precision:.2f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h3>Recall</h3><h2>{recall:.2f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h3>F1-Score</h3><h2>{f1:.2f}</h2></div>", unsafe_allow_html=True)
