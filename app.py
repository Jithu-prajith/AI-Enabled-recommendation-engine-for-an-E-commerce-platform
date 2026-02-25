import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. LOAD DATA =================
@st.cache_data
def load_data():
    try:
        # Load the cleaned data and tfidf matrix
        df = pd.read_csv("cleaned_products_v2.csv")
        with open("tfidf_matrix_v2.pkl", "rb") as f:
            tfidf_matrix = pickle.load(f)
        return df, tfidf_matrix
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.stop()

df, tfidf_matrix = load_data()

# ================= 2. FUNCTIONS =================
def get_image(row):
    # Professional fallback if URL is missing
    url = row.get('image_url')
    if pd.isna(url) or not url or str(url) == "None":
        return "https://via.placeholder.com/400x400.png?text=Product+Image+Coming+Soon"
    return url

def recommend(product_name, top_n=8):
    try:
        idx = df[df['product_name'] == product_name].index[0]
        # Target specific sub-category to prevent "Laptop Bags" for "Laptops"
        base_cat_tree = df.iloc[idx]['category']
        specific_cat = base_cat_tree.split('>>')[-1].strip()
        
        scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
        scores = [(i, s) for i, s in scores if i != idx]
        
        weighted = []
        for i, score in scores:
            item_cat = df.iloc[i]['category']
            # Bonus for exact category match; penalty for semantic drift
            if specific_cat in item_cat:
                final_score = score + 2.0  
            else:
                final_score = score * 0.1  
                
            weighted.append((i, final_score))

        weighted.sort(key=lambda x: x[1], reverse=True)
        return [df.iloc[i[0]]['product_name'] for i in weighted[:top_n]]
    except Exception:
        return []

def evaluate_model(product, recs):
    # Performance metrics for Milestone 4
    if not recs: return None, None, None
    try:
        cat = df[df['product_name'] == product]['category'].values[0]
        relevant = df[df['category'] == cat]['product_name'].tolist()
        tp = len(set(recs) & set(relevant))
        p = tp / len(recs)
        r = tp / len(relevant) if len(relevant) > 0 else 0
        f1 = (2*p*r)/(p+r) if (p+r) else 0
        return p, r, f1
    except: return None, None, None

# ================= 3. UI CONFIG & CSS (BLACK TEXT ADJUSTMENTS) =================
st.set_page_config("AI Store", "🛒", layout="wide")

st.markdown("""
<style>
    /* Global Background and text color */
    [data-testid="stAppViewContainer"] { background-color: #f1f3f6; }
    [data-testid="stHeader"] { background-color: #2874f0; height: 60px; }
    
    /* Force all major text to Black */
    h1, h2, h3, h4, h5, p, span, div { color: #212121 !important; }
    
    .white-card { background: white; padding: 20px; border-radius: 4px; border: 1px solid #e0e0e0; margin-bottom: 20px; }
    
    .cat-nav { display: flex; justify-content: space-around; background: white; padding: 10px 0; margin-bottom: 10px; box-shadow: 0 1px 1px 0 rgba(0,0,0,.16); }
    .cat-item { text-align: center; font-size: 13px; font-weight: 600; color: #212121 !important; }
    
    .product-card { background: white; border-radius: 2px; padding: 15px; text-align: center; height: 400px; border: 1px solid #f0f0f0; display: flex; flex-direction: column; }
    .product-title { font-size: 14px; color: #000000 !important; height: 40px; overflow: hidden; font-weight: 500;}
    .product-price { font-size: 18px; font-weight: 700; color: #000000 !important; }
    
    /* Detail View Styling */
    .detail-price { font-size: 28px; font-weight: 700; color: #2874f0 !important; margin: 10px 0; }
    .detail-desc { font-size: 15px; color: #212121 !important; line-height: 1.6; }

    /* Search Bar text color fix */
    .stTextInput input { color: #000000 !important; background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ================= 4. NAVIGATION & LOGIC =================

# Initialize session state for tracking selected products
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

page = st.sidebar.radio("Navigation", ["Home", "Recommend", "Explore More"])

# Reset selection when switching main pages
def reset_selection():
    st.session_state.selected_product = None

# --- HOME PAGE LOGIC ---
if page == "Home":
    # If a product is selected, show its details instead of the grid
    if st.session_state.selected_product:
        row = df[df['product_name'] == st.session_state.selected_product].iloc[0]
        
        if st.button("← Back to Home"):
            st.session_state.selected_product = None
            st.rerun()

        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(get_image(row), use_container_width=True)
        with col2:
            st.markdown(f"## {row['product_name']}")
            st.markdown(f"<div class='detail-price'>₹{row['price']}</div>", unsafe_allow_html=True)
            st.write("---")
            st.markdown("### Product Description")
            st.markdown(f"<div class='detail-desc'>{row['description'][:1000]}...</div>", unsafe_allow_html=True)
            st.button("⚡ BUY NOW", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Also show recommendations for this clicked item
        recs = recommend(row['product_name'])
        if recs:
            st.markdown("### 🔥 Similar Products")
            r_cols = st.columns(4)
            # Use the loop index 'i' to ensure every button has a unique ID
            for i, name in enumerate(recs[:4]):
                r_data = df[df['product_name'] == name].iloc[0]
                with r_cols[i]:
                    # UNIQUE KEY FIX: Include the index 'i' in the key string
                    if st.button(f"View Details", key=f"rec_{name}_{i}"):
                        st.session_state.selected_product = name
                        st.rerun()
                    st.markdown(f"""
                        <div class="product-card">
                            <img src="{get_image(r_data)}" style="height:150px; object-fit:contain;">
                            <div class="product-title" style="color:black !important;">{name[:40]}...</div>
                            <div class="product-price" style="color:black !important;">₹{r_data['price']}</div>
                        </div>
                    """, unsafe_allow_html=True)

    else:
        # Standard Home View: Categorized Rows
        st.markdown("""
            <div style="background-color: #2874f0; padding: 25px; margin: -50px -50px 20px -50px; text-align: center;">
                <h2 style="color: white !important; margin: 0;">🏠 Welcome to AI Store</h2>
            </div>
        """, unsafe_allow_html=True)

        df['top_category'] = df['category'].apply(lambda x: x.split('>>')[0].strip())
        categories = df['top_category'].unique()[:5] # Limit to top 5 categories for performance

        for cat in categories:
            st.markdown(f"<h3 style='color:black !important;'>{cat}</h3>", unsafe_allow_html=True)
            cat_items = df[df['top_category'] == cat].head(4) 
            cols = st.columns(4)
            
            # Use enumerate to get a unique number (idx) for each iteration
            for i, (original_idx, item_row) in enumerate(cat_items.iterrows()):
                with cols[i]:
                    # FIX: Added original_idx to the key to make it 100% unique
                    button_key = f"home_{cat}_{item_row['product_name']}_{original_idx}"
                    
                    if st.button("View Product", key=button_key):
                        st.session_state.selected_product = item_row['product_name']
                        st.rerun()
                    
                    st.markdown(f"""
                        <div class="product-card">
                            <div style="height:160px; display:flex; justify-content:center; align-items:center;">
                                <img src="{get_image(item_row)}" style="max-height:100%; max-width:100%; object-fit:contain;">
                            </div>
                            <div class="product-title" style="color:black !important;">{item_row['product_name'][:50]}...</div>
                            <div class="product-price" style="color:black !important;">₹{item_row['price']}</div>
                        </div>
                    """, unsafe_allow_html=True)
            st.write("---")

elif page == "Recommend":
    reset_selection()
    # ... (Your existing Recommend search logic)
    # ... (Keep your existing Recommend logic here)
    
    query = st.text_input("", placeholder="Search for Products...", label_visibility="collapsed")

    # Category Nav Bar
    st.markdown("""
    <div class="cat-nav">
        <div class="cat-item"><img src="https://rukminim1.flixcart.com/flap/128/128/image/69c60546e101711c.png"><br>Electronics</div>
        <div class="cat-item"><img src="https://rukminim1.flixcart.com/flap/128/128/image/22fddf3c7da4c4f4.png"><br>Mobiles</div>
        <div class="cat-item"><img src="https://rukminim1.flixcart.com/flap/128/128/image/ab7e2b022145a504.png"><br>Fashion</div>
        <div class="cat-item"><img src="https://rukminim1.flixcart.com/flap/128/128/image/0ff199d1bd27eb98.png"><br>Appliances</div>
    </div>
    """, unsafe_allow_html=True)

    if query:
        # Prioritize category match to fix the "Laptop Bag" issue
        matches = df[df['product_name'].str.contains(query, case=False, na=False)].copy()
        matches['cat_match'] = matches['category'].str.contains(query, case=False, na=False)
        matches = matches.sort_values(by='cat_match', ascending=False)
        
        if matches.empty:
            st.warning("No matching product found")
        else:
            row = matches.iloc[0]
            product_name = row['product_name']
            
            # Display Detailing
            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.image(get_image(row), use_container_width=True)
            with col2:
                st.markdown(f"## {product_name}")
                st.markdown(f"<div class='detail-price'>₹{row['price']}</div>", unsafe_allow_html=True)
                st.write("---")
                st.markdown("### Product Description")
                st.markdown(f"<div class='detail-desc'>{row['description'][:1000]}...</div>", unsafe_allow_html=True)
                st.write("")
                st.button("⚡ BUY NOW", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            recs = recommend(product_name)
            if recs:
                st.markdown("### 🔥 Similar Products You Might Like")
                rec_cols = st.columns(4)
                for i, name in enumerate(recs):
                    r_data = df[df['product_name'] == name].iloc[0]
                    with rec_cols[i % 4]:
                        st.markdown(f"""
                            <div class="product-card">
                                <div style="height:180px; display:flex; justify-content:center; align-items:center;">
                                    <img src="{get_image(r_data)}" style="max-height:100%; max-width:100%; object-fit:contain;">
                                </div>
                                <div class="product-title">{name[:50]}...</div>
                                <div style="background:#388e3c; color:white !important; padding:2px 6px; border-radius:3px; font-size:12px; width:fit-content; margin:0 auto 8px;">★ {r_data.get('rating', '4.1')}</div>
                                <div class="product-price">₹{r_data['price']}</div>
                                <div style="background:#fb641b; color:white !important; text-align:center; padding:8px; border-radius:2px; font-weight:600; margin-top:auto;">View Item</div>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Metrics for Milestone 4
            st.write("---")
            st.markdown("### 📊 Evaluation Metrics")
            p, r, f = evaluate_model(product_name, recs)
            if p is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{p:.2f}")
                m2.metric("Recall", f"{r:.2f}")
                m3.metric("F1-Score", f"{f:.2f}")

else:
    # Explore More logic
    st.title("🛍 Explore Trending Collection")
    sample = df.sample(12)
    exp_cols = st.columns(4)
    for i, (idx, r) in enumerate(sample.iterrows()):
        with exp_cols[i % 4]:
            st.markdown(f"""
                <div class="product-card">
                    <div style="height:180px; display:flex; justify-content:center; align-items:center;">
                        <img src="{get_image(r)}" style="max-height:100%; max-width:100%; object-fit:contain;">
                    </div>
                    <div class="product-title">{r['product_name'][:50]}...</div>
                    <div style="background:#388e3c; color:white !important; padding:2px 6px; border-radius:3px; font-size:12px; width:fit-content; margin:0 auto 8px;">★ {r.get('rating', '4.0')}</div>
                    <div class="product-price">₹{r['price']}</div>
                    <div style="background:#fb641b; color:white !important; text-align:center; padding:8px; border-radius:2px; font-weight:600; margin-top:auto;">Add to Cart</div>
                </div>
            """, unsafe_allow_html=True)
