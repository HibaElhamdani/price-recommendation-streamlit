import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================================================
# 1️⃣ Charger le modèle VotingClassifier
# ============================================================
voting_model = joblib.load("voting_final_model.pkl")

# ============================================================
# 2️⃣ Features SAFE (Price inclus)
# ============================================================
safe_features = [
    'No of Sellers', 'Few_Sellers', 
    'Cat_books', 'Cat_camera_and_photo', 'Cat_clothingand_shoes_and_jewelry',
    'Cat_electronics', 'Cat_gift_cards', 'Cat_toys_and_games', 'Cat_video_games',
    'Many_Reviews','Reviews_per_Seller','Price'
]

# ============================================================
# 3️⃣ Interface Streamlit
# ============================================================
st.set_page_config(page_title="Recommandation Prix", layout="wide")
st.title("💡 Recommandation du meilleur prix pour un produit")
st.markdown("Entrez les caractéristiques du produit et le modèle recommandera le prix optimal pour maximiser les chances d'être best-seller.")

# Colonnes pour inputs
col1, col2, col3 = st.columns(3)

with col1:
    product_name = st.text_input("Nom du produit", "Exemple")
    category = st.selectbox("Catégorie", ["Books", "Camera & Photo", "Clothing & Jewelry", 
                                          "Electronics", "Gift Cards", "Toys & Games", "Video Games"])
    
with col2:
    reviews_count = st.number_input("Nombre de reviews", min_value=0, value=16853)
    no_of_sellers = st.number_input("Nombre de vendeurs", min_value=1, value=1)

with col3:
    initial_price = st.number_input("Prix actuel", min_value=0.0, value=39.99, step=0.01)
    min_price = st.number_input("Prix minimum à tester", value=5.0)
    max_price = st.number_input("Prix maximum à tester", value=50.0)
    n_prices = st.number_input("Nombre de variantes de prix", min_value=2, value=10)

st.markdown("---")

# Bouton pour lancer la recommandation
if st.button("Recommander le meilleur prix"):

    # Créer les variants de prix
    prices = np.linspace(min_price, max_price, n_prices)
    variants = []

    for price in prices:
        temp = {
            'No of Sellers': no_of_sellers,
            'Few_Sellers': int(no_of_sellers <= 2),
            'Cat_books': int(category=="Books"),
            'Cat_camera_and_photo': int(category=="Camera & Photo"),
            'Cat_clothingand_shoes_and_jewelry': int(category=="Clothing & Jewelry"),
            'Cat_electronics': int(category=="Electronics"),
            'Cat_gift_cards': int(category=="Gift Cards"),
            'Cat_toys_and_games': int(category=="Toys & Games"),
            'Cat_video_games': int(category=="Video Games"),
            'Many_Reviews': int(reviews_count > 16853),  
            'Reviews_per_Seller': reviews_count / no_of_sellers,
            'Price': price
        }
        variants.append(temp)

    product_df = pd.DataFrame(variants)[safe_features]

    # Prédire les probabilités
    probas = voting_model.predict_proba(product_df)[:,1]

    # Meilleur prix
    best_idx = np.argmax(probas)
    best_price = prices[best_idx]
    best_prob = probas[best_idx]

    # Afficher résultat principal
    st.success(f"💰 Meilleur prix recommandé : **{best_price:.2f}**")
    st.info(f"📈 Probabilité d'être best-seller à ce prix : **{best_prob:.2f}**")

    # Top 3 des prix
    top3_idx = np.argsort(probas)[-3:][::-1]
    top3_prices = prices[top3_idx]
    top3_probs = probas[top3_idx]
    df_top3 = pd.DataFrame({"Prix": top3_prices, "Probabilité": top3_probs})
    st.subheader("🏆 Top 3 des prix recommandés")
    st.dataframe(df_top3.style.format({"Prix": "{:.2f}", "Probabilité": "{:.2f}"}))

    # Tableau complet
    df_result = pd.DataFrame({"Prix": prices, "Probabilité": probas})
    st.subheader("📊 Probabilité pour chaque variante de prix")
    st.dataframe(df_result.style.format({"Prix": "{:.2f}", "Probabilité": "{:.2f}"}))

    # Graphique interactif
    st.subheader("📈 Visualisation des probabilités selon le prix")
    fig = px.line(df_result, x="Prix", y="Probabilité", markers=True, title="Probabilité d'être best-seller selon le prix")
    fig.update_layout(yaxis_range=[0,1])
    st.plotly_chart(fig)
