import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Charger le modèle
model_path = "bestyYYmodelhair/"
model = tf.saved_model.load(model_path)

# Mapping des types de cheveux et des suggestions
hair_suggestions = {
    "Cheveux Bouclés": '''
    * Hydratez régulièrement avec des produits sans sulfate.  
    * Séchez vos cheveux avec un diffuseur pour des boucles définies.  
    ''',
    "Cheveux Raides": '''
    * Appliquez un sérum pour un effet brillant.  
    * Protégez avec un spray thermique avant le coiffage.  
    ''',
    "Cheveux Souples ou Ondulés": '''
    * Utilisez une mousse légère pour le volume.  
    * Appliquez un spray texturisant pour définir les ondulations.  
    ''',
    "Dreadlocks": '''
    * Nettoyez régulièrement avec un shampoing doux.  
    * Hydratez les racines pour éviter les démangeaisons.  
    ''',
    "Cheveux Crépus": '''
    * Hydratez intensément avec des huiles riches.  
    * Adoptez des coiffures protectrices pour préserver vos pointes.  
    '''
}

hair_types = list(hair_suggestions.keys())

# Fonction pour prédire le type de cheveux
def predict_hair_type(image):
    # Prétraiter l'image pour le modèle
    image = image.resize((224, 224))
    img_array = np.expand_dims(image, axis=0).astype(np.float32)

    # Prédiction avec le modèle
    predict_fn = model.signatures["serving_default"]
    predictions = predict_fn(tf.constant(img_array))
    probabilities = predictions["dense_10"].numpy()[0]
    predicted_index = np.argmax(probabilities)
    return hair_types[predicted_index], probabilities[predicted_index] * 100

# Interface Streamlit
st.title("✨ Analysez vos Cheveux ✨")

# Affichage des 4 images fixes avec une taille adaptée
st.subheader("📸 Prenez des photos claires :")
col1, col2 = st.columns(2)

# Images existantes
photos = ["00.jpg", "01.jpg", "02.jpg", "03.jpg"]

with col1:
    st.image(photos[0], caption="Photo 1", use_column_width=True, width=150)
    st.image(photos[1], caption="Photo 2", use_column_width=True, width=150)

with col2:
    st.image(photos[2], caption="Photo 3", use_column_width=True, width=150)
    st.image(photos[3], caption="Photo 4", use_column_width=True, width=150)

# Téléchargement de l'image
st.subheader("📂 Importez une photo de votre galerie :")
uploaded_file = st.file_uploader("Importer une photo", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Charger l'image
    image_data = Image.open(uploaded_file)

    # Prédire le type de cheveux
    hair_type, confidence = predict_hair_type(image_data)

    # Layout avec image et graphique côte à côte
    col1, col2 = st.columns([1, 1.2])

    # Afficher l'image téléchargée
    with col1:
        st.image(image_data, caption="Image téléchargée", use_column_width=True)

    # Créer un graphique avec Plotly
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "Confiance (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "darkblue"}}
        ))
        fig.update_layout(height=200, width=200)
        st.plotly_chart(fig, use_container_width=True)

    # Présenter les résultats
    st.subheader(f"💇‍♀️ Type de Cheveux : {hair_type}")
    st.markdown(f"**Suggestions :** {hair_suggestions[hair_type]}")

else:
    st.info("Veuillez importer une photo pour commencer l'analyse.")
