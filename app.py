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
    Les cheveux bouclés nécessitent une hydratation régulière avec des produits adaptés pour éviter la sécheresse.  
    * Utilisez des produits sans sulfate pour préserver la texture naturelle.  
    * Séchez vos cheveux avec un diffuseur pour maintenir la définition des boucles.  
    Découvrez nos formations pour entretenir vos cheveux ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Cheveux Raides": '''
    Les cheveux raides bénéficient d'un entretien simple mais doivent être protégés contre les agressions externes.  
    * Appliquez un sérum lissant pour un effet brillant et naturel.  
    * Protégez-les avec un spray thermique avant tout coiffage.  
    Accédez à nos conseils d'entretien ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Cheveux Souples ou Ondulés": '''
    Les cheveux souples ou ondulés nécessitent des soins pour conserver leur volume et texture naturelle.  
    * Utilisez une mousse légère pour apporter du volume.  
    * Appliquez un spray texturisant pour définir les ondulations.  
    Découvrez plus de conseils sur : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Dreadlocks": '''
    Les dreadlocks requièrent un entretien spécifique pour rester saines et brillantes.  
    * Nettoyez-les régulièrement avec un shampoing doux.  
    * Hydratez vos racines pour éviter les démangeaisons.  
    Découvrez nos formations ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Cheveux Crépus": '''
    Les cheveux crépus doivent être hydratés intensément pour prévenir la casse.  
    * Utilisez des crèmes riches et des huiles pour maintenir l'humidité.  
    * Adoptez des coiffures protectrices pour protéger vos pointes.  
    Retrouvez nos conseils ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
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

# Affichage des 4 images fixes
# Affichage des 4 images fixes
st.subheader("PRENEZ DES PHOTOS CLAIRES :")
col1, col2 = st.columns(2)

# Images existantes
photos = ["0.jpg","1.jpg","2.jpg","3.jpg"]

with col1:
    st.image(photos[0], caption="Photo 1", use_column_width=True)
    st.image(photos[1], caption="Photo 3", use_column_width=True)

with col2:
    st.image(photos[2], caption="Photo 2", use_column_width=True)
    st.image(photos[3], caption="Photo 4", use_column_width=True)

# Téléchargement de l'image
st.subheader("COMMENCEZ EN PRENANT UNE PHOTO DEPUIS VOTRE GALERIE :")
uploaded_file = st.file_uploader("Importer une photo", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Charger l'image
    image_data = Image.open(uploaded_file)

    # Prédire le type de cheveux
    hair_type, confidence = predict_hair_type(image_data)

    # Layout avec image et graphique côte à côte
    col1, col2 = st.columns(2)

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
        fig.update_layout(height=300, width=300)
        st.plotly_chart(fig, use_container_width=True)

    # Afficher le type de cheveux et les suggestions
    #st.subheader(f"Type de Cheveux : {hair_type}")

    # Présenter les suggestions en 2 colonnes
    suggestions = hair_suggestions[hair_type].split("\n")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ANALYSE DES CHEVEUX")
        st.write(f''' 
            Les cheveux identifiés sont de type {hair_type}, une catégorie reconnue pour ses caractéristiques distinctives et sa structure unique.

            Ce type de chevelure exige une attention particulière pour préserver son éclat naturel et sa vitalité.            
            
            Grâce à une approche personnalisée et des pratiques capillaires adaptées, il est possible de maximiser leur potentiel esthétique !.''')            
    with col2:
        st.subheader("ENJEUX ET SOLUTIONS ")
        for i in suggestions:
            st.write(i)
else:
    st.info("Veuillez importer une photo pour commencer l'analyse.")
