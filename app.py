import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Cache pour le chargement du modèle
@st.cache_resource
def load_model():
    model_path = "Openai_classifier_tf.h5"  # Assurez-vous que le chemin est correct
    return tf.keras.models.load_model(model_path)

model=load_model()


# Mapping des types de cheveux et des suggestions
hair_suggestions = {
    "Cheveux Bouclés": '''
    Caractéristiques : La densité et le volume naturels des cheveux bouclés offrent une base idéale pour des tresses protectrices qui subliment les boucles.

    Recommandations de coiffures tressées :

    Vanilles (Twists) : Parfaites pour protéger et définir les boucles, tout en apportant une touche chic et naturelle.

    Box Braids avec extensions : Un style protecteur emblématique qui combine élégance et polyvalence.

    Tresses collées avec rajouts : Idéales pour contrôler le volume tout en offrant un look sophistiqué. 
     
    Découvrez nos formations pour entretenir vos cheveux ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Cheveux Raides": '''
    Caractéristiques : Naturellement lisses et brillants, les cheveux raides se prêtent à des tresses élégantes qui ajoutent du relief et de la sophistication.

    Recommandations de coiffures tressées :

    Tresse française classique : Une coiffure intemporelle et raffinée qui sublime la fluidité des cheveux raides.

    Box Braids fines avec extensions : Idéales pour ajouter de la texture et jouer avec des styles audacieux.
    
    Tresse Ateba : Une touche artistique et colorée pour personnaliser votre look.  
    
    Accédez à nos conseils d'entretien ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Cheveux Souples ou Ondulés": '''
    Caractéristiques : Avec leur ondulation naturelle en "S", les cheveux ondulés donnent du caractère aux tresses et permettent de créer des styles bohèmes et décontractés.

    Recommandations de coiffures tressées :

    Bohemian Braids : Ces tresses rehaussent les ondulations naturelles grâce à des mèches libres qui apportent une touche romantique.

    Butterfly Braids : Volumineuses et spectaculaires, elles jouent avec la texture pour un look audacieux.

    Fulani Braids : Un mélange de tradition et de modernité, enrichies d’accessoires pour accentuer votre style.
      
    Découvrez plus de conseils sur : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Dreadlocks": '''
    Les dreadlocks requièrent un entretien spécifique pour rester saines et brillantes.  
    * Nettoyez-les régulièrement avec un shampoing doux.  
    * Hydratez vos racines pour éviter les démangeaisons.  
    Découvrez nos formations ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    ''',
    "Cheveux Crépus": '''
    Caractéristiques : Les cheveux crépus, avec leur structure serrée et volumineuse, sont parfaits pour des coiffures tressées durables qui célèbrent leur texture unique.

    Recommandations de coiffures tressées :

    Tresses Collées Simples : Une coiffure minimaliste et élégante qui protège efficacement les cheveux.

    Box Braids épaisses : Une option incontournable pour un style audacieux et une protection optimale.

    Vanilles épaisses : Simples à entretenir, elles mettent en valeur la richesse des cheveux crépus tout en maintenant leur hydratation. 
    Retrouvez nos conseils ici : [Formations Tresses](https://ndeyecoiffure.fr/formations-tresses).  
    '''
}

hair_types = list(hair_suggestions.keys())

# Fonction pour prédire le type de cheveux

def predict_hair_type(image):
    image = image.resize((224, 224))
    #img_array = np.array(image) / 255.0  # Normalisation
    img_array = np.expand_dims(image, axis=0)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100
    return hair_types[predicted_index], confidence



# Interface Streamlit
st.title("✨ Analysez vos Cheveux ✨")

# Affichage des 4 images fixes
st.subheader("PRENEZ DES PHOTOS CLAIRES :")
col1, col2 = st.columns(2)

# Images existantes
photos = ["00.jpg", "01.jpg", "02.jpg", "03.jpg"]

with col1:
    st.image(photos[0], caption="Photo 1", use_container_width=True)
    st.image(photos[1], caption="Photo 3", use_container_width=True)

with col2:
    st.image(photos[2], caption="Photo 2", use_container_width=True)
    st.image(photos[3], caption="Photo 4", use_container_width=True)

# Section pour télécharger ou capturer une image
st.subheader("COMMENCEZ EN PRENANT UNE PHOTO :")
image_data = None


# Test de la caméra
photo = st.camera_input("Cliquez pour prendre une photo")

if photo is not None:
    # Charger l'image capturée
    image_data = Image.open(photo)
    image_data = image_data.convert("RGB")  # S'assurer que l'image est en RGB

# Option pour télécharger une image
uploaded_file = st.file_uploader("Ou importer une photo", type=["jpg", "jpeg"], accept_multiple_files=False, label_visibility="visible")
if uploaded_file is not None:
    image_data = Image.open(uploaded_file)

# Si une image est disponible
if image_data:
    # Prédire le type de cheveux
    hair_type, confidence = predict_hair_type(image_data)

    # Layout avec image et graphique côte à côte
    col1, col2 = st.columns(2)

    # Afficher l'image téléchargée ou capturée
    with col1:
        st.image(image_data, caption="Image analysée", use_container_width=True)

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
    st.subheader(f"Type de Cheveux : {hair_type}")
    st.write(hair_suggestions[hair_type])
else:
    st.info("Veuillez importer ou capturer une photo pour commencer l'analyse.")
