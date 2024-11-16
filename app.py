import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Charger le modèle
model_path = "bestyYYmodelhair/"
model = tf.saved_model.load(model_path)

# Mapping des types de cheveux et des suggestions
hair_suggestions = {
    "Cheveux Bouclés": "Voici des astuces pour bien prendre soin des cheveux bouclés, etc...",
    "Cheveux Raides": "Voici des conseils pour les cheveux raides, etc...",
    "Cheveux Souples ou Ondulés": "Voici des recommandations pour les cheveux ondulés, etc...",
    "Dreadlocks": "Voici des conseils détaillés pour l'entretien des dreadlocks, etc...",
    "Cheveux Crépus": "Voici des astuces pour les cheveux crépus, etc..."
}

hair_types = list(hair_suggestions.keys())

# Fonction pour prédire le type de cheveux
def predict_hair_type(image_data):
    # Prétraiter l'image pour le modèle
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    #img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Prédiction avec le modèle
    predict_fn = model.signatures["serving_default"]
    predictions = predict_fn(tf.constant(img_array))
    predicted_index = np.argmax(predictions["dense_10"].numpy())
    return hair_types[predicted_index]

# Interface Streamlit
st.title("Analyseur de Type de Cheveux")
st.write("Chargez une image pour découvrir le type de cheveux et recevoir des conseils personnalisés.")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Afficher l'image téléchargée
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)

    try:
        # Lire l'image
        image_data = uploaded_file.read()

        # Prédire le type de cheveux
        hair_type = predict_hair_type(image_data)

        # Afficher le résultat
        st.subheader(f"Type de Cheveux : {hair_type}")
        st.write(hair_suggestions[hair_type])

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
