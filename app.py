import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import plotly.graph_objects as go

# Charger le modèle
model_path = "bestyYYmodelhair/"
model = tf.saved_model.load(model_path)

# Mapping des types de cheveux et des suggestions
hair_suggestions = {
    "Cheveux Bouclés": '''
    Hydratez vos boucles avec des produits sans sulfate et définissez-les avec un diffuseur. 
    
    Entretenez mieux vos cheveux 🏃‍♀️‍➡️🏃‍♀️‍➡️👉 https://ndeyecoiffure.fr/formations-tresses/ 
    
    Explorez d'autres modèles de cheveux 👉👉https://ndeyecoiffure.fr/shooting-photos/'''
    ,
    "Cheveux Raides": ''' 
    Boostez leur éclat avec un sérum lissant et protégez-les contre la chaleur avant tout coiffage. 
    
    Entretenez mieux vos cheveux 🏃‍♀️‍➡️🏃‍♀️‍➡️👉 https://ndeyecoiffure.fr/formations-tresses/ 
    
    Explorez d'autres modèles de cheveux 👉👉https://ndeyecoiffure.fr/shooting-photos/''',
    
    "Cheveux Souples ou Ondulés": ''' 
    Ajoutez du volume avec une mousse légère et définissez vos ondulations avec des sprays texturisants. 
    
    Entretenez mieux vos cheveux 🏃‍♀️‍➡️🏃‍♀️‍➡️👉 https://ndeyecoiffure.fr/formations-tresses/ 
    
    Explorez d'autres modèles de cheveux 👉👉https://ndeyecoiffure.fr/shooting-photos/''',
    
    "Dreadlocks": ''' 
    Lavez-les régulièrement avec un shampooing doux et hydratez vos racines pour des locks saines et brillantes. 
    
    Entretenez mieux vos cheveux 🏃‍♀️‍➡️🏃‍♀️‍➡️👉 https://ndeyecoiffure.fr/formations-tresses/ 
    
    Explorez d'autres modèles de cheveux 👉👉https://ndeyecoiffure.fr/shooting-photos/''' ,
    
    "Cheveux Crépus": ''' 
    Hydratez intensément avec des crèmes riches et protégez vos pointes avec des coiffures protectrices. 
    
    Entretenez mieux vos cheveux 🏃‍♀️‍➡️🏃‍♀️‍➡️👉 https://ndeyecoiffure.fr/formations-tresses/ 
    
    Explorez d'autres modèles de cheveux 👉👉https://ndeyecoiffure.fr/shooting-photos/'''
}

hair_types = list(hair_suggestions.keys())

# Fonction pour prédire le type de cheveux
def predict_hair_type(image_data):
    # Prétraiter l'image pour le modèle
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    #img_array = np.array(image) / 255.0
    img_array = np.expand_dims(image, axis=0).astype(np.float32)

    # Prédiction avec le modèle
    predict_fn = model.signatures["serving_default"]
    predictions = predict_fn(tf.constant(img_array))
    probabilities = predictions["dense_10"].numpy()[0]
    predicted_index = np.argmax(probabilities)
    return hair_types[predicted_index], probabilities[predicted_index] * 100

# Interface Streamlit
st.title("✨ Analyseur de Type de Cheveux ✨")
st.write("Chargez une image ou prenez une photo pour découvrir le type de cheveux et recevoir des conseils personnalisés.")

# Capture via caméra ou téléchargement
capture_mode = st.radio("Mode de saisie :", ("Télécharger une image", "Prendre une photo"))

if capture_mode == "Télécharger une image":
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = uploaded_file.read()
else:
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            return cv2.flip(img, 1)
    
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_processing=True
    )
    if webrtc_ctx and webrtc_ctx.video_transformer:
        frame = webrtc_ctx.video_transformer.frame
        if frame is not None:
            image_data = cv2.imencode(".jpg", frame)[1].tobytes()
        else:
            image_data = None

if 'image_data' in locals() and image_data:
    # Prédire le type de cheveux
    try:
        hair_type, confidence = predict_hair_type(image_data)

        # Layout avec image et graphique côte à côte
        col1, col2 = st.columns(2)

        # Afficher l'image
        with col1:
            st.image(image_data, caption="Image analysée", use_column_width=True)

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

        # Afficher le texte avec certitude
        st.markdown(f"<h3 style='text-align: center;'>Je suis certain à {confidence:.2f} % de ma prédiction</h3>", unsafe_allow_html=True)

        # Afficher les suggestions
        st.subheader(f"Type de Cheveux : {hair_type}")
        st.write(hair_suggestions[hair_type])

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'analyse : {e}")
else:
    st.info("Veuillez télécharger une image ou prendre une photo pour commencer.")
