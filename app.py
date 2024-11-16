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
    "Cheveux Bouclés": ''' 
    Voici des astuces pour bien prendre soin des cheveux bouclés, qui nécessitent une attention particulière pour assurer leur hydratation, leur définition et leur éclat.

1. Hydratation et Nutrition
Choisissez des après-shampoings riches en hydratation : Les cheveux bouclés ont besoin d'une bonne hydratation. Optez pour un après-shampoing riche qui nourrira vos boucles sans les alourdir.
Masques nourrissants toutes les semaines : Utilisez un masque capillaire nourrissant chaque semaine pour revitaliser vos cheveux, en mettant l'accent sur les longueurs et les pointes.
Appliquez des huiles nutritives : Les huiles comme l'huile de coco ou l'huile d'olive peuvent être appliquées sur les pointes pour éviter le dessèchement et garder les boucles bien définies.

2. Produits Adaptés
Optez pour des shampoings doux : Évitez les shampoings agressifs. Préférez des formules douces qui préservent l’hydratation naturelle de vos cheveux.
Utilisez des produits texturisants : Les gels ou crèmes coiffantes conçus pour les cheveux bouclés aident à définir vos boucles tout en apportant une hydratation supplémentaire.
Évitez les sulfates : Les shampoings contenant des sulfates peuvent assécher vos boucles. Choisissez des produits sans sulfates pour maintenir l’hydratation.

3. Techniques de Lavage
Ne lavez pas trop souvent : Un lavage tous les deux ou trois jours est idéal pour préserver l'hydratation naturelle de vos cheveux.
Utilisez de l’eau froide pour rincer : Un rinçage à l’eau froide aide à refermer les cuticules et à réduire les frisottis.
Essayez le co-wash : Utilisez un après-shampoing nettoyant une fois par semaine pour nettoyer vos cheveux sans les assécher.

4. Démêlage et Coiffage
Démêlez doucement : Utilisez un peigne à dents larges ou vos doigts pour démêler vos cheveux mouillés après application de l’après-shampoing.
Technique de scrunching : Après avoir appliqué votre produit coiffant, froissez vos cheveux vers le cuir chevelu pour bien définir les boucles.
Utilisez une serviette en microfibre : Cela aide à absorber l'excès d'eau sans créer de frisottis.

5. Protection Externe
Appliquez un protecteur thermique : Avant d'utiliser des outils chauffants, protégez vos cheveux avec un produit adapté.
Réduisez l’usage d'outils chauffants : Évitez de chauffer vos cheveux trop souvent pour prévenir la déshydratation.
Sprays anti-humidité : Utilisez un spray pour protéger vos boucles de l'humidité ambiante qui peut créer des frisottis.

6. Soins Spécifiques
Masques protéinés occasionnels : Un masque contenant des protéines une fois par mois aide à renforcer vos boucles.
Produits à base d'aloe vera : Les soins contenant de l'aloe vera sont parfaits pour hydrater sans alourdir.
Sprays hydratants pour les retouches : Ils permettent de rafraîchir vos boucles tout au long de la journée.

7. Entretien Régulier
Coupe régulière : Une coupe toutes les 6 à 8 semaines permet d'éliminer les pointes abîmées et de garder une belle forme.
Demandez une coupe adaptée : Une coupe en couches aide à mettre en valeur vos boucles sans les alourdir.

8. Sommeil Protecteur
Utilisez une taie d'oreiller en satin : Cela réduit les frottements pendant la nuit, évitant ainsi les frisottis.
Technique du "pineapple" : Attachez vos cheveux en haut de la tête avant de dormir pour préserver leur forme.

9. Contrôle des Frisottis
Sérums anti-frisottis : Un sérum léger aide à lisser les boucles tout en réduisant les frisottis.
Limitez la manipulation : Évitez de toucher vos cheveux après les avoir coiffés pour préserver leur définition.

10. Alimentation et Hydratation
Mangez équilibré : Une alimentation riche en vitamines et minéraux est essentielle pour des cheveux sains.
Buvez suffisamment d'eau : L'hydratation intérieure est clé pour maintenir l'éclat de vos boucles.

En suivant ces conseils, vous pourrez profiter de cheveux bouclés bien hydratés et définis.
    
    ''',
    "Cheveux Raides": ''' 
    Voici des conseils détaillés pour maintenir la santé, la brillance et la douceur des cheveux raides.

1. Hydratation Légère
Choisissez des après-shampoings légers : Pour éviter d’alourdir les cheveux, optez pour des produits hydratants légers.
Masques capillaires hebdomadaires : Appliquez un masque hydratant une fois par semaine, en évitant les racines.
Huiles légères : Utilisez des huiles comme le jojoba sur les pointes pour nourrir sans alourdir.

2. Choix des Produits
Shampoings volumisants : Utilisez des shampoings spécifiques pour apporter du corps aux cheveux fins.
Évitez les silicones : Privilégiez les produits sans silicone pour un aspect naturel.
Sprays texturisants : Appliquez un spray à l’eau de mer pour créer du mouvement sans alourdir.

3. Techniques de Lavage
Lavez tous les deux jours : Cela permet de garder les cheveux propres sans les dessécher.
Utilisez de l’eau tiède ou froide : Pour éviter d'ouvrir trop les cuticules, terminez par un rinçage à l’eau froide.

4. Démêlage et Coiffage
Démêlez délicatement : Un peigne à dents larges est recommandé pour éviter les nœuds.
Brossez doucement : Répartissez le sébum des racines aux pointes pour une brillance naturelle.

5. Protection Contre les Agressions
Utilisez un protecteur thermique : Avant d'utiliser des outils chauffants, appliquez un protecteur.
Limitez l'utilisation d'outils chauffants : Deux à trois fois par semaine est suffisant pour éviter la casse.
Protection solaire : Un spray UV peut aider à préserver l'éclat de vos cheveux.

6. Soins Quotidiens
Coiffage minimal : Limitez la manipulation et évitez les coiffures serrées.
Produits anti-frisottis : Un sérum léger peut aider à maintenir une apparence soignée.

7. Coupe Régulière
Coupez toutes les 6 à 8 semaines : Cela élimine les fourches et garde les cheveux en bonne santé.
Adaptez la coupe à votre style : Des couches légères peuvent ajouter du volume.

8. Évitez l'Excès de Sébum
Shampoings secs : Utilisez-en pour absorber l’excès de sébum entre les lavages.
Ne touchez pas vos cheveux : Évitez de passer les mains dans vos cheveux pour prévenir l'accumulation de sébum.

9. Alimentation et Hydratation
Consommez des oméga-3 : Aliments comme le saumon et les noix favorisent la brillance.
Hydratez-vous : Boire suffisamment d’eau aide à garder les cheveux doux.

En suivant ces conseils, vous pourrez préserver la légèreté, la brillance et la souplesse de vos cheveux raides.
    
    ''',
    "Cheveux Souples ou Ondulés": ''' 
    Voici des recommandations pour prendre soin des cheveux ondulés, en équilibrant hydratation et contrôle des frisottis pour des ondulations bien définies et volumineuses.

1. Hydratation et Nutrition
Choisissez des après-shampoings légers et hydratants : Cela permet d'hydrater sans alourdir vos ondulations.
Masques nourrissants toutes les 10 jours : Appliquez un masque sur les longueurs et les pointes.
Huiles légères : Appliquez des huiles comme l’argan sur les pointes pour éviter le dessèchement.

2. Produits Adaptés
Shampoings sans sulfates : Ils préservent les huiles naturelles essentielles pour la définition des ondulations.
Sprays texturisants : Utilisez un spray pour donner corps et tenue à vos ondulations.
Crèmes et mousses pour ondulations : Elles aident à définir et hydrater vos vagues.

3. Techniques de Lavage
Lavez tous les 2 à 3 jours : Un lavage modéré maintient l’hydratation.
Utilisez de l’eau froide pour rincer : Cela aide à dompter les frisottis et à donner de la brillance.

4. Démêlage et Coiffage
Démêlez délicatement : Préférez le démêlage sur cheveux mouillés avec un peigne large.
Coiffage doux : Évitez les coiffures trop serrées qui peuvent casser vos ondulations.

5. Protection Externe
Appliquez un protecteur thermique : Avant de coiffer avec des outils chauffants, appliquez un produit protecteur.
Évitez les outils chauffants trop fréquents : Limitez leur utilisation à quelques fois par semaine.
Sprays anti-frisottis : Un spray léger permet de dompter les frisottis pendant la journée.

6. Soins Réguliers
Utilisez un masque hydratant toutes les semaines : Cela aide à garder l'hydratation nécessaire.
Coupes régulières : Une coupe tous les 6 à 8 semaines permet d’éliminer les pointes fourchues.

7. Sommeil Protecteur
Taie d'oreiller en satin : Cela minimise les frottements pendant la nuit.
Chignon haut : Attachez vos cheveux en un chignon lâche pour préserver la forme de vos ondulations.

8. Alimentation et Hydratation
Mangez des aliments riches en vitamines : Une alimentation équilibrée favorise la santé des cheveux.
Hydratez-vous bien : Boire de l’eau contribue à l’hydratation de vos cheveux.

En suivant ces conseils, vous obtiendrez des cheveux ondulés brillants, bien définis et hydratés
    
    ''',
    "Dreadlocks": ''' 
    Voici des conseils détaillés pour l'entretien des dreadlocks, qui nécessitent des soins particuliers pour les garder propres, sains et bien formés.

1. Lavage et Nettoyage
Lavez régulièrement mais pas trop fréquemment : Un lavage toutes les 1 à 2 semaines est idéal. Les dreadlocks n’ont pas besoin d'être lavées aussi souvent que les cheveux non tressés, car elles retiennent naturellement les huiles. Un lavage trop fréquent peut perturber leur formation et les assécher.
Utilisez des shampoings sans résidus : Les dreadlocks retiennent facilement les produits, donc privilégiez un shampoing sans résidus ni sulfates pour éviter les accumulations. Appliquez-le en frottant doucement le cuir chevelu et laissez l’eau savonneuse couler à travers les locks sans trop les manipuler.
Rincez abondamment : Assurez-vous de bien rincer vos cheveux pour éviter toute accumulation de produit, car cela pourrait alourdir vos dreadlocks ou causer des démangeaisons.

2. Hydratation et Nutrition
Appliquez une huile légère après le lavage : Une fois les locks bien séchées, appliquez une huile naturelle légère comme l’huile de jojoba, de pépins de raisin ou d’olive pour les nourrir sans les alourdir.
Vaporisez vos locks avec un mélange hydratant : Utilisez un spray fait maison avec de l’eau et quelques gouttes d’huile essentielle (comme le tea tree ou la lavande) pour hydrater et rafraîchir vos dreadlocks entre les lavages. Vaporisez-le légèrement chaque matin pour éviter la sécheresse.
Évitez les produits crémeux ou épais : Les crèmes et baumes peuvent laisser des résidus difficiles à rincer dans les locks. Optez plutôt pour des huiles légères ou des sprays d’hydratation.

3. Séchage
Assurez-vous d’un séchage complet après chaque lavage : Laissez vos dreadlocks sécher à l’air libre autant que possible, ou utilisez une serviette en microfibre pour absorber l’excès d’eau. Si vous utilisez un sèche-cheveux, choisissez une chaleur modérée et terminez par de l’air froid.
Évitez de dormir avec les dreadlocks mouillées : Cela peut entraîner une mauvaise odeur ou le développement de moisissures. Assurez-vous que vos locks soient complètement sèches avant de vous coucher.

4. Entretien et Torsion
Retwisting toutes les 4 à 6 semaines : Pour maintenir vos dreadlocks bien formées et éviter les nœuds, faites une retorsion (retwist) de la racine environ toutes les 4 à 6 semaines. Cela aidera à garder les repousses bien organisées sans exercer une tension excessive sur le cuir chevelu.
Évitez les torsions trop serrées : Des torsions trop serrées peuvent causer de la tension sur le cuir chevelu et affaiblir la racine des cheveux, ce qui pourrait entraîner une chute de cheveux à long terme. Optez pour une torsion douce.
Utilisez un gel léger et sans résidus pour la retorsion : Choisissez un gel spécifique pour locks, qui offre de la tenue sans résidus. Appliquez-le uniquement à la racine et torsadez avec modération.

5. Protection Contre les Aggressions Extérieures
Protégez vos locks pendant le sommeil : Enveloppez vos dreadlocks dans un foulard en satin ou dormez sur une taie d’oreiller en soie pour réduire la friction et préserver leur forme.
Évitez les expositions prolongées au soleil : Les UV peuvent assécher et ternir les dreadlocks. En cas de forte exposition au soleil, portez un chapeau ou un foulard pour les protéger.
Rincez après exposition au chlore ou à l’eau de mer : Si vous allez à la piscine ou à la mer, rincez vos dreadlocks avec de l'eau douce après chaque baignade pour éviter que le chlore ou le sel ne les dessèche.

6. Précautions Contre les Accumulations
Pratiquez un nettoyage en profondeur tous les 2 à 3 mois : Effectuez un bain de dreadlocks avec un mélange de bicarbonate de soude et de vinaigre de cidre pour éliminer les résidus accumulés. Trempez vos dreadlocks dans une bassine de ce mélange pendant 5 à 10 minutes, puis rincez abondamment.
Évitez les produits contenant de la cire : Les cires et beurres peuvent causer une accumulation difficile à enlever. Si vous souhaitez raffermir vos dreadlocks, optez pour des gels légers et naturels.

7. Hydratation du Cuir Chevelu
Massez régulièrement le cuir chevelu : Un cuir chevelu sain favorise la bonne croissance des dreadlocks. Massez doucement votre cuir chevelu avec de l'huile de ricin ou de jojoba pour stimuler la circulation sanguine.
Utilisez des huiles essentielles apaisantes : Si vous avez des démangeaisons, appliquez quelques gouttes d'huile essentielle de tea tree ou de menthe poivrée, mélangées à une huile de support (comme le jojoba), pour apaiser les irritations.

8. Conseils Généraux de Soin
Évitez de manipuler les dreadlocks constamment : Les tirer ou les tourner excessivement peut affaiblir leur structure. Manipulez-les avec soin pour éviter de les fragiliser.
Coupez les pointes abîmées : Si certaines extrémités deviennent fines ou effilochées, envisagez de les couper pour préserver l'intégrité des dreadlocks et éviter les fourches.
Évitez les styles trop serrés : Bien que les styles tressés ou attachés soient pratiques, ne les portez pas trop serrés pour préserver vos racines et éviter les cassures.

9. Alimentation et Hydratation
Hydratez-vous suffisamment : Une bonne hydratation est essentielle pour des cheveux sains, y compris pour les dreadlocks. Buvez de l'eau en quantité suffisante chaque jour.
Mangez équilibré : Assurez-vous d’avoir une alimentation riche en vitamines et minéraux, notamment en zinc, fer, et vitamines B pour une croissance saine et durable de vos dreadlocks.

En suivant ces conseils, vous pourrez garder vos dreadlocks propres, hydratées et en excellente santé.
    
    ''',
    "Cheveux Crépus": '''
    Les cheveux crépus nécessitent une attention particulière en raison de leur texture sèche et fragile. Voici des conseils pour les entretenir efficacement :

1. Lavage et Nettoyage
Fréquence : Lavez vos cheveux une fois par semaine ou tous les dix jours.
Shampoing : Utilisez un shampoing hydratant sans sulfates.
Pré-shampoing : Appliquez une huile nourrissante avant le lavage (ex. : huile de coco, olive, avocat).

2. Hydratation Intense
Après-shampoing : Utilisez un après-shampoing riche après chaque lavage.
Masques : Faites des masques hydratants hebdomadaires (beurre de karité, huile de ricin).
Scellement : Appliquez une huile ou un beurre après l'hydratation pour retenir l'humidité.

3. Démêlage en Douceur
Démêlage humide : Démêlez vos cheveux lorsqu'ils sont humides avec un démêlant et un peigne à dents larges.
Outils à éviter : Évitez les brosses et peignes fins.

4. Hydratation Quotidienne
Spray hydratant : Utilisez un spray léger contenant de l’eau et de l’aloe vera.
Méthode LOC : Appliquez un liquide, suivi d'une huile, puis d'une crème pour une hydratation durable.

5. Protection des Cheveux
Nuit : Portez un bonnet ou un foulard en satin pour réduire les frottements.
Coiffures protectrices : Optez pour des tresses ou nattes, sans tension excessive.

6. Évitez la Chaleur Excessive
Limitation des appareils chauffants : Utilisez des températures basses et privilégiez le séchage à l'air libre.

7. Soins du Cuir Chevelu
Hydratation : Appliquez une huile légère sur le cuir chevelu pour prévenir la sécheresse.
Massages : Massez le cuir chevelu pour stimuler la circulation sanguine.

8. Évitez les Produits Lourds
Produits légers : Préférez des produits naturels et évitez les résidus.
Nettoyage clarifiant : Utilisez un shampoing clarifiant une fois par mois.

9. Favorisez une Bonne Alimentation
Hydratation : Buvez suffisamment d’eau.
Alimentation : Consommez des aliments riches en fer, zinc, oméga-3, et vitamines B.

10. Traitements Spécifiques
Bains d’huile : Pratiquez des bains d'huile toutes les deux semaines pour renforcer les cheveux.
Masques protéinés : Utilisez un masque protéiné mensuel pour renforcer les fibres capillaires.

En suivant ces conseils, vous pourrez maintenir vos cheveux crépus en bonne santé, hydratés et forts.
    
    '''
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
