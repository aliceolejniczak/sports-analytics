import os
import time
import pandas as pd
from PIL import Image, ExifTags
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

RAW_DATA_DIR = "data/raw"
OUTPUT_FILE = "data/processed/meals_metadata.csv"
FOOD_MODEL_NAME = "nateraw/food"
_food_processor = None
_food_model = None

CALORIES_PER_PORTION = {
    'salad': 250, 'soup': 300, 'pizza': 800, 'pasta': 650,
    'rice': 450, 'chicken': 500, 'beef': 600, 'steak': 650,
    'fish': 450, 'salmon': 500, 'sandwich': 550, 'burger': 750,
    'fries': 450, 'french fries': 450, 'dessert': 450, 'cake': 450,
    'ice cream': 300, 'apple pie': 400, 'bread': 250, 'egg': 150,
    'sushi': 500, 'taco': 250, 'donut': 300, 'hot dog': 550,
}

def get_exif_date(image_path):
    """
    Extrait la date de l'image (EXIF). Si l'EXIF est absent, 
    utilise la date de modification du fichier (Fallback).
    """
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag_id, val in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    return val.split(" ")[0].replace(":", "-")
    except:
        pass 

    try:
        timestamp = os.path.getmtime(image_path)
        return time.strftime('%Y-%m-%d', time.localtime(timestamp))
    except Exception as e:
        print(f"Erreur de lecture de la date pour {image_path}: {e}")
        return None

def get_food_model():
    """Charge le modèle Food-101 pré-entraîné."""
    global _food_processor, _food_model
    if _food_processor is None or _food_model is None:
        print(f"Chargement du modèle alimentaire {FOOD_MODEL_NAME}...")
        _food_processor = AutoImageProcessor.from_pretrained(FOOD_MODEL_NAME)
        _food_model = AutoModelForImageClassification.from_pretrained(FOOD_MODEL_NAME)
        _food_model.eval()
    return _food_processor, _food_model

def calories_for_label(label):
    label = label.lower().replace('_', ' ')
    for food_name, calories in CALORIES_PER_PORTION.items():
        if food_name in label:
            return calories
    return 500

def estimate_macronutrients_and_calories(image_path):
    """
    Identifie le plat avec Food-101 puis applique une portion nutritionnelle. La portion reste une approximation car l'image ne fournit pas
    directement le poids des aliments.
    """
    with Image.open(image_path) as image:
        inputs = get_food_model()[0](images=image.convert('RGB'), return_tensors='pt')

    with torch.no_grad():
        logits = get_food_model()[1](**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    class_index = int(probabilities.argmax())
    label = get_food_model()[1].config.id2label[class_index]
    confidence = float(probabilities[class_index])
    calories = calories_for_label(label)
    confidence_label = 'high' if confidence >= 0.70 else 'medium' if confidence >= 0.35 else 'low'
    protein = round(calories * 0.20 / 4, 1)
    carbohydrates = round(calories * 0.50 / 4, 1)
    fat = round(calories * 0.30 / 9, 1)

    return {
        'calories_consumed': calories,
        'detected_food': label,
        'food_model_confidence': round(confidence, 4),
        'protein_g': protein,
        'carbohydrates_g': carbohydrates,
        'fat_g': fat,
        'calorie_estimation_confidence': confidence_label
    }

def process_all_images(base_dir):
    records = []
    
    if not os.path.exists(base_dir):
        print(f"⚠️ Le dossier {base_dir} n'existe pas.")
        return pd.DataFrame()
    
    for participant in os.listdir(base_dir):
        if participant.startswith("p"):
            images_dir = os.path.join(base_dir, participant, "food-images")
            
            if os.path.exists(images_dir) and os.path.isdir(images_dir):
                images_found = [img for img in os.listdir(images_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images_found:
                    print(f"-> 📂 {len(images_found)} images trouvées dans {images_dir}")
                    
                for img_file in images_found:
                    img_path = os.path.join(images_dir, img_file)
                    
                    photo_date = get_exif_date(img_path)
                    nutrition = estimate_macronutrients_and_calories(img_path)
                    
                    if photo_date:
                        p_id = participant.replace("p", "").lstrip("0")
                        records.append({
                            "participant_id": p_id,
                            "date": photo_date,
                            "image_name": img_file,
                            **nutrition
                        })
                            
    df = pd.DataFrame(records)
    if not df.empty:
        daily = df.groupby(['participant_id', 'date']).agg(
            calories_consumed=('calories_consumed', 'sum'),
            protein_g=('protein_g', 'sum'),
            carbohydrates_g=('carbohydrates_g', 'sum'),
            fat_g=('fat_g', 'sum'),
            food_image_count=('image_name', 'count'),
            detected_food=('detected_food', lambda values: ', '.join(values)),
            food_model_confidence=('food_model_confidence', 'mean'),
            calorie_estimation_confidence=('calorie_estimation_confidence', 'first')
        ).reset_index()
        return daily
    else:
        return pd.DataFrame(columns=['participant_id', 'date', 'calories_consumed'])

def main():
    print("Démarrage de l'extraction des métadonnées images...")
    df_meals = process_all_images(RAW_DATA_DIR)
    
    if df_meals.empty:
        print("\n❌ ERREUR : Aucune image trouvée.")
    else:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_meals.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Traitement terminé. {len(df_meals)} jours de repas enregistrés.")
        print(f"Fichier sauvegardé : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()