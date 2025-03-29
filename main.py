import pandas as pd
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from PIL import Image
import re
import cv2
from flask import Flask, request, jsonify
import numpy as np
from io import BytesIO
from collections import OrderedDict

app = Flask(__name__)


def load_data():
    """Load COSING and Brand datasets"""
    file_cosing = "COSING_Cleaned_Normalized_v7(1).csv"
    file_brand = "brend_cleaned.csv"
    df_cosing = pd.read_csv(file_cosing)
    df_brand = pd.read_csv(file_brand)
    return df_cosing, df_brand


def find_best_match(query, choices):
    """Find the closest match for a given query in a list of choices"""
    matches = get_close_matches(query, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None


def predict(description):
    """Simple placeholder for BERT prediction - replace with actual model"""
    return "Safe" if "low risk" in description.lower() else "Not Safe"


def predict_with_description(ingredient, df):
    """Mencari deskripsi bahan dan menggunakan model prediksi keamanan."""
    ingredient_lower = ingredient.lower()

    df['INCI name_lower'] = df['INCI name'].str.lower()
    df['IUPAC Name_lower'] = df['IUPAC Name'].str.lower()

    best_match = find_best_match(ingredient_lower, df['INCI name_lower'].dropna().tolist())
    if not best_match or df[df['INCI name_lower'] == best_match].empty:
        best_match = find_best_match(ingredient_lower, df['IUPAC Name_lower'].dropna().tolist())

    if best_match and not df[(df['INCI name_lower'] == best_match) | (df['IUPAC Name_lower'] == best_match)].empty:
        match = df[(df['INCI name_lower'] == best_match) | (df['IUPAC Name_lower'] == best_match)].iloc[0]
        inci_name = match['INCI name'].title()
        description = match['Description'] if pd.notna(match['Description']) else "Deskripsi tidak tersedia"
        function = match['Function'] if pd.notna(match['Function']) else "Fungsi tidak tersedia"
        risk_level = match['Risk Level'] if pd.notna(match['Risk Level']) else "Tidak ada informasi risiko"
        risk_description = match['Risk Description'] if pd.notna(
            match['Risk Description']) else "Deskripsi risiko tidak tersedia"
    else:
        inci_name = ingredient.title()
        description = "Description not found"
        function = "Function not found"
        risk_level = "Risk level not found"
        risk_description = "Risk description not found"

    result = predict(description)

    return OrderedDict([
        ("Ingredient Name", inci_name),
        ("Description", description),
        ("Function", function),
        ("Risk Level", risk_level),
        ("Risk Description", risk_description)
    ])


def find_matching_products(input_ingredients, df_brand, category_filter=None):
    """
    Find up to 6 products containing the given ingredients.
    If category_filter is provided, only search within that category.
    Otherwise, ensure at least 3 different categories in the result.
    """
    if 'ingridients' not in df_brand.columns or 'type' not in df_brand.columns:
        return []

    # Copy untuk menghindari modifikasi langsung pada df_brand
    df_brand = df_brand.copy()

    pattern = '|'.join(map(re.escape, input_ingredients))  # Buat regex untuk mencari semua bahan sekaligus
    df_brand['ingridients'] = df_brand['ingridients'].astype(str).fillna("")

    # Jika kategori spesifik diberikan, langsung filter berdasarkan kategori
    if category_filter:
        matched_products = df_brand[
            df_brand['ingridients'].str.contains(pattern, case=False, na=False) &
            (df_brand['type'].str.lower() == category_filter.lower())
            ]
        return matched_products[['brand', 'name', 'type', 'ingridients']].head(6).to_dict(orient='records')

    # Jika tidak ada kategori, cari semua produk yang mengandung bahan tersebut
    matched_products = df_brand[df_brand['ingridients'].str.contains(pattern, case=False, na=False)]

    if matched_products.empty:
        return []

    # Kelompokkan berdasarkan kategori ('type')
    grouped_by_type = matched_products.groupby('type')

    selected_products = []
    categories_used = set()

    # Ambil produk dari minimal 3 kategori berbeda
    for category, group in grouped_by_type:
        if len(categories_used) >= 3 and len(selected_products) >= 6:
            break  # Hentikan jika sudah memenuhi syarat

        # Ambil maksimal 2 produk dari kategori ini
        category_products = group.head(2).to_dict(orient='records')

        selected_products.extend(category_products)
        categories_used.add(category)

    return selected_products[:6]  # Pastikan hanya mengambil maksimal 6 produk


def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Gambar tidak bisa dibaca.")
    return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def extract_text_from_image(image_file):
    """Ekstrak teks dari gambar dan hanya ambil bahan yang cocok."""
    try:
        image = Image.open(BytesIO(image_file.read()))  # Convert ke format yang bisa dibaca PIL
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Grayscale
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_image = Image.fromarray(thresh)

        text = pytesseract.image_to_string(processed_image)
        text = re.sub(r'[^a-zA-Z0-9,\s/-]', '', text).replace("\n", " ")  # Bersihkan teks

        # Hapus kata-kata tidak penting seperti "Ingredients", "Komposisi"
        text = re.sub(r'\b(Ingredients|Komposisi|Composition|Bahan|Daftar Bahan)\b', '', text, flags=re.IGNORECASE)

        return text.strip()
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""


def correct_spelling(ingredient, df_cosing):
    all_ingredients = df_cosing['INCI name'].dropna().str.lower().tolist()
    match = get_close_matches(ingredient, all_ingredients, n=1, cutoff=0.7)
    return match[0] if match else ingredient


def filter_valid_ingredients(text, df_cosing):
    """Ambil hanya bahan yang cocok dengan daftar df_cosing."""
    ingredients = [word.strip() for word in text.split(',')]  # Pisahkan berdasarkan koma
    valid_ingredients = [correct_spelling(ingredient, df_cosing) for ingredient in ingredients]
    return ', '.join(filter(None, valid_ingredients))


@app.route('/analyze', methods=['POST'])
def analyze_ingredients():
    try:
        data = request.get_json(silent=True) or {}
        input_ingredients = []

        # Jika ada file gambar yang dikirim
        if 'ingredients' in request.files:
            image = request.files['ingredients']
            if image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                extracted_text = extract_text_from_image(image)
                if extracted_text:
                    print("Bahan yang diekstrak:", extracted_text)
                    input_ingredients = [ing.strip().lower() for ing in extracted_text.split(",") if ing.strip()]
                else:
                    return jsonify({"error": "Failed to extract ingredients from image"}), 400

        # Jika dikirim sebagai JSON teks
        ingredients_data = data.get('ingredients', [])
        if isinstance(ingredients_data, str):
            input_ingredients.extend([ing.strip().lower() for ing in ingredients_data.split(",") if ing.strip()])
        elif isinstance(ingredients_data, list):
            input_ingredients.extend([ing.strip().lower() for ing in ingredients_data if ing.strip()])
        else:
            return jsonify({"error": "Invalid 'ingredients' format"}), 400

        if not input_ingredients:
            return jsonify({"error": "No valid ingredients found"}), 400

        # Load data
        cosing_dict, df_brand = load_data()

        # Koreksi ejaan
        corrected_ingredients = [correct_spelling(ing, cosing_dict) for ing in input_ingredients]

        # Analisis bahan dengan ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda ing: predict_with_description(ing, cosing_dict), corrected_ingredients))

        # Filter produk berdasarkan kategori
        category_filter = data.get('category', "").strip().lower()
        matching_products = find_matching_products(corrected_ingredients, df_brand, category_filter)

        return jsonify({
            "Ingredient Analysis": results,
            "Matching Products": matching_products
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
