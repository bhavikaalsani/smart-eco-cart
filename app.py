
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load product database
df = pd.read_csv("product_db.csv")

st.title("🌱 Smart Eco Cart – Sustainability Advisor 🛒")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Product", use_column_width=True)

    # Save temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Run YOLO
    results = model(image_path)
    boxes = results[0].boxes
    classes = [model.names[int(cls)] for cls in boxes.cls]
    st.subheader("🧠 Detected Products:")
    st.write(classes)

    detected_class = classes[0] if classes else None

    if detected_class:
        matched_products = df[df["category"] == detected_class]
        if not matched_products.empty:
            for _, row in matched_products.iterrows():
                st.markdown(f"""
                ### 🛒 Product: {row['product_name']} ({row['brand']})
                🌿 **Eco Rating:** {row['eco_rating']}/5  
                ♻️ **Suggested Alternative:** {row['eco_alternative']}  
                📦 **Packaging Type:** {row['packaging_type']}  
                💨 **Carbon Footprint:** {row['carbon_score']} kg CO₂  
                🔁 **Recyclability:** {row['recyclability']}  
                💡 **Eco Tip:** {row['eco_tip']}  
                """)
        else:
            st.warning("No eco insights available.")
