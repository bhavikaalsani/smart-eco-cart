import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# ---------------------- Load YOLO Model ----------------------
model = YOLO('yolov8n.pt')  # Replace with your custom model if needed

# ---------------------- Load Product Database ----------------------
product_db = pd.read_csv("product_db.csv")

# ---------------------- Session State ----------------------
if 'user' not in st.session_state:
    st.session_state.user = None
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------------- Barcode Decoder (OpenCV based) ----------------------
def decode_barcode_opencv(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    barcodes = cv2.barcode.detectAndDecode(img)[1]
    if barcodes:
        return barcodes[0]
    return None

# ---------------------- Sidebar Login ----------------------
st.sidebar.title("🔐 Login")
user_name = st.sidebar.text_input("Enter your name to continue")
if st.sidebar.button("Login"):
    st.session_state.user = user_name
    st.success(f"Welcome, {user_name}!")

# ---------------------- Main Interface ----------------------
if st.session_state.user:
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Upload Image", "📷 Scan Barcode", "💚 My History", "⚙️ Personalize"])

    # -------- Upload & Detect --------
    with tab1:
        st.title("🖼️ Detect Products from Image")
        uploaded_file = st.file_uploader("Upload a shopping image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results = model(img_np)
            names = results[0].names
            detected_classes = results[0].boxes.cls.tolist()

            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.subheader("🧠 Detected Products:")

            if detected_classes:
                for class_id in detected_classes:
                    product_name = names[int(class_id)]
                    match = product_db[product_db['detected_name'].str.lower() == product_name.lower()]
                    if not match.empty:
                        row = match.iloc[0]
                        eco_info = f"""
                        🛒 Product: **{row['product_name']}** ({row['brand']})  
                        🌿 Eco Rating: {row['eco_rating']}/5  
                        ♻️ Alternative: {row['eco_alternative']}  
                        📦 Packaging: {row['packaging']}  
                        💨 CO₂: {row['carbon_footprint']} kg  
                        🔁 Recyclability: {row['recyclability']}  
                        💡 Tip: _{row['eco_tip']}_  
                        """
                        st.markdown(eco_info)
                        st.session_state.history.append({"name": row['product_name'], "eco_rating": row['eco_rating']})
                    else:
                        st.warning(f"Detected '{product_name}', but not found in database.")
            else:
                st.info("No detectable product found in image.")

    # -------- Barcode Scanner --------
    with tab2:
        st.title("📷 Scan Barcode")
        picture = st.camera_input("Scan product barcode")

        if picture:
            barcode_data = decode_barcode_opencv(picture.getvalue())
            if barcode_data:
                st.success(f"Barcode: {barcode_data}")
                match = product_db[product_db['barcode'] == barcode_data]
                if not match.empty:
                    row = match.iloc[0]
                    eco_info = f"""
                    🛒 Product: **{row['product_name']}** ({row['brand']})  
                    🌿 Eco Rating: {row['eco_rating']}/5  
                    ♻️ Alternative: {row['eco_alternative']}  
                    📦 Packaging: {row['packaging']}  
                    💨 CO₂: {row['carbon_footprint']} kg  
                    🔁 Recyclability: {row['recyclability']}  
                    💡 Tip: _{row['eco_tip']}_  
                    """
                    st.markdown(eco_info)
                    st.session_state.history.append({"name": row['product_name'], "eco_rating": row['eco_rating']})
                else:
                    st.error("Product not found in database.")
            else:
                st.warning("No barcode detected in image.")

    # -------- History --------
    with tab3:
        st.title("📜 My Eco History")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            avg_rating = history_df['eco_rating'].astype(float).mean()
            st.success(f"🌍 Your average eco rating is {round(avg_rating, 2)}")
        else:
            st.info("No history found yet.")

    # -------- Personalization --------
    with tab4:
        st.title("⚙️ Personalize Recommendations")
        category_pref = st.selectbox("Preferred product category:", ["All", "Food", "Beverage", "Hygiene", "Plastic Bottles"])

        if category_pref != "All":
            st.subheader(f"🌱 Eco Products in {category_pref}")
            filtered = product_db[(product_db['category'] == category_pref) & (product_db['eco_rating'] >= 3)]
        else:
            st.subheader("🌱 All High Eco Rated Products")
            filtered = product_db[product_db['eco_rating'] >= 3]

        for i, row in filtered.iterrows():
            st.markdown(f"✅ **{row['product_name']}** – Eco Rating: {row['eco_rating']} – Alt: {row['eco_alternative']}")
else:
    st.warning("👤 Please log in to use the app.")
