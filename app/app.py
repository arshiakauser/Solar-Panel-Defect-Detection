# -*- coding: utf-8 -*-
# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

# --------------------------
# App Configuration
# --------------------------
st.set_page_config(page_title="YOLOv9 Object Detection", layout="wide")
st.title("⚡ Solar Panel Defect Detection System")

st.markdown(
    """
    This application uses a **YOLOv9 model** to detect and classify solar panel defects.
    Upload an image to visualize detected regions and view class-wise counts.
    """
)

# --------------------------
# Helper Functions
# --------------------------
@st.cache_resource
def load_model(model_path):
    """Load YOLO model and cache it for faster performance"""
    return YOLO(model_path)

def process_image(image, model, conf_threshold=0.5):
    """Convert image and perform object detection"""
    # Convert PIL to OpenCV BGR format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Run YOLO inference
    results = model.predict(image_bgr, conf=conf_threshold, imgsz=640)
    
    # Plot results on image
    results_img = results[0].plot()
    
    # Get class names and counts
    names = model.names  # Class label names
    boxes = results[0].boxes

    if boxes and len(boxes.cls) > 0:
        cls_indices = boxes.cls.cpu().numpy().astype(int)
        cls_counts = pd.Series(cls_indices).value_counts().sort_index()
        summary = {names[i]: int(cls_counts[i]) for i in cls_counts.index}
    else:
        summary = {}

    return results_img, summary


# --------------------------
# Load Model
# --------------------------
# Use relative path (place the model in the same folder or adjust as needed)
model_path = os.path.join(r"C:\Users\DELL\Downloads\360digitmg_Project1\Project_212_New_Arshia\Project_workingcode\yolov9model-20240902T192839Z-001\yolov9model", "best.pt")
if not os.path.exists(model_path):
    st.error(f"Model not found at {model_path}")
    st.stop()

model = load_model(model_path)

# --------------------------
# File Uploader
# --------------------------
uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Run detection with spinner
        with st.spinner("Processing image..."):
            results_img, class_summary = process_image(image, model, conf_threshold=0.5)
        
        # Display side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='📸 Uploaded Image', use_column_width=True)
        with col2:
            st.image(results_img, caption='🎯 Detected Objects', use_column_width=True)
        
        # --------------------------
        # Display Class Count Summary
        # --------------------------
        st.subheader("📊  Detection Summary")

        if class_summary:
            df_summary = pd.DataFrame(list(class_summary.items()), columns=["Class", "Count"])
            st.table(df_summary)
        else:
            st.info("No objects detected above confidence threshold.")
    
    except Exception as e:
        st.error(f"Error processing image: {e}")

