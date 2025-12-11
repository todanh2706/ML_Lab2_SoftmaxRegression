import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

from utils.model import SoftmaxRegression
from utils.features import FeatureExtractor

# --- CONFIGURATION ---
ST_CONFIG = {
    "page_title": "MNIST Digit Recognition",
    "page_icon": "🔢",
    "layout": "wide",
}

# --- UTILS ---
@st.cache_resource
def load_trained_model(feature_name):
    """
    Loads the trained model from .npz file
    """
    # Look for model in current directory or src/ directory
    possible_paths = [
        f"model_{feature_name.lower()}.npz",
        os.path.join("src", f"model_{feature_name.lower()}.npz"),
        os.path.join("models", f"model_{feature_name.lower()}.npz")
    ]
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        return None
        
    return SoftmaxRegression.load(model_path)

def preprocess_image(image_pil):
    """
    Convert PIL image to (28, 28) numpy array, grayscale, inverted if needed.
    Returns: np.array (28, 28) normalized [0, 1]
    """
    # Convert to grayscale
    img = image_pil.convert('L')
    img_arr = np.array(img).astype(np.float32)

    # Invert colors? 
    # MNIST digits are White on Black background.
    # If user draws Black on White (high mean), we need to invert.
    if img_arr.mean() > 127:
        img_arr = 255.0 - img_arr
        # Update PIL image from inverted array for further processing
        img = Image.fromarray(img_arr.astype(np.uint8))

    # --- MNIST-like Preprocessing ---
    # 1. Crop to bounding box
    # Get the bounding box of the non-zero regions
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # 2. Resize to fit in 20x20 box (preserving aspect ratio)
    # MNIST digits are typically fit into a 20x20 box centered in 28x28
    target_size = 20
    w, h = img.size
    ratio = min(target_size / w, target_size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. Compute Center of Mass (CoM) of the resized image
    # We want to place this image on the 28x28 canvas such that its CoM lands at (14, 14).
    
    # Calculate CoM of the resized image (img)
    img_arr_resized = np.array(img).astype(np.float32)
    # CoM calculation: Sum(coords * intensity) / Sum(intensity)
    y_idxs, x_idxs = np.indices(img_arr_resized.shape)
    total_weight = img_arr_resized.sum()
    
    if total_weight > 0:
        com_y = np.sum(y_idxs * img_arr_resized) / total_weight
        com_x = np.sum(x_idxs * img_arr_resized) / total_weight
    else:
        # Fallback if empty image
        com_y, com_x = new_h / 2, new_w / 2

    # Target CoM in the final specific 28x28 image is (14, 14) (or 13.5, 13.5 0-indexed)
    # Let's say center is 13.5
    target_cy, target_cx = 13.5, 13.5
    
    # Calculate paste position (top-left)
    # The CoM relative to the top-left of the inserted image is (com_y, com_x)
    # We want top_left_y + com_y = target_cy  => top_left_y = target_cy - com_y
    paste_y = int(round(target_cy - com_y))
    paste_x = int(round(target_cx - com_x))
    
    # Create blank canvas and paste
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, (paste_x, paste_y))
    
    img_arr = np.array(new_img).astype(np.float32)

    # 4. Clip values just in case
    img_arr = np.clip(img_arr, 0, 255)

    return img_arr

# --- MAIN APP ---
def main():
    st.set_page_config(**ST_CONFIG)
    
    st.title("Handwritten Digit Recognition")
    st.markdown("""
    Draw a digit or upload an image to see the **Softmax Regression** model predict it!
    """)

    # --- SIDEBAR ---
    st.sidebar.header("Settings")
    
    # Model Selection
    feature_type = st.sidebar.selectbox(
        "Feature Extraction Method",
        ("PIXEL", "BLOCK_AVG", "EDGE")
    )
    
    model = load_trained_model(feature_type)
    
    if model is None:
        st.error(f"⚠️ Model for **{feature_type}** not found. Please run `main.py` first to train the models!")
        st.stop()
        
    st.sidebar.success(f"✅ Model **{feature_type}** loaded!")
    
    # Feature Extractor mapping
    extractors = {
        "PIXEL": FeatureExtractor.get_pixel_features,
        "BLOCK_AVG": FeatureExtractor.get_block_features,
        "EDGE": FeatureExtractor.get_edge_features
    }
    extract_func = extractors[feature_type]

    # --- MAIN CONTENT ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        tab1, tab2 = st.tabs(["🖌️ Draw", "📤 Upload"])
        
        final_image = None
        
        with tab1:
            # Session state for canvas key to allow reset
            if "canvas_key" not in st.session_state:
                st.session_state["canvas_key"] = 0

            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="black",  # Fixed fill color with some opacity
                stroke_width=12,     # Thinner stroke 
                stroke_color="white",# White on black like MNIST
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_key}",
            )
            
            if canvas_result.image_data is not None:
                # Get the alpha channel or convert to grayscale
                # canvas_result.image_data is RGBA
                img_data = canvas_result.image_data.astype(np.uint8)
                # Check if the user drew anything (not just all zeros/background)
                if img_data.sum() > 0:
                     final_image = Image.fromarray(img_data).convert("L")

        with tab2:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                final_image = Image.open(uploaded_file)
        
        if st.button("Clear / Reset"):
            st.session_state["canvas_key"] += 1
            st.rerun()

    with col2:
        st.subheader("Prediction")
        
        if final_image is not None:
            # 1. Preprocess
            processed_img_arr = preprocess_image(final_image)
            
            # Display what the model "sees"
            st.image(processed_img_arr, caption="Processed Image (28x28)", width=150, clamp=True)
            
            # 2. Extract Features
            # extraction expects (N, 28, 28), so expand dims
            input_batch = processed_img_arr[np.newaxis, ...]
            
            try:
                X_feat = extract_func(input_batch)
                
                # 3. Predict
                probs = model.predict_proba(X_feat)[0] # get first sample
                pred_label = np.argmax(probs)
                confidence = probs[pred_label]
                
                # 4. Results
                st.metric(label="Predicted Digit", value=str(pred_label), delta=f"{confidence*100:.2f}% Confidence")
                
                # 5. Chart
                fig, ax = plt.subplots(figsize=(6, 3))
                digits = np.arange(10)
                bars = ax.bar(digits, probs, color="#4CAF50")
                ax.set_xticks(digits)
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1.1)
                ax.set_title("Probability Distribution")
                
                # Highlight the max
                bars[pred_label].set_color("#FF5722")
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
                
        else:
            st.info("Please draw a digit or upload an image to start.")

if __name__ == "__main__":
    main()
