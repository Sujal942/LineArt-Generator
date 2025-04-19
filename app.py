import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64

def convert_to_lineart(image, low_threshold, high_threshold, blur_kernel, line_thickness):
    # Convert PIL to OpenCV
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Blur to remove noise
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Optional: Dilate to make edges thicker
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=line_thickness)

    # Invert to get black lines on white background
    inverted = cv2.bitwise_not(edges_dilated)

    # Convert to RGB for Streamlit
    result = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
    return result

# Convert image to downloadable link
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img_pil = Image.fromarray(img)
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🖊️ Line Art Converter")

# Sidebar Settings
st.sidebar.header("Parameters")
low_threshold = st.sidebar.slider("Canny Low Threshold", 10, 200, 50, step=5)
high_threshold = st.sidebar.slider("Canny High Threshold", 50, 300, 150, step=5)
blur_kernel = st.sidebar.slider("Blur Kernel", 1, 11, 3, step=2)
line_thickness = st.sidebar.slider("Line Thickness", 1, 5, 1)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    original = Image.open(uploaded_file).convert("RGB")
    lineart_img = convert_to_lineart(original, low_threshold, high_threshold, blur_kernel, line_thickness)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original, width=400)
    with col2:
        st.subheader("Line Art")
        st.image(lineart_img, width=400)

    st.markdown(get_image_download_link(lineart_img, "line_art.png", "📥 Download Line Art"), unsafe_allow_html=True)
else:
    st.info("Upload an image using the sidebar to begin.")
