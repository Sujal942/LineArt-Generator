import streamlit as st
import cv2
import numpy as np
import svgwrite
from PIL import Image
import tempfile
import os
import base64

def preprocess_image(image):
    """Convert image to a clean binary edge map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection (Tuned for cleaner lines)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    return edges

def extract_contours(edge_image):
    """Extracts continuous contours from the edge image."""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contours_to_svg(contours, width, height, output_svg):
    """Convert contours to an SVG file with smooth lines."""
    dwg = svgwrite.Drawing(output_svg, size=(f"{width}px", f"{height}px"), profile='tiny')

    # ✅ Add a white background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    for contour in contours:
        if len(contour) > 10:  # Ignore tiny details for cleaner output
            path_data = "M " + " ".join(f"{p[0][0]},{p[0][1]}" for p in contour)
            dwg.add(dwg.path(d=path_data, stroke="black", fill="none", stroke_width=1))

    dwg.save()

def get_svg_base64(svg_path):
    """Convert SVG file to base64 for inline display in Streamlit."""
    with open(svg_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Streamlit UI
st.set_page_config(page_title="Single-Line Art Generator", layout="wide")

st.title("✏️ Line Art Generator")
st.write("Upload an image and get a **clean SVG line art**.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)  # ✅ Equal column size for image and SVG

    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Process image to extract edges
    edge_image = preprocess_image(image_cv)

    # Extract contours from edges
    contours = extract_contours(edge_image)

    # Create temporary file for SVG output
    temp_dir = tempfile.gettempdir()
    svg_path = os.path.join(temp_dir, "lineart.svg")

    # Convert contours to SVG
    h, w = edge_image.shape
    contours_to_svg(contours, w, h, svg_path)

    with col2:
        st.subheader("🖼️ SVG Line Art Preview")
        svg_base64 = get_svg_base64(svg_path)

        # ✅ Force white background by setting display div style
        svg_html = f'''
        <div style="background-color:white; padding:10px; border-radius:10px;">
            <img src="data:image/svg+xml;base64,{svg_base64}" width="100%" height="750px" />
        </div>
        '''
        st.markdown(svg_html, unsafe_allow_html=True)

    # Provide download button
    st.download_button(label="⬇️ Download SVG", data=open(svg_path, "rb"), file_name="single_line_art.svg", mime="image/svg+xml")

    st.success("✅ SVG generated successfully! Click the button to download.")
