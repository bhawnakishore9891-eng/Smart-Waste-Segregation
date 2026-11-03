
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="centered"
)

# Green and White Theme
st.markdown("""
<style>
    .stApp {
        background-color: #F4FAF4;
        color: #20352A;
    }

    h1, h2, h3 {
        color: #176B3A !important;
    }

    p, label {
        color: #304B39;
    }

    /* Upload section */
    [data-testid="stFileUploader"] {
        background-color: #E5F3E8;
        border: 1px dashed #218838;
        border-radius: 12px;
        padding: 12px;
    }

    /* Browse Files button */
    [data-testid="stFileUploader"] button {
        background-color: #218838 !important;
        color: white !important;
        border: none !important;
    }

    /* Uploaded filename */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderFileName"] {
        color: #20352A !important;
    }

    /* Predict button */
    .stButton > button,
    .stButton > button p {
        background-color: #218838 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #176B3A !important;
        color: #FFFFFF !important;
    }

    /* Uploaded file remove button */
    [data-testid="stFileUploader"] button[kind="tertiary"] {
        color: #176B3A !important;
    }

    /* Confidence metric */
    [data-testid="stMetric"] {
        background-color: #E5F3E8;
        padding: 15px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #176B3A !important;
    }

    [data-testid="stMetricLabel"] {
        color: #304B39 !important;
    }

    /* Result card */
    .result-card {
        background-color: #E5F3E8;
        border-left: 5px solid #218838;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }

    .tip-card {
        background-color: white;
        border: 1px solid #C7DEC9;
        padding: 15px;
        border-radius: 10px;
        margin-top: 12px;
        color: #20352A;
    }

    hr {
        border-color: #C7DEC9;
    }
</style>
""", unsafe_allow_html=True)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "waste_classifier_mobilenetv2.keras"


# Waste categories
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# Disposal suggestions
disposal_tips = {
    "cardboard": "Keep cardboard clean and dry. Send it for paper recycling.",
    "glass": "Separate glass items and place them in a designated glass recycling bin.",
    "metal": "Collect metal cans separately for metal recycling.",
    "paper": "Keep paper clean and dry before sending it for recycling.",
    "plastic": "Check the recycling symbol and local recycling guidelines.",
    "trash": "Place non-recyclable waste in the appropriate general waste bin."
}


# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# App heading
st.title("♻️ Smart Waste Segregation")

st.markdown(
    "### Make waste sorting smarter and easier! 🌱"
)

st.write(
    "Upload an image of waste and let our AI model "
    "identify its category and suggest proper disposal."
)

st.divider()


# Image upload
st.subheader("📸 Upload Waste Image")

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Your Uploaded Waste Image",
        width=300
    )

    if st.button("♻️ Predict Waste Category"):

        # Image preprocessing
        image_resized = image.resize((160, 160))

        image_array = np.array(image_resized)

        image_array = np.expand_dims(image_array, axis=0)

        image_array = preprocess_input(
            image_array.astype("float32")
        )

        # Model prediction
        predictions = model.predict(image_array, verbose=0)

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = float(
            predictions[0][predicted_index]
        ) * 100

        # Display prediction
        st.markdown(
            f"""
            <div class="result-card">
                <h3>✅ Predicted Category: {predicted_class.title()}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Disposal suggestion
        st.subheader("🌿 Disposal Suggestion")

        st.markdown(
            f"""
            <div class="tip-card">
                {disposal_tips[predicted_class]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Note: AI predictions may not always be correct. "
            "Please verify waste type before disposal."
        )


# Footer
st.divider()

st.markdown(
    "<center>♻️ Smart Waste Segregation | "
    "AI-Based Waste Classification</center>",
    unsafe_allow_html=True
)
