import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1100px;
}

h1 {
    text-align: center;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #999999;
    margin-bottom: 1.5rem;
}

.upload-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #444;
    background-color: rgba(255,255,255,0.03);
}

.result-box {
    padding: 30px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #444;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🐱 Cat vs Dog Classifier")

st.markdown(
    '<div class="subtitle">'
    'Upload an image and let the CNN model identify it.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_my_model():

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(224, 224, 3)
        )
    )

    model.add(MaxPool2D())

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPool2D())

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPool2D())

    model.add(
        Conv2D(
            128,
            (5, 5),
            activation="relu"
        )
    )

    model.add(MaxPool2D())

    model.add(
        Conv2D(
            256,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPool2D())

    model.add(Flatten())

    model.add(
        Dense(
            2,
            activation="softmax"
        )
    )

    model.load_weights("model.h5")

    return model


model = load_my_model()



left, right = st.columns(2, gap="large")


# ==================================================
# LEFT SIDE
# ==================================================

with left:

    st.subheader("📷 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a cat or dog image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )


# ==================================================
# RIGHT SIDE
# ==================================================

with right:

    st.subheader("🔍 Prediction")

    if uploaded_file is None:

        st.info("Upload an image to start prediction.")

    else:

        st.write("Image ready for prediction.")

        predict_button = st.button(
            "🚀 Predict",
            use_container_width=True
        )

        if predict_button:

            # Convert to RGB
            image = image.convert("RGB")

            # Resize
            image = image.resize((224, 224))

            # NumPy array
            img = np.array(image)

            # Normalize
            img = img / 255.0

            # Batch dimension
            img = np.expand_dims(img, axis=0)

            # Prediction
            prediction = model.predict(img, verbose=0)

            predicted_class = np.argmax(prediction)

            confidence = prediction[0][predicted_class] * 100


            # Display result

            if predicted_class == 0:

                st.success(
                    f"🐱 CAT\n\n"
                    f"Confidence: {confidence:.2f}%"
                )

            else:

                st.success(
                    f"🐶 DOG\n\n"
                    f"Confidence: {confidence:.2f}%"
                )

            # Show probabilities

            st.write("### Prediction Probability")

            st.progress(
                float(prediction[0][0]),
                text=f"🐱 Cat: {prediction[0][0] * 100:.2f}%"
            )

            st.progress(
                float(prediction[0][1]),
                text=f"🐶 Dog: {prediction[0][1] * 100:.2f}%"
            )