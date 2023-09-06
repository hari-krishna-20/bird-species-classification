import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub

# Function to make predictions
def predict(image_path, model):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    max_index = np.argmax(result)
    return class_labels[max_index], result[0][max_index]

# Streamlit app
st.title("Bird Species Classifier")

# Load your saved model path
model_path = "model.h5"

# Use keras.utils.custom_object_scope to load the model
with st.spinner("Loading model..."):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Define the class labels
class_labels = [
    "ABBOTTS BABBLER", "ABBOTTS BOOBY", "ABYSSINIAN GROUND HORNBILL",
    "AFRICAN CROWNED CRANE", "AFRICAN EMERALD CUCKOO", "AFRICAN FIREFINCH",
    "AFRICAN OYSTER CATCHER", "AFRICAN PIED HORNBILL", "AFRICAN PYGMY GOOSE", "ALBATROSS"
]

# Upload an image
uploaded_image = st.file_uploader("Upload a bird image...", type=["jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        prediction, confidence = predict(uploaded_image, model)
        st.write(f"Predicted Bird Species: {prediction}")
        st.write(f"Confidence: {confidence:.2%}")
