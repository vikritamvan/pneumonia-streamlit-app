import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image # For opening and processing images
import numpy as np
import os # For checking file existence

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Prediction App", # Title appearing in the browser tab
    page_icon="🩺", # Icon in the browser tab
    layout="centered", # Main content layout: "centered" or "wide"
    initial_sidebar_state="auto" # Sidebar initial state
)

# --- Function to Load Model (with caching) ---
# Using st.cache_resource to load the model only once
# This is important because large models don't need to be reloaded on every UI interaction
@st.cache_resource
def load_pneumonia_model():
    model_path = "pneumonia_detection_model.keras" # Your model file name
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Please ensure the model is in the same directory as 'app.py'.")
        st.stop() # Stop the application if the model is not found

    # Load Keras model. custom_objects is required for F1Score metric
    try:
        model = keras.models.load_model(
            model_path,
            # Ensure the F1Score definition in Python is exactly the same as used during training
            custom_objects={'f1_score': keras.metrics.F1Score(threshold=0.5)}
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop() # Stop the application if the model fails to load

# Load the model when the application starts (this will happen only once thanks to @st.cache_resource)
model = load_pneumonia_model()

# --- Image Preprocessing Function ---
# This function must mimic the preprocessing you performed in Python before training.
# Augmentation layers are NOT NECESSARY to apply during prediction.
# Only Rescaling (if any) and format conversion are needed.
def preprocess_image(image_file, target_size=(224, 224)):
    # Open the image using PIL (Pillow)
    img = Image.open(image_file)
    
    # Convert to grayscale if not already (X-rays are often grayscale)
    # If mode is 'L' (grayscale) or 'LA' (grayscale + alpha), leave it.
    # If RGB/RGBA, convert to grayscale first.
    if img.mode not in ('L', 'LA'):
        img = img.convert('L') # Convert to grayscale

    # Resize the image using Lanczos algorithm for high quality
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convert PIL image to a NumPy array with float32 data type
    img_array = np.array(img, dtype=np.float32)

    # If the image has 1 channel (grayscale), duplicate it to 3 channels (RGB)
    # Your model expects input (224, 224, 3)
    if img_array.ndim == 2: # If the array is only (height, width)
        img_array = np.stack([img_array, img_array, img_array], axis=-1) # Becomes (height, width, 3)
    elif img_array.shape[-1] == 4: # If there's an alpha channel (RGBA), discard alpha
        img_array = img_array[:, :, :3]
    
    # Normalization: Your model has Rescaling(1./127.5, offset=-1) as the first layer after input.
    # This means the input image is expected to be in the range [0, 255] and the model will normalize it itself.
    # So, we don't need manual normalization here if img_array is already in [0, 255].
    # PIL Image.open() and np.array() will produce 0-255 values for standard images.

    # Add batch dimension (for a single image)
    img_array = np.expand_dims(img_array, axis=0) # Becomes (1, height, width, channels)

    return img_array

# --- Sidebar for Page Navigation ---
st.sidebar.title("Navigation")
# Create radio buttons in the sidebar to select pages
page = st.sidebar.radio("Select Page", ["Home", "Prediction"])

# --- Page Content Based on Sidebar Selection ---

if page == "Home":
    # Using raw HTML to center the main title
    st.markdown(f'<div style="text-align: center;"><h1><b>Pneumonia Prediction System from X-Ray Images</b></h1></div>', unsafe_allow_html=True)
    
    # Inventor Title with underline and centered
    st.markdown("<h3 style='text-align: center;'><u>Inventors :</u></h3>", unsafe_allow_html=True)
    
    # List of Inventors without bullet points, centered
    st.markdown("<p style='text-align: center;'>Puspita Kartikasari, S.Si., M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Prof. Dr. Rukun Santoso, M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Dra. Suparti, M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Rizwan Arisandi, S.Si., M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Vikri Haikal</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) # Empty line for separation
    
    # University and Year Details, centered
    st.markdown("<h3 style='text-align: center;'>DEPARTMENT OF STATISTICS</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>FACULTY OF SCIENCE AND MATHEMATICS</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>DIPONEGORO UNIVERSITY</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>YEAR 2025</h3>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) # Empty line
    
    # Application description
    st.write("This application uses a Deep Learning model trained with Transfer Learning techniques to predict whether a lung X-ray image shows signs of Pneumonia or is Normal.")
    st.write("Navigate to the 'Prediction' page in the sidebar to upload an image and get the results.")

elif page == "Prediction":
    st.title("Pneumonia Prediction")

    st.write("Upload a lung X-ray image (JPG/PNG format) to get a prediction.")
    # Widget for uploading files
    uploaded_file = st.file_uploader("Upload Lung X-Ray Image", type=["jpg", "jpeg", "png"])

    # Only display this section if a file is uploaded
    if uploaded_file is not None:
        st.subheader("Uploaded Image:")
        # Display the uploaded image
        st.image(uploaded_file, caption='Your X-Ray Image', use_column_width=True)

        # Button to trigger prediction
        if st.button("Perform Prediction"):
            # Display loading spinner
            with st.spinner("Performing prediction..."):
                try:
                    # Preprocess the image
                    img_for_pred = preprocess_image(uploaded_file)
                    
                    # Perform prediction using the model
                    # The output of model.predict() is an array, get its scalar value
                    prediction = model.predict(img_for_pred)[0][0] 
                    
                    # Interpret prediction results
                    probability_pneumonia = float(prediction) # Ensure float data type
                    
                    threshold = 0.5 # Threshold for classification

                    st.subheader("Prediction Results:")
                    st.write(f"Probability of Pneumonia: {probability_pneumonia * 100:.2f}%")

                    st.subheader("Interpretation:")
                    if probability_pneumonia >= threshold:
                        # Interpretation text with red color for Pneumonia
                        st.markdown(
                            f"<p style='color:red;'><b>Pneumonia</b></p>"
                            f"The model identifies the possibility of pneumonia with a probability of "
                            f"{probability_pneumonia * 100:.2f}%.",
                            unsafe_allow_html=True
                        )
                    else:
                        # Interpretation text with green color for Normal
                        st.markdown(
                            f"<p style='color:green;'><b>Normal</b></p>"
                            f"The model does not identify signs of pneumonia (pneumonia probability "
                            f"{probability_pneumonia * 100:.2f}%).",
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    # Catch and display error if it occurs during prediction
                    st.error(f"An error occurred during prediction: {e}")
