import streamlit as st
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Configure the Streamlit page
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide"
)

# Load the model (with caching for better performance)
@st.cache_resource
def load_tb_model():
    try:
        model = load_model("TuberPneu_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image function
def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image_resized = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    return img_array

# Predict function
def predict_chest_condition(model, image_array):
    """Make prediction using the loaded model"""
    prediction = model.predict(image_array, verbose=0)
    
    # Check if it's a multi-class model (3 outputs) or binary model (1 output)
    if len(prediction[0]) == 3:
        # Multi-class model: [Normal, Tuberculosis, Pneumonia]
        class_names = ["Normal", "Tuberculosis", "Pneumonia"]
        predicted_class = np.argmax(prediction[0])
        label = class_names[predicted_class]
        confidence = prediction[0][predicted_class]
    else:
        # Binary model: single output
        prediction_score = prediction[0][0]
        
        # Determine label and confidence based on your corrected logic
        if prediction_score > 0.5:
            label = "Normal"
            confidence = prediction_score
        else:
            label = "Tuberculosis"
            confidence = 1 - prediction_score
    
    # Determine TB stage if tuberculosis is detected
    if label == "Tuberculosis":
        if confidence > 0.90:
            stage = "Advanced stage"
        elif confidence > 0.70:
            stage = "Intermediate stage"
        else:
            stage = "Early stage"
    else:
        stage = "N/A"
    
    return label, confidence, stage, prediction

# Main app
def main():
    # Header
    st.title("🫁 Chest X-ray Analysis System")
    st.markdown("Upload a chest X-ray image to detect tuberculosis and pneumonia using AI")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This AI system analyzes chest X-ray images to detect tuberculosis and pneumonia.
        
        **Instructions:**
        1. Upload a chest X-ray image (JPG, PNG, JPEG)
        2. Click 'Analyze Image' to get prediction
        3. View results with confidence score and staging
        
        **Model Info:**
        - Trained on 16,930 images
        - Categories: Normal, Tuberculosis, Pneumonia
        - Multi-class or Binary classification supported
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational purposes only. 
        Always consult healthcare professionals for medical diagnosis.
        """)
    
    # Load model
    model = load_tb_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please ensure 'TuberPneu_model.h5' is available.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.header("📁 Upload Chest X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
            
            # Image info
            st.info(f"""
            **Image Details:**
            - Filename: {uploaded_file.name}
            - Size: {image.size}
            - Mode: {image.mode}
            """)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Analyze button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... Please wait."):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        label, confidence, stage, raw_prediction = predict_chest_condition(model, processed_image)
                        
                        # Display results
                        if label == "Tuberculosis":
                            st.error(f"🦠 **Detection: {label}**")
                            st.error(f"📊 **Confidence: {confidence*100:.2f}%**")
                            st.error(f"🏥 **Stage: {stage}**")
                            
                            # Additional recommendations for TB cases
                            st.markdown("### 🚨 Recommendations:")
                            st.markdown("""
                            - **Seek immediate medical attention**
                            - **Consult a pulmonologist or infectious disease specialist**
                            - **Avoid close contact with others until evaluated**
                            - **Follow up with additional diagnostic tests (sputum, CT scan)**
                            """)
                            
                        elif label == "Pneumonia":
                            st.warning(f"🫁 **Detection: {label}**")
                            st.warning(f"📊 **Confidence: {confidence*100:.2f}%**")
                            st.warning(f"🏥 **Stage: {stage}**")
                            
                            # Additional recommendations for Pneumonia cases
                            st.markdown("### ⚠️ Recommendations:")
                            st.markdown("""
                            - **Consult a healthcare provider promptly**
                            - **Monitor symptoms closely**
                            - **Rest and stay hydrated**
                            - **Follow up with additional tests if recommended**
                            - **Seek emergency care if breathing difficulties worsen**
                            """)
                            
                        else:  # Normal
                            st.success(f"✅ **Detection: {label}**")
                            st.success(f"📊 **Confidence: {confidence*100:.2f}%**")
                            st.success(f"🏥 **Stage: {stage}**")
                            
                            st.markdown("### ✅ Good News!")
                            st.markdown("The analysis suggests no signs of tuberculosis or pneumonia in this X-ray.")
                        
                        # Confidence meter
                        st.markdown("### 📊 Confidence Meter")
                        st.progress(float(confidence), text=f"Model Confidence: {confidence*100:.1f}%")
                        
                        # Technical details
                        with st.expander("🔬 Technical Details"):
                            st.json({
                                "Raw Prediction": raw_prediction.tolist() if len(raw_prediction[0]) > 1 else float(f"{raw_prediction[0][0]:.6f}"),
                                "Model Type": "Multi-class" if len(raw_prediction[0]) > 1 else "Binary",
                                "Threshold": 0.5,
                                "Image Shape": processed_image.shape,
                                "Preprocessing": "Resized to 224x224, normalized to [0,1]"
                            })
                            
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")

if __name__ == "__main__":
    main()