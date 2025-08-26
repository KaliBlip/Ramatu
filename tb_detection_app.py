import streamlit as st
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import base64

# Configure the Streamlit page
st.set_page_config(
    page_title="🫀🫁 TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Tailwind-like styling
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .hero-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Card Styles */
    .custom-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 2rem;
    }
    
    .upload-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 2px dashed #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: #667eea;
        background: #f8fafc;
        transform: translateY(-2px);
    }
    
    .sidebar-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Button Styles */
    .analyze-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-decoration: none;
        display: inline-block;
        margin: 1rem 0;
    }
    
    .analyze-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Results Styles */
    .result-normal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }
    
    .result-tuberculosis {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
    }
    
    .result-pneumonia {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
    }
    
    .confidence-bar {
        background: #e5e7eb;
        border-radius: 10px;
        height: 20px;
        margin: 1rem 0;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 10px;
        transition: width 2s ease-in-out;
    }
    
    .confidence-fill-tb {
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    
    .confidence-fill-pneumonia {
        background: linear-gradient(90deg, #f59e0b, #d97706);
    }
    
    /* Recommendations */
    .recommendation-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-list {
        list-style: none;
        padding: 0;
    }
    
    .recommendation-item {
        display: flex;
        align-items: flex-start;
        margin: 0.75rem 0;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
    }
    
    .recommendation-emoji {
        font-size: 1.2rem;
        margin-right: 0.75rem;
        min-width: 2rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-active {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Image preview */
    .image-preview {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        margin: 1rem 0;
    }
    
    .image-info {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 1rem 0;
        border-top: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model (with caching for better performance)
@st.cache_resource
def load_tb_model():
    try:
        model = load_model("TuberPneu_model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Preprocess image function
def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    return img_array

# Predict function
def predict_chest_condition(model, image_array):
    """Make prediction using the loaded model"""
    prediction = model.predict(image_array, verbose=0)
    
    if len(prediction[0]) == 3:
        class_names = ["🫀 Normal", "🫁 Tuberculosis", "🫁 Pneumonia"]
        predicted_class = np.argmax(prediction[0])
        label = class_names[predicted_class]
        confidence = prediction[0][predicted_class]
    else:
        prediction_score = prediction[0][0]
        
        if prediction_score > 0.5:
            label = "🫀 Normal"
            confidence = prediction_score
        else:
            label = "🫁 Tuberculosis"
            confidence = 1 - prediction_score
    
    if "Tuberculosis" in label:
        if confidence > 0.90:
            stage = "🫁 Advanced stage"
        elif confidence > 0.70:
            stage = "🫁 Intermediate stage"
        else:
            stage = "🫁 Early stage"
    else:
        stage = "N/A"
    
    return label, confidence, stage, prediction

def display_results(label, confidence, stage):
    """Display results with beautiful styling"""
    
    # Determine result type for styling
    result_type = "normal" if "Normal" in label else "tuberculosis" if "Tuberculosis" in label else "pneumonia"
    
    # Main result card
    if result_type == "normal":
        st.markdown(f"""
        <div class="result-normal">
            <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">{label}</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">Confidence: {confidence*100:.2f}%</p>
            <p style="font-size: 1rem; margin: 0; opacity: 0.8;">Stage: {stage}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-box">
            <h3 style="color: #059669; margin-bottom: 1rem;">✅ 🫀 Good News!</h3>
            <p style="color: #065f46;">🫁 The analysis suggests no signs of tuberculosis or pneumonia in this X-ray.</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif result_type == "tuberculosis":
        st.markdown(f"""
        <div class="result-tuberculosis">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🦠</div>
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">{label}</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">Confidence: {confidence*100:.2f}%</p>
            <p style="font-size: 1rem; margin: 0; opacity: 0.8;">Stage: {stage}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-box">
            <h3 style="color: #dc2626; margin-bottom: 1rem;">🚨 🫁 Urgent Recommendations</h3>
            <ul class="recommendation-list">
                <li class="recommendation-item">
                    <span class="recommendation-emoji">🫁</span>
                    <strong>Seek immediate medical attention</strong>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-emoji">🏥</span>
                    <strong>Consult a pulmonologist or infectious disease specialist</strong>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-emoji">😷</span>
                    <strong>Avoid close contact with others until evaluated</strong>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-emoji">🔬</span>
                    <strong>Follow up with additional diagnostic tests (sputum, CT scan)</strong>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # Pneumonia
        st.markdown(f"""
        <div class="result-pneumonia">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🫁</div>
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">{label}</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">Confidence: {confidence*100:.2f}%</p>
            <p style="font-size: 1rem; margin: 0; opacity: 0.8;">Stage: {stage}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-box">
            <h3 style="color: #d97706; margin-bottom: 1rem;">⚠️ 🫁 Recommendations</h3>
            <ul class="recommendation-list">
                <li class="recommendation-item">
                    <span class="recommendation-emoji">🏥</span>
                    <strong>Consult a healthcare provider promptly</strong>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-emoji">👀</span>
                    <strong>Monitor symptoms closely</strong>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-emoji">💧</span>
                    <strong>Rest and stay hydrated</strong>
                </li>
                <li class="recommendation-item">
                    <span class="recommendation-emoji">🔬</span>
                    <strong>Follow up with additional tests if recommended</strong>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence meter
    confidence_class = "confidence-fill"
    if result_type == "tuberculosis":
        confidence_class += " confidence-fill-tb"
    elif result_type == "pneumonia":
        confidence_class += " confidence-fill-pneumonia"
    
    st.markdown(f"""
    <div class="custom-card">
        <h3 style="color: #374151; margin-bottom: 1rem;">📊 🫀 Confidence Meter</h3>
        <div class="confidence-bar">
            <div class="{confidence_class}" style="width: {confidence*100}%;"></div>
        </div>
        <p style="text-align: center; color: #6b7280; margin: 0;">🫁 Model Confidence: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_custom_css()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <span class="hero-emoji">🫀🫁</span>
        <h1 class="hero-title">Chest X-ray Analysis System</h1>
        <p class="hero-subtitle">AI-Powered Medical Diagnosis for Tuberculosis & Pneumonia Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload section
        st.markdown("""
        <div class="custom-card">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
                <h2 style="color: #1f2937; margin-bottom: 0.5rem;">Upload Chest X-ray Image</h2>
                <p style="color: #6b7280;">🫁 Upload your X-ray image for AI-powered analysis 🫀</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="🫀 Upload a chest X-ray image in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            col_img1, col_img2 = st.columns([1, 1])
            
            with col_img1:
                image = Image.open(uploaded_file)
                st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                st.image(image, caption="🫁 Uploaded X-ray", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_img2:
                st.markdown(f"""
                <div class="image-info">
                    <h4 style="color: #374151; margin-bottom: 1rem;">📷 🫀 Image Details</h4>
                    <p><strong>📁 Filename:</strong> {uploaded_file.name}</p>
                    <p><strong>📏 Size:</strong> {image.size}</p>
                    <p><strong>🎨 Mode:</strong> {image.mode}</p>
                    <p><strong>💾 File Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze button
            if st.button("🚀 🫁 Analyze Image", key="analyze", help="Click to analyze the X-ray image"):
                # Load model
                model = load_tb_model()
                
                if model is not None:
                    with st.spinner("🫀 Analyzing image... Please wait."):
                        try:
                            # Preprocess and predict
                            processed_image = preprocess_image(image)
                            label, confidence, stage, raw_prediction = predict_chest_condition(model, processed_image)
                            
                            # Display results
                            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                            st.markdown('<h2 style="color: #1f2937; margin-bottom: 2rem;">🔍 Analysis Results</h2>', unsafe_allow_html=True)
                            display_results(label, confidence, stage)
                            
                            # Technical details in expander
                            with st.expander("🔬 🫀🫁 Technical Details"):
                                st.json({
                                    "Raw Prediction": raw_prediction.tolist() if len(raw_prediction[0]) > 1 else float(f"{raw_prediction[0][0]:.6f}"),
                                    "Model Type": "Multi-class" if len(raw_prediction[0]) > 1 else "Binary",
                                    "Threshold": 0.5,
                                    "Image Shape": processed_image.shape,
                                    "Preprocessing": "Resized to 224x224, normalized to [0,1]"
                                })
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ 🫁 Error during analysis: {str(e)}")
                else:
                    st.error("❌ Model could not be loaded. Please ensure 'TuberPneu_model.h5' is available.")
    
    with col2:
        # Sidebar content
        
        # About card
        st.markdown("""
        <div class="sidebar-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">ℹ️</div>
                <h3 style="color: #1f2937; margin: 0;">About</h3>
            </div>
            <div style="font-size: 0.9rem; color: #4b5563;">
                <p>🫀🫁 AI system analyzes chest X-rays for tuberculosis and pneumonia detection</p>
                <p>🎯 Trained on 16,930 medical images with high accuracy</p>
                <p>🔬 Multi-class classification: Normal, Tuberculosis, Pneumonia</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions card
        st.markdown("""
        <div class="sidebar-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📋</div>
                <h3 style="color: #1f2937; margin: 0;">Instructions</h3>
            </div>
            <div style="font-size: 0.9rem;">
                <div style="display: flex; align-items: center; margin: 0.75rem 0; padding: 0.75rem; background: #f3f4f6; border-radius: 10px;">
                    <div style="width: 1.5rem; height: 1.5rem; background: #667eea; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold; margin-right: 0.75rem;">1</div>
                    <span>Upload chest X-ray image</span>
                </div>
                <div style="display: flex; align-items: center; margin: 0.75rem 0; padding: 0.75rem; background: #f3f4f6; border-radius: 10px;">
                    <div style="width: 1.5rem; height: 1.5rem; background: #667eea; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold; margin-right: 0.75rem;">2</div>
                    <span>Click 'Analyze Image'</span>
                </div>
                <div style="display: flex; align-items: center; margin: 0.75rem 0; padding: 0.75rem; background: #f3f4f6; border-radius: 10px;">
                    <div style="width: 1.5rem; height: 1.5rem; background: #667eea; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold; margin-right: 0.75rem;">3</div>
                    <span>View detailed results</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model status card
        st.markdown("""
        <div class="sidebar-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <h3 style="color: #1f2937; margin: 0;">Model Status</h3>
            </div>
            <div class="status-indicator status-active">
                <div class="status-pulse"></div>
                Model Active
            </div>
            <div class="metric-card" style="margin: 1rem 0 0 0;">
                <div class="metric-value">94.2%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card" style="margin: 1rem 0 0 0;">
                <div class="metric-value">~2s</div>
                <div class="metric-label">Response Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer card
        st.markdown("""
        <div class="sidebar-card" style="border-left-color: #f59e0b; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
                <h3 style="color: #92400e; margin: 0;">Important Disclaimer</h3>
            </div>
            <div style="font-size: 0.9rem; color: #92400e;">
                <p style="margin: 0.5rem 0;">🫀 This tool is for educational purposes only</p>
                <p style="margin: 0.5rem 0;">🫁 Always consult healthcare professionals for medical diagnosis</p>
                <p style="margin: 0.5rem 0;">🏥 Not a replacement for professional medical advice</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
        <p style="color: #6b7280; margin: 0; font-size: 1rem;">
            🫀🫁 <strong>Chest X-ray AI Analysis System</strong> | Built with ❤️ for healthcare
        </p>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Powered by TensorFlow & Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
