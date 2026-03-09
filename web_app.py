import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(page_title="Pneumonia Classification AI", layout="centered")
st.title("Automated Classification of Pneumonia in Medical Radiography", anchor=False)
st.write("AI Screening Tool for High-Risk Case Prioritization")

@st.cache_resource
def load_model():
    return YOLO('yolov26_best.pt') 

model = load_model()
classes = ['Bacterial', 'Normal', 'Viral']

if 'predicted' not in st.session_state:
    st.session_state.predicted = False

uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])
results_container = st.container()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Radiograph', width=400)
    button_placeholder = st.empty()

    if not st.session_state.predicted:
        if button_placeholder.button("Submit for Analysis", use_container_width=True):
            with st.spinner('Analyzing Radiograph... Please wait.'):
                results = model(image)
                time.sleep(0.5)
                
                st.session_state.results = results
                st.session_state.predicted = True
                st.rerun()

    if st.session_state.predicted:
        res = st.session_state.results[0]
        probs = res.probs
        top1_idx = int(probs.top1)
        
        with results_container:
            st.success(f"Analysis Complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", classes[top1_idx])
            with col2:
                confidence = probs.top1conf.item() * 100
                st.metric("Confidence", f"{confidence:.2f}%")

            st.write("### Probability Distribution")
            for i, class_name in enumerate(classes):
                score = probs.data[i].item()
                st.progress(score, text=f"{class_name}: {score*100:.2f}%")
            
            st.divider()

        if button_placeholder.button("Clear and Predict Again", use_container_width=True):
            st.session_state.predicted = False
            st.session_state.results = None
            st.rerun()
else:
    st.session_state.predicted = False
    st.session_state.results = None