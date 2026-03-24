import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NeuroTwin-AI", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn():
    return load_model("models/detection_model/brain_tumor_cnn.keras")

model = load_cnn()

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    if img.shape[-1] == 3:
        img = np.mean(img, axis=-1, keepdims=True)

    img = img / 255.0
    img = img.reshape(1, 224, 224, 1)

    return img

def predict(img):
    return model.predict(img)[0]

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 NeuroTwin-AI")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "MRI Diagnosis", "Tumor Insights", "AI Reasoning", "Risk & Analytics", "System Info"]
)

# ==================================================
# ✅ OVERVIEW (UPDATED ONLY THIS)
# ==================================================
if page == "Overview":
    st.title("🧠 NeuroTwin-AI")

    st.subheader("AI-Powered Brain Tumor Detection System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "88.5%")
    col2.metric("Tumor Types", "4")
    col3.metric("Prediction Time", "< 1 sec")

    st.divider()

    st.markdown("""
### 📌 About the Project
NeuroTwin-AI is a deep learning-based system that analyzes MRI scans to detect brain tumors.

It uses a **Convolutional Neural Network (CNN)** trained on labeled MRI datasets to classify images into:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

### 🧠 Why This Matters
Brain tumors are critical conditions where early detection can save lives.  
AI helps assist doctors by:
- Faster diagnosis  
- Pattern recognition  
- Supporting decision-making  

### ⚙️ Features
✔ MRI Image Analysis  
✔ Tumor Classification  
✔ Confidence Scoring  
✔ Explainable AI  


""")

# ==================================================
# MRI DIAGNOSIS (UNCHANGED)
# ==================================================
elif page == "MRI Diagnosis":
    st.header("MRI Diagnosis")

    uploaded = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)

        processed = preprocess(img)
        preds = predict(processed)

        pred_class = CLASSES[np.argmax(preds)]
        confidence = np.max(preds) * 100

        st.success(f"Prediction: {pred_class}")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(CLASSES, preds)
        ax1.set_ylim(0, 1)
        plt.xticks(rotation=15)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax2.pie(
            preds,
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
            startangle=90
        )
        ax2.legend(wedges, CLASSES, loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.axis("equal")
        st.pyplot(fig2)

# ==================================================
# ✅ TUMOR INSIGHTS (UPDATED ONLY THIS)
# ==================================================
elif page == "Tumor Insights":
    st.header("🧠 Tumor Insights")

    st.subheader("Types of Brain Tumors")

    st.markdown("""
### 🧬 Glioma
- Develops from glial cells  
- Can be aggressive  
- Common symptoms: headaches, seizures  

### 🧠 Meningioma
- Usually benign  
- Forms in brain covering  
- Slow growing  

### 🧪 Pituitary Tumor
- Affects hormone levels  
- Located in pituitary gland  
- May cause vision issues  

### ✅ No Tumor
- Normal MRI scan  
- No abnormal growth detected  
""")

    st.subheader("📊 Key Differences")

    st.markdown("""
| Feature | Glioma | Meningioma | Pituitary |
|--------|--------|------------|-----------|
| Nature | Aggressive | Benign | Hormonal |
| Growth | Fast | Slow | Moderate |
| Risk | High | Medium | Medium |
""")

    st.info("💡 Early detection significantly improves treatment outcomes.")

# ==================================================
# ✅ AI REASONING (EXPANDED ONLY THIS)
# ==================================================
elif page == "AI Reasoning":
    st.header("AI Explanation & Knowledge")

    QUESTIONS = {
        "What is brain tumor?": "Abnormal growth of cells in brain.",
        "Types of tumor?": "Glioma, Meningioma, Pituitary.",
        "What is glioma?": "Tumor from glial cells.",
        "What is meningioma?": "Tumor from brain covering.",
        "What is pituitary tumor?": "Hormonal tumor.",
        "What is no tumor?": "Normal condition.",
        "How AI works?": "CNN detects patterns.",
        "What is accuracy?": "Correct prediction rate.",
        "What is confidence?": "Probability score.",
        "Can AI replace doctors?": "No.",
        "What is MRI?": "Brain imaging technique.",
        "Why grayscale?": "Focus on structure.",
        "What is CNN?": "Deep learning model.",
        "What is false positive?": "Wrong detection.",
        "What is false negative?": "Missed tumor.",
        "What is dataset bias?": "Unbalanced data.",
        "Can image quality affect result?": "Yes.",
        "Is model perfect?": "No.",
        "How model trained?": "Using MRI dataset.",
        "What is probability graph?": "Confidence distribution.",
        "What is classification?": "Label prediction.",
        "What is overfitting?": "Memorizing data.",
        "What is underfitting?": "Poor learning.",
        "Is AI safe?": "With supervision.",
        "Can AI assist doctors?": "Yes."
    }

    cols = st.columns(3)
    i = 0
    selected = None

    for q in QUESTIONS:
        if cols[i].button(q):
            selected = q
        i = (i + 1) % 3

    if selected:
        st.success(QUESTIONS[selected])

# ==================================================
# RISK & ANALYTICS (UNCHANGED)
# ==================================================
elif page == "Risk & Analytics":
    st.header("⚠ Risk Awareness & AI Confidence")

    st.warning("AI predictions are not a substitute for professional medical diagnosis.")

    st.subheader("🔍 Understanding AI Risk")

    st.markdown("""
AI-based medical systems are powerful but not perfect. The following risks must be considered:

• **False Positive** → AI predicts tumor when none exists  
• **False Negative** → AI fails to detect actual tumor  
• **Overconfidence** → High confidence does NOT guarantee correctness  
• **Data Dependency** → Model depends on training dataset quality  
""")

    st.subheader("📊 Confidence Interpretation")

    st.markdown("""
| Confidence Range | Meaning |
|-----------------|--------|
| 90% – 100% | Strong prediction |
| 70% – 90% | Moderate confidence |
| Below 70% | Low reliability |
""")

    st.subheader("🧠 Clinical Limitations")

    st.markdown("""
• Cannot replace radiologists  
• Cannot detect rare/unseen tumor types  
• Sensitive to image quality and noise  
• Requires human verification  
""")

    st.subheader("⚠ Important Disclaimer")

    st.error("""
This system is built for:
✔ Research  
✔ Learning  
✔ Demonstration  

❌ NOT for real-world medical diagnosis  
""")

# ==================================================
# SYSTEM INFO (UNCHANGED)
# ==================================================
elif page == "System Info":
    st.header("⚙ System Information")

    st.subheader("🧠 Model Details")

    st.markdown("""
• **Model Type:** Convolutional Neural Network (CNN)  
• **Input Size:** 224 x 224 pixels  
• **Color Mode:** Grayscale  
• **Classes:** 4 (Glioma, Meningioma, Pituitary, No Tumor)  
• **Accuracy:** ~88.5%  
""")

    st.subheader("⚙ Technology Stack")

    st.markdown("""
• **Frontend:** Streamlit  
• **Backend:** Python  
• **Deep Learning:** TensorFlow / Keras  
• **Visualization:** Matplotlib  
""")

    st.subheader("🚀 Performance")

    st.markdown("""
• Fast prediction (< 1 second)  
• Lightweight model  
• Runs locally without GPU  
""")

    st.subheader("📦 Features")

    st.markdown("""
✔ MRI Image Classification  
✔ Tumor Type Prediction  
✔ Confidence Score Visualization  
✔ Explainable AI System  
✔ Risk Awareness Module  
""")

    st.subheader("🔒 Limitations")

    st.markdown("""
• Depends on dataset quality  
• Not trained on all tumor types  
• Not medically certified  
• Should be used with expert supervision  
""")