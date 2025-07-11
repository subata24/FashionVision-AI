import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 🎨 Page config
st.set_page_config(page_title="FashionVision AI 👗", page_icon="🛍️", layout="centered")

# 💅 Custom Style
st.markdown("""
<style>
body {
    background-color: #fceff9;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #fdf2f8, #ffffff);
}
h1, h2, h3, h4 {
    color: #a83279;
    text-align: center;
}
.css-1v0mbdj p {
    text-align: center;
}
.result-box {
    background-color: #fff0f5;
    border: 2px solid #ffa5ba;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-top: 20px;
}
.tip-box {
    background-color: #fffafc;
    border-left: 6px solid #ff69b4;
    padding: 16px;
    margin-top: 10px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# 💖 App Title and Intro
st.title("🎀 FashionVision AI 👠")
st.markdown("""
### 🔍 What is it?
Welcome to **FashionVision AI** — your personal **AI fashion stylist**!  
This tool uses a smart CNN model trained on women's outfit images to detect what *type of look* you're rocking 👗✨

---

### 👗 Style Categories:
Each outfit will be classified into one of the following:
- 👖 **Casual** — Chill fits, t-shirts, jeans, comfort wear  
- 🧥 **Formal** — Office-ready, boss vibes, sleek & smart  
- 💃 **Party** — Bold, sparkly, flirty, ready to slay  
- 👟 **Sporty** — Gym or street, athletic & powerful  
- 🪔 **Traditional** — Cultural classics, ethnic elegance  

---

### 💡 Why it’s special:
- ✅ Built on **MobileNetV2**
- ✅ Personalized **style tips** after prediction
- ✅ Works on desktop + mobile
- ✅ Beautiful UI + one-click upload
- ✅ Perfect for fashion apps, portfolios, or e-commerce previews

👇 Ready to test your fashion fit? Upload an outfit photo below:
""")

# Class labels
class_names = ['casual', 'formal', 'party', 'sporty', 'traditional']

# Load model
model = tf.keras.models.load_model("fashion_classifier_model.keras")

# Upload
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("📸 Upload your outfit photo", type=["jpg", "jpeg", "png"])

# Predict
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown("### 📷 Here's your uploaded outfit:")
    st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("💭 Analyzing your look..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # 🎯 Styled Results
    st.markdown(f"""
    <div class="result-box">
        <h2>👑 Your Style Match: <span style='color:#d63384'>{predicted_class.upper()}</span></h2>
        <p>📊 <strong>Confidence:</strong> <span style='color:#8B008B;'>{confidence:.2f}%</span></p>
    </div>
    """, unsafe_allow_html=True)

    # 💡 Fashion Tips
    tips = {
        "casual": "✨ *Add a denim jacket or crossbody bag to glow up your everyday look.*",
        "formal": "✨ *Blazers + sleek ponytail = Powerhouse vibes.*",
        "party": "✨ *Bold lips or glitter heels will steal the spotlight.*",
        "sporty": "✨ *Pair it with sneakers, a cap, and bold energy.*",
        "traditional": "✨ *Colorful bangles or a bright dupatta will complete your ethnic glam.*"
    }

    st.markdown(f"""
    <div class="tip-box">
        💡 <strong>Style Tip:</strong> {tips[predicted_class]}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("📌 [View on GitHub](https://github.com/subata24) &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; 🧵 Coded with 💖 by **Subata** using TensorFlow + Streamlit")