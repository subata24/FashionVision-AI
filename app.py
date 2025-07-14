import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("fashion_classifier_model.keras")

# Style categories
class_names = ['casual', 'formal', 'party', 'sporty', 'traditional']

# Prediction function
def predict(image):
    img = image.resize((224, 224))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Style tip
    tips = {
        "casual": "✨ Add a denim jacket or crossbody bag to glow up your everyday look.",
        "formal": "✨ Blazers + sleek ponytail = Powerhouse vibes.",
        "party": "✨ Bold lips or glitter heels will steal the spotlight.",
        "sporty": "✨ Pair it with sneakers, a cap, and bold energy.",
        "traditional": "✨ Colorful bangles or a bright dupatta will complete your ethnic glam."
    }

    result = f"👑 **Your Style Match:** {predicted_class.upper()} \n\n"
    result += f"📊 **Confidence:** {confidence:.2f}%\n\n"
    result += f"💡 **Style Tip:** {tips[predicted_class]}"
    return result

# App description
description = """
# 🎀 FashionVision AI 👠  
Welcome to **FashionVision AI** — your personal **AI fashion stylist for women!**  
Upload your outfit photo and let the model guess your look — casual, party, sporty, traditional or formal.  
✨ Built using TensorFlow + Gradio — deployed on Hugging Face Spaces.
"""

# Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📸 Upload Your Outfit Photo"),
    outputs=gr.Markdown(label="🔮 Prediction & Style Tip"),
    title="FashionVision AI 👗",
    description=description,
    theme=gr.themes.Soft(primary_hue="pink"),
    allow_flagging="never"
)

# Launch
if __name__ == "__main__":
    demo.launch()