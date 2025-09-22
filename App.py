import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load your trained model
model = load_model("flower_cnn.h5")

# Define class labels (same order as training)
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Prediction function
def predict_flower(img: np.ndarray):
    # Convert to PIL image
    img = Image.fromarray(img).convert("RGB")

    # Resize to match model input (150x150)
    img = img.resize((150, 150))

    # Convert to array and normalize
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    return f"🌸 Predicted Flower: {classes[pred_index]} (Confidence: {confidence:.2f}%)"

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown(
        "<h1 style='text-align: center; color: #166534;'>🌼 Flower Classifier</h1>"
        "<p style='text-align: center;'>Upload a flower image and classify it into one of five categories</p>"
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(
                type="numpy",
                image_mode="RGB",
                sources=["upload", "webcam"],
                label="Upload Flower Image"
            )
            clear_btn = gr.Button("Clear", variant="secondary")
            submit_btn = gr.Button("Predict", variant="primary")

        with gr.Column():
            output_box = gr.Textbox(
                label="Prediction",
                placeholder="Model prediction will appear here"
            )

    # Button actions
    submit_btn.click(fn=predict_flower, inputs=img_input, outputs=output_box)
    clear_btn.click(fn=lambda: None, inputs=None, outputs=img_input)

# Run the app
if __name__ == "__main__":
    demo.launch()
