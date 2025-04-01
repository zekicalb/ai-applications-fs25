import gradio as gr
from transformers import pipeline

classifier = pipeline("image-classification", model="kuhs/vit-base-oxford-iiit-pets")

def classify_pet(image):
    results = classifier(image)
    return {result['label']: result['score'] for result in results}

iface = gr.Interface(
    fn=classify_pet,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(),
    title="Pet Classification with ViT",
    description="Upload an image of a pet, and the model will classify it."
)

iface.launch()