import gradio as gr
from predict import *

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_toxicity,
    inputs=gr.Textbox(lines=2, placeholder='Type your comment here...'),
    outputs=[gr.Number(label=class_name) for class_name in CLASSES]
)

# Launch the Gradio app
iface.launch(share=True)