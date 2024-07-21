import gradio as gr
from lid import identify_and_plot, LID_EXAMPLES

lid_indentify = gr.Interface(
    fn=identify_and_plot,
    inputs=gr.Textbox(lines=2, label="Input text"),
    outputs=[gr.Textbox(label="Language"), gr.Plot(label="Confidence Plot")],
    examples=LID_EXAMPLES,
    title="Language Identification Demo",
    description="Identify the language of input text and view confidence levels.",
    allow_flagging="never",
)

with gr.Blocks() as demo:
    lid_indentify.render()

demo.launch()
