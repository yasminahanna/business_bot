"""Web interface: run with `python app.py`."""
import gradio as gr
from agent import run_agent

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=run_agent,
        title="FabFlow | Nova",
        description="A fictional prototyping studio assistant. Ask about 3D printing, request a quote, or submit feedback.",
        examples=["What materials can FabFlow 3D print?", "I'd like a quote for 20 PLA prototypes.",
                  "I wish you offered weekend appointments.", "Can you print in titanium?"],
    )
    demo.launch()
