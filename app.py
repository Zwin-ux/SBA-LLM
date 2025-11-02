import gradio as gr
from sba_llm import analyze  # your function: bytes/pdf -> dict/text

def run(file):
    return analyze(file)

demo = gr.Interface(
    fn=run,
    inputs=gr.File(file_types=[".pdf"]),
    outputs="json",
    title="SBA-LLM",
)

if __name__ == "__main__":
    demo.launch()
