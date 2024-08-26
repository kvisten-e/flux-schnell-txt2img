from argparse import ArgumentParser
import gradio as gr
from flux_schnell import FluxSchnell



def parse_arguments():
    """Returns parsed arguments"""
    pass


if __name__ == "__main__":
    args = parse_arguments()
    flux_schnell = FluxSchnell(None,
                               False,
                               args.enable_sequential_cpu_offload)
    gr_interface = gr.Interface(
        fn=lambda prompt, num_inference_steps: flux_schnell.generate(prompt,
                                                                     num_inference_steps,
                                                                     False,
                                                                     False)[0],
        inputs=[
            
        ],
        outputs=gr.Image(type="pil")
    )

    