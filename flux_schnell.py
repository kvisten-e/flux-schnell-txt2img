import os
from diffusers import DiffusionPipeline


class FluxSchnell:
    def __init__(self):
        self.model = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

    def generate(self, prompt):
        pass


if __name__ == "__main__":
    flux_schnell = FluxSchnell()