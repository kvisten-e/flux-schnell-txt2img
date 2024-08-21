import os
import torch
from diffusers import DiffusionPipeline


class FluxSchnell:
    repo_id = "black-forest-labs/FLUX.1-schnell"

    def __init__(self,
                 device: str = "cpu",
                 create_dirs: bool = True,
                 enable_sequential_cpu_offload: bool = True):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.model = self.instantiate_model(self.__class__.repo_id, 
                                            self.device,
                                            torch.bfloat16)
        if create_dirs: self.create_dirs(self.module_dir)

    def generate(self, prompt, save=True, show=True):
        """Returns list of generated images for given prompts"""
        images = self.model(prompt)
        for i, image in enumerate(images):
            if save:
                image.save(os.path.join(self.module_dir, 
                                        "generated-images",
                                        f"generated_image_{i}.png"))
            if show:
                image.show()
        return images

    def instantiate_model(self, repo_id, device, dtype, enable_sequential_cpu_offload):
        """Returns instantiated model"""
        model = DiffusionPipeline.from_pretrained(repo_id,
                                                  torch_dtype=dtype).to(device)
        if enable_sequential_cpu_offload:
            model.enable_sequential_cpu_offload(device=device)
        return model
    
    def initialize_device(self, device: str):
        """Return the GPU device based on availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)

    def create_dirs(self, root):
        """Creates required directories under given root directory"""
        dir_names = ["generated-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)



if __name__ == "__main__":
    flux_schnell = FluxSchnell()