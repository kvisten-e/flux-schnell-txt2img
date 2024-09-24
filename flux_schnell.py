import os
import torch
from diffusers import DiffusionPipeline, FluxPipeline



class FluxSchnell:
    """A class for managing and running the FLUX.1-schnell model"""
    repo_id = "black-forest-labs/FLUX.1-schnell"

    def __init__(self,
                 device: str = None,
                 create_dirs: bool = True,
                 enable_sequential_cpu_offload: bool = True):
        """Initializes FluxSchnell class"""
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.model = self.instantiate_model(self.__class__.repo_id, 
                                            self.device,
                                            torch.bfloat16,
                                            enable_sequential_cpu_offload)
        if create_dirs: self.create_dirs(self.module_dir)

    def generate(self, prompt, num_inference_steps=4, seed=None, width=1024, height=1024, guidance_scale=7.5, save=True, show=True):
        """Returns list of generated images for given prompts"""
        if seed is None:
            seed = -1
            
        
        
        generator = torch.Generator("cpu").manual_seed(seed)
        
        images = self.model(prompt, 
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            width=width,
                            height=height,
                            max_sequence_length=256,
                            guidance_scale=guidance_scale
                            ).images
        for i, image in enumerate(images):
            if save:
                image.save(os.path.join(self.module_dir, 
                                        "generated-images",
                                        f"generated_image_{i}.png"))
            if show:
                image.show()
        return images

    def instantiate_model(self, repo_id, device, dtype, enable_sequential_cpu_offload, model_name=None, weight_name=None):
        """Returns instantiated model"""
        model = FluxPipeline.from_pretrained(repo_id,
                                            torch_dtype=dtype)
        if enable_sequential_cpu_offload:
            model.enable_sequential_cpu_offload(device=device)
        else:
            model = model.to(device)
            
        if model_name is not None:
            if weight_name is not None:    
                model.load_lora_weights(model_name, weight_name=weight_name)
            else:
                model.load_lora_weights(model_name)
        elif model_name is None and weight_name is not None:
            model.load_lora_weights(weight_name)
              
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
    prompt = "A cat holding a sign that says hello world"
    flux_schnell = FluxSchnell().generate(prompt, 4)
