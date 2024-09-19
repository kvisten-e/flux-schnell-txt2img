from argparse import ArgumentParser
from flux_schnell import FluxSchnell



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate images by prompt using Flux-schnell")
    parser.add_argument("prompt", 
                        type=str, 
                        nargs="+",
                        help="Prompt that be used during inference")
    parser.add_argument("--num_inference_steps",
                        type=int,
                        default=4,
                        help="Number of inference steps used during generating")
    parser.add_argument("--device", 
                        type=str,
                        default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="The device used during inference. Default: `None`")
    parser.add_argument("--enable_sequential_cpu_offload",
                        action="store_true",
                        help="Enables sequential cpu offload during inference")
    
    parser.add_argument("--seed", 
                        type=int, 
                        default=None,
                        help="Random seed for reproducibility. Default: `None`")
    parser.add_argument("--width", 
                        type=int, 
                        default=1024,
                        help="Width of the generated image. Default: 512")
    parser.add_argument("--height", 
                        type=int, 
                        default=1024,
                        help="Height of the generated image. Default: 512")
    parser.add_argument("--guidance_scale", 
                        type=float, 
                        default=7.5,
                        help="Guidance scale for controlling adherence to the prompt. Default: 7.5")
    parser.add_argument("--num_images", 
                        type=int, 
                        default=1,
                        help="Amount of images to generate. Default: 1")
             
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    FluxSchnell(
        device=args.device,
        enable_sequential_cpu_offload=args.enable_sequential_cpu_offload
    ).generate(args.prompt,
               args.num_inference_steps,
               args.seed,
               args.width,
               args.height,
               args.guidance_scale)