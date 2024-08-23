from argparse import ArgumentParser
from flux_schnell import FluxSchnell



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate images by prompt using Flux-schnell")
    parser.add_argument("prompt", 
                        type=str, 
                        nargs="+",
                        help="Prompt that be used during inference")
    parser.add_argument("--device", 
                        type=str,
                        default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="The device used during inference. Default: `None`")
    parser.add_argument("--enable_sequential_cpu_offload",
                        action="store_true",
                        help="Enables sequentail cpu offload during inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")