# Generate by Prompt using Flux-schnell

## Introduction

We design a module that generates photo-realistic & high resolution images based on user-defined prompts. While preparing the module,
we utilize the pretrained model [Flux-schnell  at Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell) provided by [black forests labs](https://blackforestlabs.ai/).

## Setting Up the Environment

### Using Conda (recommended)

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository:
    ~~~
    git clone https://github.com/byrkbrk/generating-by-prompt-flux-schnell.git
    ~~~
3. Change the directory:
    ~~~
    cd generating-by-prompt-flux-schnell
    ~~~
4. Create the environment:
    ~~~
    conda env create -f environment.yaml
    ~~~
5. Activate the environment:
    ~~~
    conda activate generating-by-prompt-flux-schnell
    ~~~

### Using pip

1. Download & install [Python](https://www.python.org/downloads/) (version==3.11)
2. Clone the repository:
    ~~~
    git clone https://github.com/byrkbrk/generating-by-prompt-flux-schnell.git
    ~~~
3. Change the directory:
    ~~~
    cd generating-by-prompt-flux-schnell
    ~~~
4. Install packages using `pip`:
    ~~~
    pip install -r requirements.txt
    ~~~

## Generating Images

Check it out how to use:

~~~
python3 generate.py --help
~~~

Output:

~~~
Generate images by prompt using Flux-schnell

positional arguments:
  prompt                Prompt that be used during inference

options:
  -h, --help            show this help message and exit
  --num_inference_steps NUM_INFERENCE_STEPS
                        Number of inference steps used during generating
  --device {cuda,mps,cpu}
                        The device used during inference. Default: `None`
  --enable_sequential_cpu_offload
                        Enables sequential cpu offload during inference
~~~

### Example usages
Execute the following code blocks to generate the corresponding images displayed below. The results will be saved into the folder `./generated-images`.

~~~
python3 generate.py\
 "an image of a turtle in Picasso style"\
 --num_inference_steps 4\
 --enable_sequential_cpu_offload
~~~

<p align="center">
  <img src="assets/generated_image_picasso.png" width="85%" />
</p>

~~~
python3 generate.py\
 "an image of a turtle in Camille Pissarro style"\
 --num_inference_steps 4\
 --enable_sequential_cpu_offload
~~~

<p align="center">
  <img src="assets/generated_image_pissarro.png" width="85%" />
</p>

~~~
python3 generate.py\
 "an image of a turtle in Claude Monet style"\
 --num_inference_steps 4\
 --enable_sequential_cpu_offload
~~~

<p align="center">
  <img src="assets/generated_image_monet.png" width="85%" />
</p>

## Generating via Gradio

Check it out how to use:

~~~
python3 app.py --help
~~~

Output:

~~~
Generate image using Flux-schnell via Gradio

options:
  -h, --help            show this help message and exit
  --enable_sequential_cpu_offload
                        Enables sequential cpu offload
  --share               Allows Gradio to produce public link
~~~
