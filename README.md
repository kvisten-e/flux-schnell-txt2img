# Generate by Prompt using Flux-schnell

## Introduction

We design a module that generates photo-realistic & high resolution images based on user-defined prompts. While preparing the module,
we utilize the pretrained model [Flux-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) provided by [black forests labs](https://blackforestlabs.ai/) at Hugging Face.

## Setting Up the Environment

## Generating Images 

### Example usage
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
