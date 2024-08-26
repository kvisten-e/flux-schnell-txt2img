# Generate by Prompt using Flux-schnell

## Introduction

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
