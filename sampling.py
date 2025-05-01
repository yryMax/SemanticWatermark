import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers import PixArtAlphaPipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    height=512,
    width=512,
).images[0]
image.save("image_sd3.png")



pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A cat holding a sign that says hello world"
images = pipe(prompt=prompt).images[0]
images.save("image_flux.png")

# stable diffusion 2.1-base
scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt =  "A cat holding a sign that says hello world"
image = pipe(prompt).images[0]

image.save("image_sd_2_1.png")

scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt =  "A cat holding a sign that says hello world"
image = pipe(prompt).images[0]

image.save("image_sd_2.png")