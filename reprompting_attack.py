import os
import subprocess
def get_free_gpu():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    memory_free = [int(x) for x in output.decode().strip().split('\n')]
    return memory_free.index(max(memory_free))

free_gpu = get_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)

import torch

import argparse

from tqdm import tqdm
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, FluxPipeline, PixArtAlphaPipeline, \
    StableDiffusion3Pipeline, StableDiffusionPipeline, DDIMInverseScheduler
from diffusion_utils import ModifiedStableDiffusion3Pipeline, ModifiedPixArtAlphaPipeline, InversableStableDiffusionPipeline, RFInversionFluxPipeline
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
import clip
from utils import get_dataset, transform_img, measure_similarity, image_distortion, save_metrics



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Define different schedulers for each synthesizer
    synthesizers = {
        "sd3": ModifiedStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                  torch_dtype=torch.float16),
        #"sd2": InversableStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16),

        "flux": RFInversionFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16),

        "pixart": ModifiedPixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16)

    }
    # loop through synthesizers
    for key, synthesizer in synthesizers.items():
        synthesizers[key] = synthesizer.to(device)

    proxy_model = InversableStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
    proxy_model_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder='scheduler')
    proxy_model = proxy_model.to(device)
    proxy_model.scheduler = proxy_model_scheduler

    prompts = ["A cat holding a sign that says hello world",
                "An astronaut floating in space"]

    os.makedirs(args.output_path, exist_ok=True)

    # Assume at the detection time, the original prompt is unknown
    tester_prompt = ''

    # Initialize dictionaries to store results for each synthesizer
    results = {synth: {'accuracy': [], 'clip_scores': []} for synth in synthesizers}

    for i in tqdm(range(0, len(prompts))):
        seed = i + args.gen_seed
        current_prompt = prompts[i]

        for synthesizer_name, synthesizer_pipeline in synthesizers.items():
            if synthesizer_name in ['flux', 'sd3']:
                args.channel_copy = 4
                channel = 16
                height = 64
            else:
                args.channel_copy = 1
                channel = 4
                height = 64
            if args.chacha:
                watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number,
                                                    channel=channel, height=height)
            else:
                watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number, channel=channel,
                                             height=height)
            # initialize watermark
            init_latents_w = watermark.create_watermark_and_return_w()
            if synthesizer_name == 'flux':
                init_latents_w = init_latents_w.to(torch.bfloat16).to(device).reshape(1, 1024, 64)
            else:
                init_latents_w = init_latents_w.to(torch.float16).to(device)
            # generate watermarked image
            outputs = synthesizer_pipeline(
                current_prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
            image_w = outputs.images[0]

            output_image_path = os.path.join(args.output_path,
                                             f'{synthesizer_name}_image_{i}_watermarked.png')

            image_w.save(output_image_path)

            # Calculate distortion
            image_w_distortion = image_distortion(image_w, seed, args)



            # Reverse image by proxy model
            text_embeddings = proxy_model.get_text_embedding(tester_prompt)
            image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
            image_latents_w = proxy_model.get_image_latents(image_w_distortion, sample=False)
            reversed_latents_w = proxy_model.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.num_inversion_steps,
            )

            outputs = proxy_model(
                "a picture of a dog in a park",
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=reversed_latents_w,
            )
            image_w = outputs.images[0]

            output_image_path = os.path.join(args.output_path,
                                             f'Reprompt_{synthesizer_name}_image_{i}_watermarked.png')

            image_w.save(output_image_path)


            # detect watermark
            image_w_distortion = image_distortion(image_w, seed, args)


            dtype = {'flux': torch.bfloat16, 'sd3': torch.float16, 'pixart': torch.float16}

            image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(dtype[synthesizer_name]).to(device)
            if synthesizer_name != 'flux':
                inverse_scheduler = DDIMInverseScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder='scheduler')
                synthesizer_pipeline.scheduler = inverse_scheduler
                image_latents_w = synthesizer_pipeline.get_image_latents(image_w_distortion, sample=False)
                reversed_latents_w = synthesizer_pipeline.backward_sampling(
                    "",
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    latents=image_latents_w.to(device),
                    output_type='latent'
                )[0]
            else:
                reversed_latents_w, _, _ = synthesizer_pipeline.invert(
                    image_w_distortion,
                    height=args.image_length,
                    width=args.image_length,
                    num_inversion_steps=args.num_inference_steps,
                    gamma=0
                )
                reversed_latents_w = reversed_latents_w.reshape(1, 16, 64, 64)
            print(reversed_latents_w.shape)

            # Accuracy metric
            acc_metric = watermark.eval_watermark(reversed_latents_w)
            # CLIP
            scores = measure_similarity([image_w], "a picture of a dog in a park", clip_model, clip_preprocess, clip.tokenize, device)

            results[synthesizer_name]['accuracy'].append(acc_metric)
            results[synthesizer_name]['clip_scores'].append(scores[0])

            print(f"{synthesizer_name}: Accuracy: {acc_metric}, Clip Score: {scores}")

    # Collect TPR metric
    tpr_detection, tpr_traceability = watermark.get_tpr()

    # Save metrics for each synthesizer
    for synth_name, result in results.items():
        # print(f"Saving results for {synth_name}...")
        print(f"{synthesizer_name}: Detection TPR: {tpr_detection}, Traceability TPR: {tpr_traceability}",
              f"Accuracy: {result['accuracy']}, Clip Score: {result['clip_scores']}")
        save_metrics(args, tpr_detection, tpr_traceability, result['accuracy'], result['clip_scores'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-base')

    # For image distortion
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
