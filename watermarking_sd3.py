import argparse
import copy

import pandas as pd
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPTokenizer
from diffusion_utils import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, FluxPipeline, PixArtAlphaPipeline, \
    StableDiffusion3Pipeline, StableDiffusionPipeline, DDIMInverseScheduler
from diffusion_utils import ModifiedStableDiffusion3Pipeline
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from torch.cuda.amp import autocast
from utils import get_dataset, transform_img, measure_similarity, image_distortion, save_metrics
import os


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Define different schedulers for each synthesizer

    detector_scheduler = DDIMInverseScheduler.from_pretrained(args.model_path, subfolder='scheduler')

    synthesizers = [
        ModifiedStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                  torch_dtype=torch.float16),
    ]
    for i in range(len(synthesizers)):
        synthesizers[i] = synthesizers[i].to(device)
    detector = ModifiedStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                  torch_dtype=torch.float16, scheduler=detector_scheduler)
    detector = detector.to(device)

    prompts = ["A cat holding a sign that says hello world"]

    # Class for watermark
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number, channel=16, height=64)
    else:
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number, channel=16, height=64)

    os.makedirs(args.output_path, exist_ok=True)

    # Assume at the detection time, the original prompt is unknown
    tester_prompt = ''

    # Initialize dictionaries to store results for each synthesizer
    results = {synth.__class__.__name__: {'accuracy': [], 'clip_scores': []} for synth in synthesizers}

    for i in tqdm(range(0, len(prompts))):
        seed = i + args.gen_seed
        current_prompt = prompts[i]
        init_latents_w = watermark.create_watermark_and_return_w()

        for synthesizer in synthesizers:
            synthesizer_pipeline = synthesizer


            outputs = synthesizer_pipeline(
                current_prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
            image_w = outputs.images[0]
            output_image_path = os.path.join(args.output_path,
                                             f'{synthesizer_pipeline.__class__.__name__}_image_{i}_watermarked.png')
            image_w.save(output_image_path)

            # Calculate distortion
            image_w_distortion = image_distortion(image_w, seed, args)



            # Reverse image
            image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(init_latents_w.dtype).to(device)
            image_latents_w = detector.get_image_latents(image_w_distortion, sample=False)
            reversed_latents_w = detector.backward_sampling(
                "",
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=image_latents_w.to(device),
                output_type='latent'
            )[0]
            print(reversed_latents_w.shape)
            # Accuracy metric
            acc_metric = watermark.eval_watermark(reversed_latents_w)
            results[synthesizer_pipeline.__class__.__name__]['accuracy'].append(acc_metric)
            print(f"{synthesizer_pipeline.__class__.__name__} Accuracy: {acc_metric}")

    # Collect TPR metric
    tpr_detection, tpr_traceability = watermark.get_tpr()

    # Save metrics for each synthesizer
    for synth_name, result in results.items():
        print(f"Saving results for {synth_name}...")
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
