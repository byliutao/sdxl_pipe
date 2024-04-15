from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, EDMDPMSolverMultistepScheduler
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from dataclasses import dataclass
import os

import pipeline_stable_diffusion_xl


@dataclass
class TestConfig:
    model_name_or_path = '/data/model/playground-v2.5-1024px-aesthetic'
    device = 'cuda'
    project_dir = '/data/liutao/workspace/sdxl'
args = TestConfig()

scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
tokenizer    = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer")
tokenizer_2    = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(args.model_name_or_path, subfolder="text_encoder")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.model_name_or_path, subfolder="text_encoder_2")
unet = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet")
pipe = pipeline_stable_diffusion_xl.StableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=scheduler
)
pipe.to(args.device)

prompt = "A photo of an old man wearing a hat, walking in the park, with a dog"
prompt_2 = "A photo of an old man wearing a hat, walking in the park, with a dog"
images = pipe(prompt=prompt, prompt_2=prompt_2,guidance_scale=5.0).images[0]
images.save(os.path.join(args.project_dir,prompt+".jpg"))
