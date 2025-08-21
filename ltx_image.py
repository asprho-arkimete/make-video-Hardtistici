import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    LTXConditionPipeline,
    LTXLatentUpsamplePipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video
from transformers import BitsAndBytesConfig as HFBitsAndBytesConfig, T5EncoderModel
from PIL import Image

# Step 1. Quantize text encoder (transformers)
quant_config_hf = HFBitsAndBytesConfig(load_in_8bit=True)
text_encoder = T5EncoderModel.from_pretrained(
    "Lightricks/LTX-Video-0.9.7-distilled",
    subfolder="text_encoder",
    quantization_config=quant_config_hf,
    torch_dtype=torch.float16,
)

# Step 2. Quantize transformer (diffusers)
quant_config_diff = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer = LTXVideoTransformer3DModel.from_pretrained(
    "Lightricks/LTX-Video-0.9.7-distilled",
    subfolder="transformer",
    quantization_config=quant_config_diff,
    torch_dtype=torch.float16,
    
)

# Step 3. Load LTXConditionPipeline with quantized components
pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.7-distilled",
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=torch.float16,
)
pipe.vae.enable_tiling()

# Step 4. Load Upsample Pipeline (non quantized, but moved to GPU)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    "Lightricks/ltxv-spatial-upscaler-0.9.7",
    vae=pipe.vae,
    torch_dtype=torch.float16
).to("cuda")

# Utility: round dimensions to VAE compression ratio
def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

# Input image (converted to fake video for conditioning)
image = Image.open("./image_flux_w.png")
video_cond_input = load_video(export_to_video([image]))  # Convert image to temporary MP4
condition1 = LTXVideoCondition(video=video_cond_input, frame_index=0)

# Prompts & parameters
prompt = "frontal view, full body, a blond girl with long hair, wearing red dress and heels shoes, walking forward , in a cyberpunk style city"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
expected_height, expected_width = 480, 832
downscale_factor = 2 / 3
num_frames = 96

# Step 5. Generate base video (low resolution)
downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)

latents = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=downscaled_width,
    height=downscaled_height,
    num_frames=num_frames,
    num_inference_steps=7,
    guidance_scale=1.0,
    decode_timestep=0.05,
    decode_noise_scale=0.025,
    generator=torch.Generator(device="cuda").manual_seed(0),
    output_type="latent",
).frames

# Step 6. Upscale latents to higher resolution (x2)
upscaled_latents = pipe_upsample(
    latents=latents,
    output_type="latent"
).frames

# Step 7. Final denoising pass (improves texture)
upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
video = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=upscaled_width,
    height=upscaled_width,
    num_frames=num_frames,
    denoise_strength=0.3,
    num_inference_steps=10,
    guidance_scale=1.0,
    latents=upscaled_latents,
    decode_timestep=0.05,
    decode_noise_scale=0.025,
    image_cond_noise_scale=0.025,
    generator=torch.Generator(device="cuda").manual_seed(0),
    output_type="pil",
).frames[0]

# Step 8. Resize to final output resolution
video = [frame.resize((expected_width, expected_height)) for frame in video]

# Export video
export_to_video(video, "output_LTX.mp4", fps=24)
