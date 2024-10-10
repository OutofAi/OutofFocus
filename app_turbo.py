import warnings

warnings.filterwarnings("ignore")
from diffusers import DiffusionPipeline, DDIMInverseScheduler, DDIMScheduler
import torch
from typing import Optional
from tqdm import tqdm
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import gc
import gradio as gr
import numpy as np
import os
import pickle
import argparse
from PIL import Image
import requests
import math
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline


def save_state_to_file(state):
    filename = "state.pkl"
    with open(filename, "wb") as f:
        pickle.dump(state, f)
    return filename


def load_state_from_file(filename):
    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state

guidance_scale_value = 7.5
num_inference_steps = 10
weights = {}
res_list = []
foreground_mask = None
heighest_resolution = -1
signal_value = 2.0
blur_value = None
allowed_res_max = 1.0


def weight_population(layer_type, resolution, depth, value):
    # Check if layer_type exists, if not, create it
    if layer_type not in weights:
        weights[layer_type] = {}
    
    # Check if resolution exists under layer_type, if not, create it
    if resolution not in weights[layer_type]:
        weights[layer_type][resolution] = {}

    global heighest_resolution
    if resolution > heighest_resolution:
        heighest_resolution = resolution
  
    # Add/Modify the value at the specified depth (which can be a string)
    weights[layer_type][resolution][depth] = value

def resize_image_with_aspect(image, res_range_min=128, res_range_max=1024):
    # Get the original width and height of the image
    width, height = image.size
    
    # Determine the scaling factor to maintain the aspect ratio
    scaling_factor = 1
    if width < res_range_min or height < res_range_min:
        scaling_factor = max(res_range_min / width, res_range_min / height)
    elif width > res_range_max or height > res_range_max:
        scaling_factor = min(res_range_max / width, res_range_max / height)
    
    # Calculate the new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    print(f'{new_width}-{new_height}')
    
    # Resize the image with the new dimensions while maintaining the aspect ratio
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

def reconstruct(input_img, caption):
    global weights
    weights = {}

    prompt = caption

    img = input_img

    img = resize_image_with_aspect(img, res_range_min, res_range_max)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    if torch_dtype == torch.float16:
        loaded_image = transform(img).half().to("cuda").unsqueeze(0)
    else:
        loaded_image = transform(img).to("cuda").unsqueeze(0)

    if loaded_image.shape[1] == 4:
        loaded_image = loaded_image[:,:3,:,:]
        
    with torch.no_grad():
        encoded_image = pipe.vae.encode(loaded_image*2 - 1)
        real_image_latents = pipe.vae.config.scaling_factor * encoded_image.latent_dist.sample()


    # notice we disabled the CFG here by setting guidance scale as 1
    guidance_scale = 1.0
    inverse_scheduler.set_timesteps(num_inference_steps, device="cuda")
    timesteps = inverse_scheduler.timesteps

    latents = real_image_latents

    inversed_latents = [latents]

    def store_latent(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        with torch.no_grad():
            if step != num_inference_steps - 1:
                inversed_latents.append(latents)

        return callback_kwargs

    with torch.no_grad():

        replace_attention_processor(pipe.unet, True)

        pipe.scheduler = inverse_scheduler
        latents = pipe(prompt=prompt, 
            guidance_scale = guidance_scale,
            output_type="latent", 
            return_dict=False, 
            num_inference_steps=num_inference_steps, 
            latents=latents, 
            callback_on_step_end=store_latent,
            callback_on_step_end_tensor_inputs=["latents"],)[0]

    # initial state
    real_image_initial_latents = latents

    guidance_scale = guidance_scale_value
    scheduler.set_timesteps(num_inference_steps, device="cuda")
    timesteps = scheduler.timesteps

    def adjust_latent(pipe, step, timestep, callback_kwargs):

        with torch.no_grad():
            callback_kwargs["latents"] = inversed_latents[len(timesteps) - 1 - step].detach()

        return callback_kwargs
        
    with torch.no_grad():

        replace_attention_processor(pipe.unet, True)

        intermediate_values = real_image_initial_latents.clone()

        pipe.scheduler = scheduler
        intermediate_values = pipe(prompt=prompt, 
            guidance_scale = guidance_scale,
            output_type="latent", 
            return_dict=False, 
            num_inference_steps=num_inference_steps, 
            latents=intermediate_values,
            callback_on_step_end=adjust_latent,
            callback_on_step_end_tensor_inputs=["latents"],)[0]

        image = pipe.vae.decode(intermediate_values / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image_np = image.squeeze(0).float().permute(1, 2, 0).detach().cpu()
        image_np = (image_np / 2 + 0.5).clamp(0, 1).numpy()
        image_np = (image_np * 255).astype(np.uint8)

        update_scale(12)

        return image_np, caption, 12, [caption, real_image_initial_latents.detach(), inversed_latents, weights]

class AttnReplaceProcessor(AttnProcessor2_0):

    def __init__(self, replace_all, layer_type, layer_count, blur_sigma=None):
        super().__init__()
        self.replace_all = replace_all
        self.layer_type = layer_type
        self.layer_count = layer_count
        self.weight_populated = False
        self.blur_sigma = blur_sigma

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:


        dimension_squared = hidden_states.shape[1]

        is_cross = not encoder_hidden_states is None

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        height = width = math.isqrt(query.shape[2])

        
        if self.replace_all:
            weight_value = weights[self.layer_type][dimension_squared][self.layer_count]
            
            ucond_attn_scores, attn_scores = query.chunk(2)
            attn_scores[1].copy_(weight_value * attn_scores[0] + (1.0 - weight_value) * attn_scores[1])
            ucond_attn_scores[1].copy_(weight_value * ucond_attn_scores[0] + (1.0 - weight_value) * ucond_attn_scores[1])


            ucond_attn_scores, attn_scores = key.chunk(2)
            attn_scores[1].copy_(weight_value * attn_scores[0] + (1.0 - weight_value) * attn_scores[1])
            ucond_attn_scores[1].copy_(weight_value * ucond_attn_scores[0] + (1.0 - weight_value) * ucond_attn_scores[1])
        else:
            weight_population(self.layer_type, dimension_squared, self.layer_count, 1.0)


        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def replace_attention_processor(unet, clear=False, blur_sigma=None):
    attention_count = 0


    for name, module in unet.named_modules():
        if "attn1" in name and "to" not in name:
            layer_type = name.split(".")[0].split("_")[0]
            attention_count += 1
            
            if not clear:
                if layer_type == "down":
                    module.processor = AttnReplaceProcessor(True, layer_type, attention_count, blur_sigma=blur_sigma)
                elif layer_type == "mid":
                    module.processor = AttnReplaceProcessor(True, layer_type, attention_count, blur_sigma=blur_sigma)
                elif layer_type == "up":
                    module.processor = AttnReplaceProcessor(True, layer_type, attention_count, blur_sigma=blur_sigma)
                
            else:
                module.processor = AttnReplaceProcessor(False, layer_type, attention_count, blur_sigma=blur_sigma)


def apply_prompt(meta_data, new_prompt):

    caption, real_image_initial_latents, inversed_latents, _ = meta_data
    negative_prompt = ""

    inference_steps = len(inversed_latents)

    guidance_scale = guidance_scale_value
    scheduler.set_timesteps(inference_steps, device="cuda")
    timesteps = scheduler.timesteps

    initial_latents = torch.cat([real_image_initial_latents] * 2)

    def adjust_latent(pipe, step, timestep, callback_kwargs):
        replace_attention_processor(pipe.unet)

        with torch.no_grad():
            callback_kwargs["latents"][1] = callback_kwargs["latents"][1] + (inversed_latents[len(timesteps) - 1 - step].detach() - callback_kwargs["latents"][0])
            callback_kwargs["latents"][0] = inversed_latents[len(timesteps) - 1 - step].detach()

        return callback_kwargs

    
    with torch.no_grad():

        replace_attention_processor(pipe.unet)

        pipe.scheduler = scheduler
        latents = pipe(prompt=[caption, new_prompt], 
            negative_prompt=[negative_prompt, negative_prompt], 
            guidance_scale = guidance_scale,
            output_type="latent", 
            return_dict=False, 
            num_inference_steps=num_inference_steps, 
            latents=initial_latents,
            callback_on_step_end=adjust_latent,
            callback_on_step_end_tensor_inputs=["latents"],)[0]
        
        replace_attention_processor(pipe.unet, True)

        image = pipe.vae.decode(latents[1].unsqueeze(0) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image_np = image.squeeze(0).float().permute(1, 2, 0).detach().cpu()
        image_np = (image_np / 2 + 0.5).clamp(0, 1).numpy()
        image_np = (image_np * 255).astype(np.uint8)

    return image_np


def on_image_change(filepath):
    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]

    if filename in ["example1", "example3", "example4"]:

        meta_data_raw = load_state_from_file(f"assets/{filename}-turbo.pkl")

        global weights
        _, _, _, weights = meta_data_raw

        global num_inference_steps
        num_inference_steps = 10
        scale_value = 7

        if filename == "example1":
            scale_value = 8
            new_prompt = "a photo of a tree, summer, colourful"

        elif filename == "example3":
            scale_value = 6
            new_prompt = "a realistic photo of a female warrior, flowing dark purple or black hair, bronze shoulder armour, leather chest piece, sky background with clouds"

        elif filename == "example4":
            scale_value = 13
            new_prompt = "a photo of plastic bottle on some sand, beach background, sky background"

        update_scale(scale_value)
        img = apply_prompt(meta_data_raw, new_prompt)

    return filepath, img, meta_data_raw, num_inference_steps, scale_value, scale_value


def update_value(value, layer_type, resolution, depth):
    global weights
    weights[layer_type][resolution][depth] = value


def update_step(value):
    global num_inference_steps
    num_inference_steps = value

def adjust_ends(values, adjustment):
    # Forward loop to adjust the first valid element from the left
    for i in range(len(values)):
        if (adjustment > 0 and values[i + 1] == 1.0) or (adjustment < 0 and values[i] > 0.0):
            values[i] = values[i] + adjustment
            break

    # Backward loop to adjust the first valid element from the right
    for i in range(len(values)-1, -1, -1):
        if (adjustment > 0 and values[i - 1] == 1.0) or (adjustment < 0 and values[i] > 0.0):
            values[i] = values[i] + adjustment
            break

    return values

max_scale_value = 16

def update_scale(scale):
    global weights

    value_count = 0

    for outer_key, inner_dict in weights.items():
        for inner_key, values in inner_dict.items():
            for _, value in enumerate(values):
                value_count += 1

    list_values = [1.0] * value_count

    for _ in range(scale, max_scale_value):
        adjust_ends(list_values, -0.5)

    value_index = 0

    for outer_key, inner_dict in weights.items():
        for inner_key, values in inner_dict.items():
            for idx, value in enumerate(values):
                
                weights[outer_key][inner_key][value] = list_values[value_index]
                value_index += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable sharing of the Gradio interface")
    args = parser.parse_args()

    num_inference_steps = 10

    # model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    # guidance_scale_value = 7.5
    # resadapter_model_name = "resadapter_v2_sdxl"
    # res_range_min = 256 
    # res_range_max = 1536
    model_id = "runwayml/stable-diffusion-v1-5"
    guidance_scale_value = 7.5
    resadapter_model_name = "resadapter_v2_sd1.5"
    res_range_min = 128 
    res_range_max = 1024


    torch_dtype = torch.float16

    # torch_dtype = torch.float16
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to("cuda")
    pipe.load_lora_weights(
        hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="pytorch_lora_weights.safetensors"), 
        adapter_name="res_adapter",
        ) # load lora weights
    pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])
    pipe.unet.load_state_dict(
        load_file(hf_hub_download(repo_id="jiaxiangc/res-adapter", subfolder=resadapter_model_name, filename="diffusion_pytorch_model.safetensors")),
        strict=False,
        ) # load norm weights

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(
        """
            <div style="text-align: center;">
                <div style="display: flex; justify-content: center;">
                    <img src="https://github.com/user-attachments/assets/55a38e74-ab93-4d80-91c8-0fa6130af45a" alt="Logo">
                </div>
                <h1>Out of Focus v1.0 Turbo</h1>
                <p style="font-size:16px;">Out of AI presents a flexible tool to manipulate your images. This is our first version of Image modification tool through prompt manipulation by reconstruction through diffusion inversion process</p>
            </div>
            <br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href="https://www.buymeacoffee.com/outofai" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-red?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a> &ensp;
                <a href="https://twitter.com/OutofAi" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Ashleigh%20Watson"></a> &ensp;
                <a href="https://twitter.com/banterless_ai" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Alex%20Nasa"></a>
            </div>
            """
    )
    with gr.Row():
        with gr.Column():

            with gr.Row():
                example_input = gr.Image(type="filepath", visible=False)
                image_input = gr.Image(type="pil", label="Upload Source Image")
            steps_slider = gr.Slider(minimum=5, maximum=50, step=5, value=num_inference_steps, label="Steps", info="Number of inference steps required to reconstruct and modify the image")
            prompt_input = gr.Textbox(label="Prompt", info="Give an initial prompt in details, describing the image")
            reconstruct_button = gr.Button("Reconstruct")
        with gr.Column():

            with gr.Row():
                reconstructed_image = gr.Image(type="pil", label="Reconstructed")
                invisible_slider = gr.Slider(minimum=0, maximum=9, step=1, value=7, visible=False)
            interpolate_slider = gr.Slider(minimum=0, maximum=max_scale_value, step=1, value=max_scale_value, label="Cross-Attention Influence", info="Scales the related influence the source image has on the target image")
            new_prompt_input = gr.Textbox(label="New Prompt", interactive=False, info="Manipulate the image by changing the prompt or adding words at the end; swap words instead of adding or removing them for better results")

            with gr.Row():
                apply_button = gr.Button("Generate Vision", variant="primary", interactive=False)

            with gr.Row():
                show_case = gr.Examples(
                    examples=[
                        ["assets/example4.png", "a photo of plastic bottle on a rock, mountain background, sky background", "a photo of plastic bottle on some sand, beach background, sky background", 13],
                        ["assets/example1.png", "a photo of a tree, spring, foggy", "a photo of a tree, summer, colourful", 8],
                        [
                            "assets/example3.png",
                            "a digital illustration of a female warrior, flowing dark purple or black hair, bronze shoulder armour, leather chest piece, sky background with clouds",
                            "a realistic photo of a female warrior, flowing dark purple or black hair, bronze shoulder armour, leather chest piece, sky background with clouds",
                            6 ,
                        ],
                    ],
                    inputs=[example_input, prompt_input, new_prompt_input, interpolate_slider],
                    label=None,
                )

    meta_data = gr.State()

    example_input.change(fn=on_image_change, inputs=example_input, outputs=[image_input, reconstructed_image, meta_data, steps_slider, invisible_slider, interpolate_slider]).then(lambda: gr.update(interactive=True), outputs=apply_button).then(
        lambda: gr.update(interactive=True), outputs=new_prompt_input
    )
    steps_slider.release(update_step, inputs=steps_slider)
    interpolate_slider.release(update_scale, inputs=interpolate_slider)

    value_trigger = True

    def triggered():
        global value_trigger
        value_trigger = not value_trigger
        return value_trigger

    reconstruct_button.click(reconstruct, inputs=[image_input, prompt_input], outputs=[reconstructed_image, new_prompt_input, interpolate_slider, meta_data]).then(lambda: gr.update(interactive=True), outputs=reconstruct_button).then(lambda: gr.update(interactive=True), outputs=new_prompt_input).then(
        lambda: gr.update(interactive=True), outputs=apply_button
    )

    reconstruct_button.click(lambda: gr.update(interactive=False), outputs=reconstruct_button)

    reconstruct_button.click(lambda: gr.update(interactive=False), outputs=apply_button)

    apply_button.click(apply_prompt, inputs=[meta_data, new_prompt_input], outputs=reconstructed_image)

    demo.queue()
    demo.launch(share=args.share, inbrowser=True)

