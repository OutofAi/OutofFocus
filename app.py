import warnings
warnings.filterwarnings("ignore")
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler
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
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

weights = {
    'down': {
        4096: 0.0,
        1024: 1.0,
        256: 1.0,
    },
    'mid': {
        64: 1.0,
    },
    'up': {
        256: 1.0,
        1024: 1.0,
        4096: 0.0,
    }
}
num_inference_steps = 10
model_id = "stabilityai/stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler")
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

should_stop = False

def save_state_to_file(state):
    filename = "state.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(state, f) 
    return filename

def load_state_from_file(filename):
    with open(filename, 'rb') as f:
        state = pickle.load(f) 
    return state 

def stop_reconstruct():
  global should_stop
  should_stop = True

def reconstruct(input_img, caption):

  img = input_img

  cond_prompt_embeds = pipe.encode_prompt(prompt=caption, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
  uncond_prompt_embeds = pipe.encode_prompt(prompt="", device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]

  prompt_embeds_combined = torch.cat([uncond_prompt_embeds, cond_prompt_embeds])


  transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((512, 512)),
      torchvision.transforms.ToTensor()
  ])

  loaded_image = transform(img).to("cuda").unsqueeze(0)

  if loaded_image.shape[1] == 4:
      loaded_image = loaded_image[:,:3,:,:]

  with torch.no_grad():
      encoded_image = pipe.vae.encode(loaded_image*2 - 1)
      real_image_latents = pipe.vae.config.scaling_factor * encoded_image.latent_dist.sample()

  guidance_scale = 1
  inverse_scheduler.set_timesteps(num_inference_steps, device="cuda")
  timesteps = inverse_scheduler.timesteps

  latents = real_image_latents

  inversed_latents = []

  with torch.no_grad():

      replace_attention_processor(pipe.unet, True)

      for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Inference steps"):

          inversed_latents.append(latents)

          latent_model_input = torch.cat([latents] * 2)

          noise_pred = pipe.unet(
              latent_model_input,
              t,
              encoder_hidden_states=prompt_embeds_combined,
              cross_attention_kwargs=None,
              return_dict=False,
          )[0]


          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
          noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

          latents = inverse_scheduler.step(noise_pred, t, latents, return_dict=False)[0]


  # initial state
  real_image_initial_latents = latents

  W_values = uncond_prompt_embeds.repeat(num_inference_steps, 1, 1)
  QT = nn.Parameter(W_values.clone())


  guidance_scale = 7.5
  scheduler.set_timesteps(num_inference_steps, device="cuda")
  timesteps = scheduler.timesteps

  optimizer = torch.optim.AdamW([QT], lr=0.008)

  pipe.vae.eval()
  pipe.vae.requires_grad_(False)
  pipe.unet.eval()
  pipe.unet.requires_grad_(False)

  last_loss = 1

  for epoch in range(50):
      gc.collect()
      torch.cuda.empty_cache()

      if last_loss < 0.02:
          break
      elif last_loss < 0.03:
          for param_group in optimizer.param_groups:
              param_group['lr'] = 0.003
      elif last_loss < 0.035:
          for param_group in optimizer.param_groups:
              param_group['lr'] = 0.006

      intermediate_values = real_image_initial_latents.clone()


      for i in range(num_inference_steps):
          latents = intermediate_values.detach().clone()

          t = timesteps[i]

          prompt_embeds = torch.cat([QT[i].unsqueeze(0), cond_prompt_embeds.detach()])

          latent_model_input = torch.cat([latents] * 2)

          noise_pred_model = pipe.unet(
              latent_model_input,
              t,
              encoder_hidden_states=prompt_embeds,
              cross_attention_kwargs=None,
              return_dict=False,
          )[0]

          noise_pred_uncond, noise_pred_text = noise_pred_model.chunk(2)
          noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

          intermediate_values = scheduler.step(noise_pred, t, latents, return_dict=False)[0]


          loss = F.mse_loss(inversed_latents[len(timesteps) - 1 - i].detach(), intermediate_values, reduction="mean")
          last_loss = loss

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      global should_stop
      if should_stop:
        should_stop = False
        break

      image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
      image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
      safety_checker_input = feature_extractor(image, return_tensors="pt", do_rescale=False).to("cuda")
      image = safety_checker(images=[image], clip_input=safety_checker_input.pixel_values.to("cuda"))[0]
      image_np = image[0].squeeze(0).float().permute(1,2,0).detach().cpu().numpy()
      image_np = (image_np * 255).astype(np.uint8)

      yield image_np, caption, [caption, real_image_initial_latents, QT]

  image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
  image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
  safety_checker_input = feature_extractor(image, return_tensors="pt", do_rescale=False).to("cuda")
  image = safety_checker(images=[image], clip_input=safety_checker_input.pixel_values.to("cuda"))[0]
  image_np = image[0].squeeze(0).float().permute(1,2,0).detach().cpu().numpy()
  image_np = (image_np * 255).astype(np.uint8)
  
  yield image_np, caption, [caption, real_image_initial_latents, QT]


class AttnReplaceProcessor(AttnProcessor2_0):

    def __init__(self, replace_all, weight):
        super().__init__()
        self.replace_all = replace_all
        self.weight = weight

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

        residual = hidden_states

        is_cross = not encoder_hidden_states is None

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_scores = attn.scale * torch.bmm(query, key.transpose(-1, -2))

        dimension_squared = hidden_states.shape[1]

        if not is_cross and (self.replace_all):
            ucond_attn_scores_src, ucond_attn_scores_dst, attn_scores_src, attn_scores_dst = attention_scores.chunk(4)
            attn_scores_dst.copy_(self.weight[dimension_squared] * attn_scores_src + (1.0 - self.weight[dimension_squared]) * attn_scores_dst)
            ucond_attn_scores_dst.copy_(self.weight[dimension_squared] * ucond_attn_scores_src + (1.0 - self.weight[dimension_squared]) * ucond_attn_scores_dst)

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        del attention_probs

        hidden_states = attn.to_out[0](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def replace_attention_processor(unet, clear = False):

  for name, module in unet.named_modules():
    if 'attn1' in name and 'to' not in name:
        layer_type = name.split('.')[0].split('_')[0]

        if not clear:
          if layer_type == 'down':
              module.processor = AttnReplaceProcessor(True, weights['down'])
          elif layer_type == 'mid':
              module.processor = AttnReplaceProcessor(True, weights['mid'])
          elif layer_type == 'up':
              module.processor = AttnReplaceProcessor(True, weights['up'])
        else:
          module.processor = AttnReplaceProcessor(False, 0.0)

def apply_prompt(meta_data, new_prompt):

  caption, real_image_initial_latents, QT = meta_data

  inference_steps = len(QT)

  cond_prompt_embeds = pipe.encode_prompt(prompt=caption, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
#   uncond_prompt_embeds = pipe.encode_prompt(prompt=caption, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
  new_prompt_embeds = pipe.encode_prompt(prompt=new_prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]

  guidance_scale = 7.5
  scheduler.set_timesteps(inference_steps, device="cuda")
  timesteps = scheduler.timesteps

  latents = torch.cat([real_image_initial_latents] * 2)

  with torch.no_grad():
    replace_attention_processor(pipe.unet)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Inference steps"):

        modified_prompt_embeds = torch.cat([QT[i].unsqueeze(0), QT[i].unsqueeze(0), cond_prompt_embeds, new_prompt_embeds])
        latent_model_input = torch.cat([latents] * 2)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=modified_prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]


        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    replace_attention_processor(pipe.unet, True)

    image = pipe.vae.decode(latents[1].unsqueeze(0) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
    safety_checker_input = feature_extractor(image, return_tensors="pt", do_rescale=False).to("cuda")
    image = safety_checker(images=[image], clip_input=safety_checker_input.pixel_values.to("cuda"))[0]
    image_np = image[0].squeeze(0).float().permute(1,2,0).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
  return image_np



def on_image_change(filepath):
    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Check if the filename is "example1" or "example2"
    if filename in ["example1", "example2", "example3", "example4"]:
        meta_data_raw = load_state_from_file(f"assets/{filename}.pkl")
        _, _, QT_raw = meta_data_raw

        global num_inference_steps
        num_inference_steps = len(QT_raw)
        scale_value = 7
        new_prompt = ""

        if filename == "example1":
            scale_value = 7
            new_prompt = "a photo of a tree, summer, colourful"
            
        elif filename == "example2":
            scale_value = 8
            new_prompt = "a photo of a panda, two ears, white background"

        elif filename == "example3":
            scale_value = 7
            new_prompt = "a realistic photo of a female warrior, flowing dark purple or black hair, bronze shoulder armour, leather chest piece, sky background with clouds"
            
        elif filename == "example4":
            scale_value = 7 
            new_prompt = "a photo of plastic bottle on some sand, beach background, sky background"

        update_scale(scale_value)
        img = apply_prompt(meta_data_raw, new_prompt)
            
    return filepath, img, meta_data_raw, num_inference_steps, scale_value, scale_value

def update_value(value, key, res):
    global weights
    weights[key][res] = value

def update_step(value):
    global num_inference_steps
    num_inference_steps = value

def update_scale(scale):
    values = [1.0] * 7

    if scale == 9:
        return values
    
    reduction_steps = (9 - scale) * 0.5
    
    for i in range(4):  # There are 4 positions to reduce symmetrically
        if reduction_steps >= 1:
            values[i] = 0.0
            values[-(i + 1)] = 0.0
            reduction_steps -= 1
        elif reduction_steps > 0:
            values[i] = 0.5
            values[-(i + 1)] = 0.5
            break

    global weights
    index = 0

    for outer_key, inner_dict in weights.items():
        for inner_key in inner_dict:
            inner_dict[inner_key] = values[index]
            index += 1
    
    return weights['down'][4096], weights['down'][1024], weights['down'][256], weights['mid'][64], weights['up'][256], weights['up'][1024], weights['up'][4096]
            

with gr.Blocks() as demo:
    gr.Markdown(
            '''
            <div style="text-align: center;">
                <div style="display: flex; justify-content: center;">
                    <img src="https://github.com/user-attachments/assets/1a94d4a6-6d76-4af8-bc53-6c6e04f7b71d" alt="Logo">
                </div>
                <h1>Out of Focus 1.0</h1>
                <p style="font-size:16px;">Out of AI presents a flexible tool to manipulate your images. This is our first version of Image modification tool through prompt manipulation by reconstruction through diffusion inversion process</p>
            </div>
            <br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href="https://www.buymeacoffee.com/outofai" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-red?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a> &ensp;
                <a href="https://twitter.com/OutofAi" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Ashleigh%20Watson"></a> &ensp;
                <a href="https://twitter.com/banterless_ai" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Alex%20Nasa"></a>
            </div>
            '''
        )
    with gr.Row():
      with gr.Column():

          with gr.Row():
            example_input = gr.Image(height=512, width=512, type="filepath", visible=False)
            image_input = gr.Image(height=512, width=512, type="pil", label="Upload Source Image")
          steps_slider = gr.Slider(minimum=5, maximum=25, step=5, value=num_inference_steps, label="Steps", info="Number of inference steps required to reconstruct and modify the image")
          prompt_input = gr.Textbox(label="Prompt", info="Give an initial prompt in details, describing the image")
          reconstruct_button = gr.Button("Reconstruct")
          stop_button = gr.Button("Stop", variant="stop", interactive=False)
      with gr.Column():
        reconstructed_image = gr.Image(type="pil", label="Reconstructed")

        with gr.Row():
            invisible_slider = gr.Slider(minimum=0, maximum=9, step=1, value=7, visible=False)
            interpolate_slider = gr.Slider(minimum=0, maximum=9, step=1, value=7, label="Cross-Attention Influence", info="Scales the related influence the source image has on the target image")
        with gr.Row():  
            new_prompt_input = gr.Textbox(label="New Prompt", interactive=False, info="Manipulate the image by changing the prompt or word addition at the end, achieve the best results by swapping words instead of adding or removing in between")
        with gr.Row():
            apply_button = gr.Button("Generate Vision", variant="primary", interactive=False)
        with gr.Row():
            with gr.Accordion(label="Advanced Options", open=False):
                    gr.Markdown(
                        '''
                        <div style="text-align: center;">
                            <h1>Weight Adjustment</h1>
                            <p style="font-size:16px;">Specific Cross-Attention Influence weights can be manually modified for given resolutions (1.0 = Fully Source Attn 0.0 = Fully Target Attn)</p>
                        </div>
                        '''
                    )
                    down_slider_4096 = gr.Number(minimum=0.0, maximum=1.0, step=0.1, value=weights['down'][4096], label="Self-Attn Down 64x64")
                    down_slider_1024 = gr.Number(minimum=0.0, maximum=1.0, step=0.1, value=weights['down'][1024], label="Self-Attn Down 32x32")
                    down_slider_256 = gr.Number(minimum=0.0, maximum=1.0, step=0.1, value=weights['down'][256], label="Self-Attn Down 16x16")
                    mid_slider_64 = gr.Number(minimum=0.0, maximum=1.0, step=0.1, value=weights['mid'][64], label="Self-Attn Mid 8x8")
                    up_slider_256 = gr.Number(minimum=0.0, maximum=1.0, step=0.1, value=weights['up'][256], label="Self-Attn Up 16x16")
                    up_slider_1024 = gr.Number(minimum=0.0, maximum=1.0, step=0.1, value=weights['up'][1024], label="Self-Attn Up 32x32")
                    up_slider_4096 = gr.Number(minimum=0.0, maximum=1.0, step=0.1, value=weights['up'][4096], label="Self-Attn Up 64x64")

        with gr.Row():
            show_case = gr.Examples(
                examples=[
                    ["assets/example4.png", "a photo of plastic bottle on a rock, mountain background, sky background", "a photo of plastic bottle on some sand, beach background, sky background"],
                    ["assets/example1.png", "a photo of a tree, spring, foggy", "a photo of a tree, summer, colourful"], 
                    ["assets/example2.png", "a photo of a cat, two ears, white background", "a photo of a panda, two ears, white background"], 
                    ["assets/example3.png", "a digital illustration of a female warrior, flowing dark purple or black hair, bronze shoulder armour, leather chest piece, sky background with clouds", "a realistic photo of a female warrior, flowing dark purple or black hair, bronze shoulder armour, leather chest piece, sky background with clouds"],
                     
                ],
                inputs=[example_input, prompt_input, new_prompt_input],
                label=None
            )

    meta_data = gr.State()

    example_input.change(
        fn=on_image_change,
        inputs=example_input,
        outputs=[image_input, reconstructed_image, meta_data, steps_slider, invisible_slider, interpolate_slider]
    ).then(
        lambda: gr.update(interactive=True),
        outputs=apply_button
    ).then(
        lambda: gr.update(interactive=True),
        outputs=new_prompt_input
    )
    steps_slider.release(update_step, inputs=steps_slider)
    interpolate_slider.release(update_scale, inputs=interpolate_slider, outputs=[down_slider_4096, down_slider_1024, down_slider_256, mid_slider_64, up_slider_256, up_slider_1024, up_slider_4096 ])
    invisible_slider.change(update_scale, inputs=invisible_slider, outputs=[down_slider_4096, down_slider_1024, down_slider_256, mid_slider_64, up_slider_256, up_slider_1024, up_slider_4096 ])

    up_slider_4096.change(update_value, inputs=[up_slider_4096, gr.State('up'), gr.State(4096)])
    up_slider_1024.change(update_value, inputs=[up_slider_1024, gr.State('up'), gr.State(1024)])
    up_slider_256.change(update_value, inputs=[up_slider_256, gr.State('up'), gr.State(256)])

    down_slider_4096.change(update_value, inputs=[down_slider_4096, gr.State('down'), gr.State(4096)])
    down_slider_1024.change(update_value, inputs=[down_slider_1024, gr.State('down'), gr.State(1024)])
    down_slider_256.change(update_value, inputs=[down_slider_256, gr.State('down'), gr.State(256)])

    mid_slider_64.change(update_value, inputs=[mid_slider_64, gr.State('mid'), gr.State(64)])

    reconstruct_button.click(reconstruct, inputs=[image_input, prompt_input], outputs=[reconstructed_image, new_prompt_input, meta_data]).then(
        lambda: gr.update(interactive=True),
        outputs=reconstruct_button
    ).then(
        lambda: gr.update(interactive=True),
        outputs=new_prompt_input
    ).then(
        lambda: gr.update(interactive=True),
        outputs=apply_button
    ).then(
        lambda: gr.update(interactive=False),
        outputs=stop_button
    )

    reconstruct_button.click(
        lambda: gr.update(interactive=False),
        outputs=reconstruct_button
    )

    reconstruct_button.click(
        lambda: gr.update(interactive=True),
        outputs=stop_button
    )

    reconstruct_button.click(
        lambda: gr.update(interactive=False),
        outputs=apply_button
    )

    stop_button.click(
        lambda: gr.update(interactive=False),
        outputs=stop_button
    )

    apply_button.click(apply_prompt, inputs=[meta_data, new_prompt_input], outputs=reconstructed_image)
    stop_button.click(stop_reconstruct)

if __name__ == "__main__":
    demo.launch(share=True)
    # demo.launch()