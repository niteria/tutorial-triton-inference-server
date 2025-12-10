import inspect
import numpy
import torch
import tritonclient.grpc as grpcclient
from transformers import CLIPTokenizer
from typing import List, Optional, Union
from diffusers.schedulers import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusionXLPipelineEnsemble():
    def __init__(self, url="localhost:8001"):
        super().__init__()
        self.client = grpcclient.InferenceServerClient(url=url)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        self.scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            trained_betas=None,
            skip_prk_steps=True,
            set_alpha_to_one=False,
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
        )
        self.vae_scale_factor = 8

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # First text encoder
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_numpy = text_inputs.input_ids.cpu().numpy().astype(numpy.int64)
        tokens_input = grpcclient.InferInput("input_ids", tokens_numpy.shape, "INT64")
        tokens_input.set_data_from_numpy(tokens_numpy)
        result = self.client.infer(
            model_name="text_encoder",
            inputs=[tokens_input],
            outputs=[grpcclient.InferRequestedOutput("last_hidden_state")]
        )
        last_hidden_state_numpy = result.as_numpy("last_hidden_state")
        hidden_states_1 = torch.from_numpy(last_hidden_state_numpy).to(device=device, dtype=torch.float16)
        hidden_states_1 = hidden_states_1.to(dtype=torch.float32, device=device)

        # Second text encoder
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_numpy_2 = text_inputs_2.input_ids.cpu().numpy().astype(numpy.int64)
        tokens_input_2 = grpcclient.InferInput("input_ids", tokens_numpy_2.shape, "INT64")
        tokens_input_2.set_data_from_numpy(tokens_numpy_2)
        outputs_2 = [
            grpcclient.InferRequestedOutput("last_hidden_state"),
            grpcclient.InferRequestedOutput("pooler_output")
        ]
        result_2 = self.client.infer(
            model_name="text_encoder_2",
            inputs=[tokens_input_2],
            outputs=outputs_2
        )
        last_hidden_state_numpy_2 = result_2.as_numpy("last_hidden_state")
        hidden_states_2 = torch.from_numpy(last_hidden_state_numpy_2).to(device=device, dtype=torch.float16)
        hidden_states_2 = hidden_states_2.to(dtype=torch.float32, device=device)
        pooler_output_numpy = result_2.as_numpy("pooler_output")
        add_text_embeds = torch.from_numpy(pooler_output_numpy).to(device=device, dtype=torch.float16)
        add_text_embeds = add_text_embeds.to(dtype=torch.float32, device=device)
        print("hidden_states_1 shape:", hidden_states_1.shape)
        print("hidden_states_2 shape:", hidden_states_2.shape)

        prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # Negative first encoder
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokens_numpy = uncond_input.input_ids.cpu().numpy().astype(numpy.int64)
            tokens_input = grpcclient.InferInput("input_ids", tokens_numpy.shape, "INT64")
            tokens_input.set_data_from_numpy(tokens_numpy)
            result = self.client.infer(
                model_name="text_encoder",
                inputs=[tokens_input],
                outputs=[grpcclient.InferRequestedOutput("last_hidden_state")]
            )
            last_hidden_state_numpy = result.as_numpy("last_hidden_state")
            neg_hidden_states_1 = torch.from_numpy(last_hidden_state_numpy).to(device=device, dtype=torch.float16)
            neg_hidden_states_1 = neg_hidden_states_1.to(dtype=torch.float32, device=device)

            # Negative second encoder
            uncond_input_2 = self.tokenizer_2(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokens_numpy_2 = uncond_input_2.input_ids.cpu().numpy().astype(numpy.int64)
            tokens_input_2 = grpcclient.InferInput("input_ids", tokens_numpy_2.shape, "INT64")
            tokens_input_2.set_data_from_numpy(tokens_numpy_2)
            result_2 = self.client.infer(
                model_name="text_encoder_2",
                inputs=[tokens_input_2],
                outputs=outputs_2
            )
            last_hidden_state_numpy_2 = result_2.as_numpy("last_hidden_state")
            neg_hidden_states_2 = torch.from_numpy(last_hidden_state_numpy_2).to(device=device, dtype=torch.float16)
            neg_hidden_states_2 = neg_hidden_states_2.to(dtype=torch.float32, device=device)
            pooler_output_numpy = result_2.as_numpy("pooler_output")
            neg_add_text_embeds = torch.from_numpy(pooler_output_numpy).to(device=device, dtype=torch.float16)
            neg_add_text_embeds = neg_add_text_embeds.to(dtype=torch.float32, device=device)

            negative_prompt_embeds = torch.cat([neg_hidden_states_1, neg_hidden_states_2], dim=-1)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            add_text_embeds = torch.cat([neg_add_text_embeds, add_text_embeds])

        # For classifier free guidance, we repeat/reshape if needed, but already handled in cat
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

        return prompt_embeds, add_text_embeds

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = 0,
    ):
        self._guidance_scale = guidance_scale
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = torch.device("cuda")

        prompt_embeds, add_text_embeds = self.encode_prompt(
            prompt,
            device,
            self.do_classifier_free_guidance,
            negative_prompt,
        )

        # Prepare added time ids
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor(add_time_ids, dtype=prompt_embeds.dtype).to(device)
        add_time_ids = add_time_ids.unsqueeze(0).expand(batch_size, -1)
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        generator = torch.Generator("cuda").manual_seed(seed)
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                sample_numpy = latent_model_input.cpu().numpy().astype(numpy.float16)
                timestep_numpy = t[None].cpu().numpy().astype(numpy.float16)
                encoder_hidden_states_numpy = prompt_embeds.cpu().numpy().astype(numpy.float16)
                text_embeds_numpy = add_text_embeds.cpu().numpy().astype(numpy.float16)
                time_ids_numpy = add_time_ids.cpu().numpy().astype(numpy.float16)

                input_sample = grpcclient.InferInput("sample", sample_numpy.shape, "FP16")
                input_sample.set_data_from_numpy(sample_numpy)
                input_timestep = grpcclient.InferInput("timestep", timestep_numpy.shape, "FP16")
                input_timestep.set_data_from_numpy(timestep_numpy)
                input_encoder_hidden_states = grpcclient.InferInput("encoder_hidden_states", encoder_hidden_states_numpy.shape, "FP16")
                input_encoder_hidden_states.set_data_from_numpy(encoder_hidden_states_numpy)
                input_text_embeds = grpcclient.InferInput("text_embeds", text_embeds_numpy.shape, "FP16")
                input_text_embeds.set_data_from_numpy(text_embeds_numpy)
                input_time_ids = grpcclient.InferInput("time_ids", time_ids_numpy.shape, "FP16")
                input_time_ids.set_data_from_numpy(time_ids_numpy)

                result = self.client.infer(
                    model_name="unet",
                    inputs=[input_sample, input_timestep, input_encoder_hidden_states, input_text_embeds, input_time_ids],
                    outputs=[grpcclient.InferRequestedOutput("outputs")]
                )
                output_numpy = result.as_numpy("outputs")
                noise_pred = torch.from_numpy(output_numpy).to(device=latent_model_input.device, dtype=torch.float16)
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents_numpy = (latents / 0.18215).cpu().numpy().astype(numpy.float16)
        vae_input = grpcclient.InferInput("sample", latents_numpy.shape, "FP16")
        vae_input.set_data_from_numpy(latents_numpy)
        result = self.client.infer(
            model_name="vae_decoder",
            inputs=[vae_input],
            outputs=[grpcclient.InferRequestedOutput("image")]
        )
        result = result.as_numpy("image")
        image = torch.tensor(result, dtype=torch.float16)
        image = (image / 2 + 0.5).clamp(0, 1) * 255
        image = image.cpu().permute(0, 2, 3, 1).float().numpy().astype(numpy.uint8)
        return image
