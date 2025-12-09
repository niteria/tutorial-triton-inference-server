import os
import torch
import numpy
import onnxruntime as ort
from diffusers import StableDiffusionPipeline
from utils import *
from torch.export import Dim

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype=torch.float16)
VAE_ENCODER_DIR = os.path.join(MODELS_DIR_PATH, "vae_encoder")
os.makedirs(VAE_ENCODER_DIR, exist_ok=True)
VAE_ENCODER_ONNX_PATH = os.path.join(VAE_ENCODER_DIR, "model.onnx")
vae = pipe.vae.to(dtype=torch.float16).eval().to("cuda")  # Use vae directly, ensure fp16
# batch_dim = Dim("batch", max=1)  # Adjust max as needed; omit for unbounded
dynamic_shapes = {
    # "sample": {0: batch_dim},
}
# Override to use deterministic mode (mean) instead of sampling with random noise
vae.forward = lambda sample: vae.encode(sample).latent_dist.mode()
sample_input = torch.randn(1, 3, 512, 512).to(torch.float16).to("cuda")
with torch.no_grad():
    torch.onnx.export(
        vae,
        sample_input,
        VAE_ENCODER_ONNX_PATH,
        dynamo=True,  # Explicit, but default in recent PyTorch
        dynamic_shapes=dynamic_shapes,
        input_names=["sample"],
        output_names=["latent"],
        opset_version=18
    )
sample = torch.randn(1, 3, 512, 512).to(torch.float16).to("cuda")
with torch.no_grad():
    output_pytorch = vae(sample)
ort_session = ort.InferenceSession(VAE_ENCODER_ONNX_PATH, providers=['CUDAExecutionProvider'])
sample_numpy = numpy.array(sample.cpu(), dtype=numpy.float16)
output_onnx = ort_session.run(None, {"sample": sample_numpy})
# Diagnostic checks for fp16 differences
diff = numpy.abs(output_pytorch.cpu().numpy() - output_onnx[0])
print("Max abs diff:", diff.max())
print("Mean abs diff:", diff.mean())
print("Are VAE encoder outputs the same?", numpy.allclose(output_pytorch.cpu().numpy(), output_onnx[0], rtol=1e-02, atol=1e-02))
