import os
import torch
import numpy
import onnxruntime as ort
from diffusers import StableDiffusionXLPipeline
from utils import *

pipe = StableDiffusionXLPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype=torch.float32)
VAE_DECODER_DIR = os.path.join(MODELS_DIR_PATH, "vae_decoder")
os.makedirs(VAE_DECODER_DIR, exist_ok=True)
VAE_DECODER_ONNX_PATH = os.path.join(VAE_DECODER_DIR, "model.onnx")
vae = pipe.vae.to(dtype=torch.float32).eval().to("cuda")
vae.forward = lambda z: vae.decode(z).sample
sample_input = torch.randn(1, 4, 64, 64).to(torch.float32).to("cuda")
with torch.no_grad():
    torch.onnx.export(
        vae,
        sample_input,
        VAE_DECODER_ONNX_PATH,
        dynamo=True,
        dynamic_shapes={},
        input_names=["sample"],
        output_names=["image"],
        opset_version=18
    )
sample = torch.randn(1, 4, 64, 64).to(torch.float32).to("cuda")
with torch.no_grad():
    output_pytorch = vae(sample)
ort_session = ort.InferenceSession(VAE_DECODER_ONNX_PATH, providers=['CUDAExecutionProvider'])
sample_numpy = numpy.array(sample.cpu(), dtype=numpy.float32)
output_onnx = ort_session.run(None, {"sample": sample_numpy})
diff = numpy.abs(output_pytorch.cpu().numpy() - output_onnx[0])
print("Max abs diff:", diff.max())
print("Mean abs diff:", diff.mean())
print("Are VAE decoder outputs the same?", numpy.allclose(output_pytorch.cpu().numpy(), output_onnx[0], rtol=0.2, atol=0.2))
