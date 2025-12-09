import os
import torch
import numpy
import onnxruntime as ort
from diffusers import StableDiffusionPipeline
from utils import *
from torch.export import Dim

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype=torch.float16)
TEXT_ENCODER_DIR = os.path.join(MODELS_DIR_PATH, "text_encoder")
os.makedirs(TEXT_ENCODER_DIR, exist_ok=True)
TEXT_ENCODER_ONNX_PATH = os.path.join(TEXT_ENCODER_DIR, "model.onnx")
text_encoder = pipe.text_encoder.to(dtype=torch.float16).eval().to("cuda")  # Explicit fp16 conversion
batch_dim = Dim("batch", min=1, max=4)  # Adjust max as needed; omit for unbounded
dynamic_shapes = {
    "input_ids": {0: batch_dim},
}
# Use a real prompt for better validation
prompt = "a photo of an astronaut riding a horse on mars"
input_ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    torch.onnx.export(
        text_encoder,
        input_ids,
        TEXT_ENCODER_ONNX_PATH,
        dynamo=True,  # Explicit, but default
        dynamic_shapes=dynamic_shapes,
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_out"],
        opset_version=18
    )
with torch.no_grad():
    output_pytorch = text_encoder(input_ids)
ort_session = ort.InferenceSession(TEXT_ENCODER_ONNX_PATH, providers=['CUDAExecutionProvider'])
onnx_output = ort_session.run(None, {"input_ids": input_ids.cpu().numpy()})
# Diagnostic checks
diff = numpy.abs(output_pytorch[0].cpu().numpy() - onnx_output[0])
print("Max abs diff:", diff.max())
print("Mean abs diff:", diff.mean())
# Loosen tolerance for fp16
print("Are text encoder outputs the same?", numpy.allclose(output_pytorch[0].cpu().numpy(), onnx_output[0], rtol=0.1, atol=0.1))
