import os
import torch
import onnx
import numpy
import onnxruntime as ort
from diffusers import StableDiffusionPipeline
from utils import *
from torch.export import Dim

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype=torch.float16)
directory_unet = os.path.join(MODELS_DIR_PATH, "unet")
os.makedirs(directory_unet, exist_ok=True)
path_unet_onnx = os.path.join(directory_unet, 'model.onnx')
unet = pipe.unet.to("cuda")
unet.eval()
batch_dim = Dim("batch", min=1, max=4)  # Adjust max as needed; omit for unbounded
dynamic_shapes = {
    "sample": {0: batch_dim},
    "timestep": {},
    "encoder_hidden_states": {0: batch_dim},
}
sample_input = torch.randn((2, 4, 64, 64)).to(torch.float16).to("cuda")
timestep_input = torch.randn((1,)).to(torch.float16).to("cuda")
encoder_hidden_states_input = torch.randn((2, 77, 768)).to(torch.float16).to("cuda")
with torch.no_grad():
    torch.onnx.export(
        unet,
        (sample_input, timestep_input, encoder_hidden_states_input),
        path_unet_onnx,
        dynamo=True,  # Explicit, but default in 2.9
        dynamic_shapes=dynamic_shapes,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["outputs"],
        opset_version=18
    )

onnx_model = onnx.load(path_unet_onnx)
onnx.save(onnx_model, path_unet_onnx, save_as_external_data=False)  # Saves embedded

sample = torch.randn((2, 4, 64, 64)).to(torch.float16).to("cuda")
timestep = torch.randn((1,)).to(torch.float16).to("cuda")
encoder_hidden_states = torch.randn((2, 77, 768)).to(torch.float16).to("cuda")
with torch.no_grad():
    output_pytorch = unet(sample, timestep, encoder_hidden_states)
ort_session = ort.InferenceSession(path_unet_onnx, providers=['CUDAExecutionProvider'])
sample_numpy = numpy.array(sample.cpu(), dtype=numpy.float16)
timestep_numpy = numpy.array(timestep.cpu(), dtype=numpy.float16)
encoder_hidden_states_numpy = numpy.array(encoder_hidden_states.cpu(), dtype=numpy.float16)
output_onnx = ort_session.run(None, {"sample": sample_numpy, "timestep": timestep_numpy, "encoder_hidden_states": encoder_hidden_states_numpy})
print("Are unet outputs the same? ", numpy.allclose(output_pytorch.sample.cpu().numpy(), output_onnx[0], rtol=1e-02, atol=1e-02))
