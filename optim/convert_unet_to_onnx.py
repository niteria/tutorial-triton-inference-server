import os
import torch
import onnx
import numpy
import onnxruntime as ort
from diffusers import StableDiffusionXLPipeline
from utils import *
from onnx.external_data_helper import convert_model_to_external_data

from onnx import numpy_helper
from torch.export import Dim

class WrappedUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

pipe = StableDiffusionXLPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype=torch.float)
directory_unet = os.path.join(MODELS_DIR_PATH, "unet")
os.makedirs(directory_unet, exist_ok=True)
path_unet_onnx = os.path.join(directory_unet, 'model.onnx')
unet = pipe.unet.to("cuda")
unet.eval()
wrapped_unet = WrappedUNet(unet)
batch_dim = Dim("batch", min=1, max=4)  # Adjust max as needed; omit for unbounded
dynamic_shapes = {
    "sample": {0: batch_dim},
    "timestep": {0: batch_dim},
    "encoder_hidden_states": {0: batch_dim},
    "text_embeds": {0: batch_dim},
    "time_ids": {0: batch_dim}
}
sample_input = torch.randn((2, 4, 64, 64)).to(torch.float).to("cuda")
timestep_input = torch.randn((2,)).to(torch.float).to("cuda")  # Adjusted to match batch
encoder_hidden_states_input = torch.randn((2, 77, 2048)).to(torch.float).to("cuda")
text_embeds_input = torch.randn((2, 1280)).to(torch.float).to("cuda")
time_ids_input = torch.randn((2, 6)).to(torch.float).to("cuda")
with torch.no_grad():
    torch.onnx.export(
        wrapped_unet,
        (sample_input, timestep_input, encoder_hidden_states_input, text_embeds_input, time_ids_input),
        path_unet_onnx,
        external_data=True,
        dynamo=True,  # Explicit, but default in 2.9
        dynamic_shapes=dynamic_shapes,
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["outputs"],
        opset_version=18
    )

# # Post-export fix: Convert all float32 constants to float
# onnx_model = onnx.load(path_unet_onnx)
# for init in onnx_model.graph.initializer:
#     if init.data_type == onnx.TensorProto.FLOAT:
#         fp32_arr = numpy_helper.to_array(init)
#         fp16_arr = fp32_arr.astype(numpy.float)
#         new_init = numpy_helper.from_array(fp16_arr, init.name)
#         init.CopyFrom(new_init)
# # Convert constant nodes' attributes
# for node in onnx_model.graph.node:
#     if node.op_type == 'Constant':
#         for attr in node.attribute:
#             if attr.type == AttributeProto.TENSOR:
#                 t = attr.t
#                 if t.data_type == TensorProto.FLOAT:
#                     fp32_arr = numpy_helper.to_array(t, base_dir=directory_unet)
#                     fp16_arr = fp32_arr.astype(np.float)
#                     new_t = numpy_helper.from_array(fp16_arr, t.name)
#                     attr.t.CopyFrom(new_t)
# convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location='weights.data', size_threshold=0, convert_attribute=False)
# onnx.save(onnx_model, path_unet_onnx, save_as_external_data=True, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# onnx.checker.check_model(directory_unet)  # Validate the fixed model


sample = torch.randn((2, 4, 64, 64)).to(torch.float).to("cuda")
timestep = torch.randn((2,)).to(torch.float).to("cuda")  # Adjusted to match batch
encoder_hidden_states = torch.randn((2, 77, 2048)).to(torch.float).to("cuda")
added_cond_kwargs = {
    "text_embeds": torch.randn((2, 1280)).to(torch.float).to("cuda"),
    "time_ids": torch.randn((2, 6)).to(torch.float).to("cuda")
}
with torch.no_grad():
    output_pytorch = unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
ort_session = ort.InferenceSession(path_unet_onnx, providers=['CUDAExecutionProvider'])
inputs = {
    "sample": numpy.array(sample.cpu(), dtype=numpy.float32),
    "timestep": numpy.array(timestep.cpu(), dtype=numpy.float32),
    "encoder_hidden_states": numpy.array(encoder_hidden_states.cpu(), dtype=numpy.float32),
    "text_embeds": numpy.array(added_cond_kwargs["text_embeds"].cpu(), dtype=numpy.float32),
    "time_ids": numpy.array(added_cond_kwargs["time_ids"].cpu(), dtype=numpy.float32)
}
output_onnx = ort_session.run(None, inputs)
print("Are unet outputs the same? ", numpy.allclose(output_pytorch.sample.cpu().numpy(), output_onnx[0], rtol=1e-02, atol=1e-02))
