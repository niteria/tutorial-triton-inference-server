import torch
from diffusers import DiffusionPipeline

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained("eniora/Juggernaut_XL_Ragnarok", dtype=torch.bfloat16, device_map="cuda")

prompt = "cat in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
# print("Image shape:", image.shape)
#
# img = Image.fromarray(image)
image.save("cat.png")
print("Image saved to cat.png")
