from pipelines.ensemble_ragnarok import StableDiffusionXLPipelineEnsemble
from PIL import Image  # Add this import

if __name__ == "__main__":
    pipe = StableDiffusionXLPipelineEnsemble()
    prompt = "cat in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipe(prompt=prompt, seed=42)
    print("Image shape:", image.shape)

    img = Image.fromarray(image[0])
    img.save("cat.png")
    print("Image saved to cat.png")
