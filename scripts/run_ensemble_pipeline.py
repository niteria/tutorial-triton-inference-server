from pipelines.ensemble import StableDiffusionPipelineEnsemble
from PIL import Image  # Add this import

if __name__ == "__main__":
    pipe = StableDiffusionPipelineEnsemble()
    image = pipe(prompt="a cat", seed=42)
    print("Image shape:", image.shape)

    img = Image.fromarray(image[0])
    img.save("cat.png")
    print("Image saved to cat.png")
