from pipelines.bls import StableDiffusionPipelineBLS
from PIL import Image  # Add this import

if __name__ == "__main__":
    pipe = StableDiffusionPipelineBLS()
    image = pipe(prompt="a cat", seed=31337)
    print("Image shape:", image.shape)

    img = Image.fromarray(image)
    img.save("robot.png")
    print("Image saved to robot.png")
