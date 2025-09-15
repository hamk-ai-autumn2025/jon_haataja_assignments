import sys
import base64
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import argparse

# This script performs the following:
# 1. Takes an input image file path. 
# 2. Uses GPT with vision capabilities to describe the image in one sentence.
# 3. Uses the generated description to create a new image using DALL·E.
# 4. Saves the newly generated image to a specified output file.
# Install with: pip install openai python-dotenv
# Note: This script uses asynchronous programming for better performance.
# The script supports image sizes: 1024x1024, 1024x1536, 1536x1024, and auto (default size: 1024x1024).
# The generated image is saved in PNG format by default.
# The script uses the gpt-4o-mini model for image description and gpt-image-1 for image generation.

# Usage: python assignment-5.py <input_image_path> --size <image_size> --output <output_image_path>
# Example: python assignment-5.py input.jpg --size 1024x1024 --output new_image.png

# Parameters available:
# --size: Size of the generated image (default: 1024x1024)
# --output: Filename for the generated image (default: generated_image.png)
# --image: Path to the input image file (required)

load_dotenv()

async def image_to_text_to_image(image_path: str, size: str, output_path: str):
    # Initialize client with API key from environment
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Describe the image using GPT with vision
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print("Describing image...")
    description_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes images."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
                }}
            ]}
        ],
        max_tokens=50
    )

    description = description_response.choices[0].message.content.strip()
    print("\nImage Description:", description)

    # Step 2: Generate image from description
    print(f"\nGenerating image from description (size={size})...")
    image_response = await client.images.generate(
        model="gpt-image-1",
        prompt=description,
        size=size,
        n=1
    )

    # Save the generated image
    image_base64 = image_response.data[0].b64_json
    generated_image = base64.b64decode(image_base64)

    with open(output_path, "wb") as f:
        f.write(generated_image)

    print(f"\nNew image saved as: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Image → Text → Image generator using OpenAI")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--size", default="1024x1024",
                        choices=["1024x1024", "1024x1536", "1536x1024", "auto"],
                        help="Size of generated image (default: 1024x1024)")
    parser.add_argument("--output", default="generated_image.png",
                        help="Filename for the generated image (default: generated_image.png)")
    args = parser.parse_args()

    await image_to_text_to_image(args.image, args.size, args.output)


if __name__ == "__main__":
    asyncio.run(main())
