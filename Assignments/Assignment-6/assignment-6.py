import argparse
import os
import requests
import time
import base64
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

# This program supports the following parameters:
# --prompt: Main text prompt for the image (required)
# --negative_prompt: What to avoid in the image (optional)
# --aspect_ratio: Aspect ratio of the image (default: 1:1). Supported: 1:1, 2:3, 3:2, auto, 16:9, 4:3, 3:4
# --n: Number of images to generate (default: 1, max: 10


# Aspect ratio presets mapped to valid API sizes
ASPECT_RATIOS = {
    "1:1": "1024x1024",    # Square
    "2:3": "1024x1536",    # Portrait
    "3:2": "1536x1024",    # Landscape
    "auto": "auto",        # Let API decide
    # Aliases for convenience
    "16:9": "1536x1024",   # Approximate widescreen -> landscape
    "4:3": "1536x1024",    # Approximate -> landscape
    "3:4": "1024x1536",    # Approximate -> portrait
}

def download_image(url, prefix="image"):
    response = requests.get(url)
    response.raise_for_status()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}_{int(time.time()*1000)}.png"
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def save_base64_image(b64_data, prefix="image"):
    image_data = base64.b64decode(b64_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}_{int(time.time()*1000)}.png"
    with open(filename, "wb") as f:
        f.write(image_data)
    return filename

def upload_to_imgbb(filepath, api_key):
    with open(filepath, "rb") as f:
        image_data = f.read()
    url = "https://api.imgbb.com/1/upload"
    resp = requests.post(url, data={"key": api_key}, files={"image": image_data})
    resp.raise_for_status()
    j = resp.json()
    if j.get("success"):
        return j["data"]["url"]
    else:
        raise RuntimeError(f"ImgBB upload failed: {j}")

def generate_images(prompt, negative_prompt=None, aspect_ratio="1:1", n=1):
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect ratio {aspect_ratio}. Supported: {list(ASPECT_RATIOS.keys())}")
    
    if not IMGBB_API_KEY:
        raise EnvironmentError("IMGBB_API_KEY not set. Please set it as an environment variable.")

    size = ASPECT_RATIOS[aspect_ratio]

    # Construct full prompt (negative prompt as avoidance)
    full_prompt = prompt
    if negative_prompt:
        full_prompt += f". Avoid: {negative_prompt}"

    response = client.images.generate(
        model="gpt-image-1",
        prompt=full_prompt,
        size=size,
        n=n,
    )

    hosted_urls = []
    local_files = []
    for i, item in enumerate(response.data):
        if hasattr(item, "url") and item.url:
            url = item.url
            filename = download_image(url, prefix=f"gen_{i}")
        elif hasattr(item, "b64_json") and item.b64_json:
            filename = save_base64_image(item.b64_json, prefix=f"gen_{i}")
        else:
            continue

        local_files.append(filename)
        hosted_url = upload_to_imgbb(filename, IMGBB_API_KEY)
        hosted_urls.append(hosted_url)
        print(f"Uploaded {filename} -> {hosted_url}")
    
    return hosted_urls, local_files

def main():
    parser = argparse.ArgumentParser(description="Versatile OpenAI Image Generator with ImgBB Upload")
    parser.add_argument("--prompt", required=True, help="Main text prompt for the image")
    parser.add_argument("--negative_prompt", default=None, help="What to avoid in the image")
    parser.add_argument("--aspect_ratio", choices=ASPECT_RATIOS.keys(), default="1:1", help="Aspect ratio")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
    
    args = parser.parse_args()

    hosted_urls, local_files = generate_images(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        aspect_ratio=args.aspect_ratio,
        n=args.n,
    )

    print("\nHosted URLs (ImgBB):")
    for url in hosted_urls:
        print(url)

    print("\nSaved Local Files:")
    for fname in local_files:
        print(fname)

if __name__ == "__main__":
    main()