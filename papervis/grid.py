import os
from typing import Tuple

from PIL import Image
import math

def transform(img):
    return img
    # Identity transform for now
    return crop(img, (50,100), (50, 100))

def crop(img: Image.Image, x: Tuple[int, int], y: Tuple[int, int]) -> Image.Image:
    """
    Crop an image using x and y coordinate ranges.

    Parameters:
        img: PIL.Image.Image — the input image
        x: Tuple[int, int] — horizontal pixel bounds (x_min, x_max)
        y: Tuple[int, int] — vertical pixel bounds (y_min, y_max)

    Returns:
        A cropped Image.
    """
    x_min, x_max = x
    y_min, y_max = y
    return img.crop((x_min, y_min, x_max, y_max))

def find_topdown_images(root_dir):
    topdown_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'topdown.png':
                topdown_paths.append(os.path.join(dirpath, filename))
    return topdown_paths

def load_and_transform_images(paths):
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    return images

def create_image_grid(images):
    if not images:
        return None

    # Determine grid size (closest to square)
    count = len(images)
    grid_cols = math.ceil(math.sqrt(count))
    grid_rows = math.ceil(count / grid_cols)

    # Assume all images are the same size
    width, height = images[0].size
    grid_image = Image.new('RGB', (grid_cols * width, grid_rows * height))

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        grid_image.paste(img, (col * width, row * height))

    return grid_image

# Example usage
if __name__ == "__main__":
    root_directory = "/home/charlie/Desktop/Holodeck/hippo/sampled_scenes/sacha_kitchen/sacha_kitchen/generated_on_2025-07-01-15-20-26"
    paths = find_topdown_images(root_directory)
    images = load_and_transform_images(paths)
    grid = create_image_grid(images)
    if grid:
        grid.save("grid_output.png")
