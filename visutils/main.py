from PIL import Image, ImageSequence
import numpy as np

def gif_to_trail_image(gif_path, output_path, diff_threshold=30, alpha=0.7):
    """
    Creates a single image from a GIF by overlaying all robot positions while keeping the background.
    Older positions are blended with new ones to create a smooth trail effect.

    Parameters:
        gif_path (str): Path to the input GIF.
        output_path (str): Path to save the output image.
        diff_threshold (int): Sensitivity threshold for detecting motion.
        alpha (float): Blending factor (0 to 1) for older positions (higher = newer positions more visible).
    """
    # Open the GIF and extract frames
    gif = Image.open(gif_path)
    frames = [np.array(frame.convert("RGBA"), dtype=np.float32) for frame in ImageSequence.Iterator(gif)]
    frames = list(reversed(frames))

    # Use the first frame as the static background
    background = frames[0].copy()
    trail = background.copy()

    for frame in frames[1:]:
        # Compute pixel-wise difference from the background
        diff_mask = np.abs(frame[:, :, :3] - background[:, :, :3]).sum(axis=-1) > diff_threshold

        # Blend new moving pixels with previous ones
        trail[diff_mask] = (alpha * frame[diff_mask] + (1 - alpha) * trail[diff_mask])

    # Convert back to uint8 and save the result
    trail = np.clip(trail, 0, 255).astype(np.uint8)
    Image.fromarray(trail).save(output_path)


# Example usage
gif_to_trail_image("/home/charlie/Downloads/obs/out.gif", "/home/charlie/Downloads/obs/single_output.png")