import copy

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(video_path):
    """Extract frames from the video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter % 10 == 0:   # removes frames
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            frames.append(frame)
        counter += 1
    cap.release()
    return frames

def get_mask(f1, f2_or_v2):
    return (f1 == f2_or_v2).all(axis=2)[:,:,np.newaxis]

def compute_diffs(frames):
    """Compute the difference between consecutive frames."""
    diffs = []
    for i in tqdm(range(1, len(frames))):
        #mask = (frames[i-1] == frames[i]).all(axis=2)#[:,:,None]

        mask = get_mask(frames[i-1], frames[i])

        #cv2.imshow("t", frames[i-1])
        #cv2.waitKey(0)
        #cv2.imshow("t", frames[i])
        #cv2.waitKey(0)

        diff = np.where(np.logical_not(mask), frames[i], 0) # transparent where frame is identical
        #diff = frames[i][mask]
        #cv2.imshow("t", diff)
        #cv2.waitKey(0)

        #diff = cv2.absdiff(frames[i - 1], frames[i])
        diffs.append(diff)
    return diffs


def remove_background(frames, diffs, threshold=0.9):
    """Remove background pixels from the diffs."""

    empty = np.zeros_like(frames[0])
    #voting = np.zeros(frames[0].shape)[:,:,0]
    i=0
    for  frame in tqdm(frames[:-1]):
        #frame = copy.deepcopy(frame)

        #mask = get_mask(diffs[i], 0)

        #frame = np.where(mask, frame, 0)
        #frame[np.lmask] = -1

        empty_needs_pixels = get_mask(empty, 0)
        no_diff = get_mask(diffs[i], 0)
        can_fill_pixel = np.logical_and(empty_needs_pixels, no_diff)

        empty = np.where(can_fill_pixel, frame, empty)
        i = i+1
        #voting += (empty == frames).astype(int)
    #voting /= len(frames)
    #voting = voting > 0.9

    #cv2.imshow("t", empty)
    #cv2.waitKey(0)

    processed_diffs = []
    for diff in tqdm(diffs):
        mask = get_mask(diff, empty)
        diff = np.where(mask, diff, 0)
        processed_diffs.append(diff)
    return processed_diffs



    return

    # Stack diffs to create a 3D array (height, width, num_frames)
    stacked_diffs = np.stack(diffs, axis=-1)

    # Calculate the percentage of frames where each pixel is different
    diff_percentage = np.mean(stacked_diffs > 0, axis=-1)

    # Create a mask for background pixels (pixels that are different in >90% of frames)
    background_mask = diff_percentage > threshold

    # Remove background pixels from each diff
    processed_diffs = []
    for diff in tqdm(diffs):
        diff[background_mask] = 0
        processed_diffs.append(diff)

    return processed_diffs


def combine_diffs(processed_diffs):
    """Combine the processed diffs into a single image."""
    combined_image = np.zeros_like(processed_diffs[0])
    for diff in tqdm(processed_diffs):
        combined_image = cv2.add(combined_image, diff)
    return combined_image


def overlay_image(base_image, overlay_image):
    """Overlay the combined image onto the first frame."""
    result = cv2.addWeighted(base_image, 1, overlay_image, 1, 0)
    return result


def main(video_path, output_path):
    # Step 1: Extract frames from the video
    frames = extract_frames(video_path)

    # Step 2: Compute the difference between consecutive frames
    diffs = compute_diffs(frames)

    # Step 3: Remove background pixels from the diffs
    processed_diffs = remove_background(frames, diffs)

    # Step 4: Combine the processed diffs into a single image
    combined_image = combine_diffs(processed_diffs)

    # Step 5: Overlay the combined image onto the first frame
    final_image = overlay_image(frames[0], combined_image)

    # Save the final image
    cv2.imwrite(output_path, final_image)


if __name__ == "__main__":
    video_path = "/home/charlie/Downloads/obs/temp.mkv"  # Replace with your video path
    output_path = "/home/charlie/Downloads/obs/output_image.png"  # Replace with your desired output path
    main(video_path, output_path)

# Example usage
#gif_path = "/home/charlie/Downloads/obs/output.gif" #"input.gif"  # Replace with your GIF file path
#output_path = "/home/charlie/Downloads/obs/single_output.png"  # Replace with your desired output file path
#gif_to_trace_image(gif_path, output_path)