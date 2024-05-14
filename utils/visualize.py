import math
import numpy as np

def generate_frames(generate_image_fn, initial_position, initial_orientation, num_frames, delta_degrees):
    """
    Generate multiple frames by panning the camera view.

    Args:
        generate_image_fn (function): A function that generates an RGB and depth image pair
            given a camera position and orientation.
        initial_position (list or np.array): The initial position of the camera in 3D space.
        initial_orientation (list or np.array): The initial orientation of the camera in Euler angles
            (roll, pitch, yaw) in degrees.
        num_frames (int): The number of frames to generate.
        delta_degrees (float): The number of degrees to pan the camera view between frames.

    Returns:
        list: A list of tuples, where each tuple contains the RGB and depth image pair for a frame.
    """
    frames = []
    position = np.array(initial_position)
    orientation = np.array(initial_orientation)

    for _ in range(num_frames):
        rgb_image, depth_image = generate_image_fn(position, orientation)
        frames.append((rgb_image, depth_image))

        # Update the camera orientation by panning the yaw angle
        orientation[2] += delta_degrees
        orientation[2] %= 360

    return frames