import os

import cv2
import numpy as np
import torch

def generate_frames(generator, rgb_depth_pairs, initial_position, initial_orientation, num_frames, delta_degrees, device, output_dir):
    """
    Generate multiple frames by panning the camera view using the generator model.

    Args:
        generator (nn.Module): The trained generator model.
        rgb_depth_pairs (list): A list of tuples, where each tuple contains an RGB and depth image pair.
        initial_position (list or np.array): The initial position of the camera in 3D space.
        initial_orientation (list or np.array): The initial orientation of the camera in Euler angles
            (roll, pitch, yaw) in degrees.
        num_frames (int): The number of frames to generate.
        delta_degrees (float): The number of degrees to pan the camera view between frames.
        device (torch.device): The device to use for generating the frames.

    Returns:
        list: A list of numpy arrays, where each array represents a generated frame.
    """
    frames = []
    position = initial_position
    orientation = initial_orientation

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for frame_index in range(num_frames):
            frame_tensors = []
            for rgb_image, depth_image in rgb_depth_pairs:
                # Ensure input tensors have the correct shape
                if rgb_image.dim() == 3:  # Ensure 3 channels for RGB image
                    rgb_image = rgb_image.unsqueeze(0)
                if depth_image.dim() == 2:  # Ensure single channel for depth image
                    depth_image = depth_image.unsqueeze(0)

                rgb_image = rgb_image.to(device)
                depth_image = depth_image.to(device)
                generated_scene = generator(rgb_image, depth_image)
                frame_tensors.append(generated_scene)

            batch_frames = torch.cat(frame_tensors, dim=0)
            frames.extend([frame.permute(1, 2, 0).cpu().numpy() for frame in batch_frames])

            # Save the generated image
            frame_image = np.uint8(batch_frames[0].permute(1, 2, 0).cpu().numpy() * 255)
            output_path = os.path.join(output_dir, f"frame_{frame_index}.png")
            cv2.imwrite(output_path, frame_image)

            # Update the camera orientation by panning the yaw angle
            orientation[2] += delta_degrees
            orientation[2] %= 360  # Keep angle within range [0, 360)

        return frames