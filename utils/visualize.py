# import os
# import cv2
# import numpy as np
# import torch
# from torchvision.transforms.functional import affine
#
#
# def generate_frames(generator, rgb_depth_pair, initial_position, initial_orientation, num_frames, delta_degrees, device,
#                     save_dir):
#     """
#     Generate multiple frames by panning the camera view using the generator model.
#
#     Args:
#         generator (nn.Module): The trained generator model.
#         rgb_depth_pair (tuple): A tuple containing an RGB and depth image pair.
#         initial_position (list or np.array): The initial position of the camera in 3D space.
#         initial_orientation (list or np.array): The initial orientation of the camera in Euler angles
#             (roll, pitch, yaw) in degrees.
#         num_frames (int): The number of frames to generate.
#         delta_degrees (float): The incremental angle to pan the camera view between frames.
#         device (torch.device): The device to use for generating the frames.
#         save_dir (str): The directory to save the generated frames.
#
#     Returns:
#         list: A list of numpy arrays, where each array represents a generated frame.
#     """
#     rgb_image, depth_image = rgb_depth_pair
#     if rgb_image.dim() == 3:  # Ensure 3 channels for RGB image
#         rgb_image = rgb_image.unsqueeze(0)
#     if depth_image.dim() == 2:  # Ensure single channel for depth image
#         depth_image = depth_image.unsqueeze(0)
#
#     # Convert initial orientation to radians
#     initial_orientation = np.radians(initial_orientation)
#     orientation = initial_orientation.copy()
#
#     frames = []
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     with torch.no_grad():
#         for frame_index in range(num_frames):
#             rgb_image = rgb_image.to(device)
#             depth_image = depth_image.to(device)
#             generated_scene = generator(rgb_image, depth_image).cpu().numpy()
#
#             # Update the camera orientation by panning the yaw angle
#             orientation[2] += np.radians(delta_degrees)
#             orientation[2] %= 2 * np.pi
#
#             frame_image = np.uint8(generated_scene[0].transpose(1, 2, 0) * 255)
#             output_path = os.path.join(save_dir, f"frame_{frame_index}.png")
#             cv2.imwrite(output_path, frame_image)
#
#
#             # Apply affine transformation to simulate camera pan
#             angle = np.degrees(orientation[2])  # Convert yaw angle to degrees
#             transform_matrix = cv2.getRotationMatrix2D((frame_image.shape[1] / 2, frame_image.shape[0] / 2), angle, 1)
#             frame_image = cv2.warpAffine(frame_image, transform_matrix, (frame_image.shape[1], frame_image.shape[0]))
#             frames.append(frame_image)
#
#
#
#     return frames

import os

import cv2
import numpy as np
import torch

def generate_frames(generator, rgb_depth_pair,  initial_orientation, num_frames, delta_degrees, device, save_dir, attention=True):
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
    rgb_image, depth_image = rgb_depth_pair
    if rgb_image.dim() == 3:  # Ensure 3 channels for RGB image
        rgb_image = rgb_image.unsqueeze(0)
    if depth_image.dim() == 2:  # Ensure single channel for depth image
        depth_image = depth_image.unsqueeze(0)
    orientation = initial_orientation
    frames = []
    warped_frames = []


    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for frame_index in range(num_frames):
            rgb_image = rgb_image.to(device)
            depth_image = depth_image.to(device)
            generated_scene = generator(rgb_image, depth_image).cpu().numpy()

            frame_image = np.uint8(generated_scene[0].transpose(1, 2, 0) * 255)
            # output_path = os.path.join(save_dir, f"frame_{frame_index}.png")
            # cv2.imwrite(output_path, frame_image)
            frames.append(frame_image)

            # Update the camera orientation by panning the yaw angle
            orientation[2] += delta_degrees
            orientation[2] %= 360

            angle = np.degrees(orientation[2])  # Convert yaw angle to degrees
            transform_matrix = cv2.getRotationMatrix2D((frame_image.shape[1] / 2, frame_image.shape[0] / 2), angle, 1)
            warped_frame = cv2.warpAffine(frame_image, transform_matrix, (frame_image.shape[1], frame_image.shape[0]))
            warped_frames.append(warped_frame)

    return frames, warped_frames